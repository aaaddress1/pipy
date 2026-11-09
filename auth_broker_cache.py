"""Encrypted v2 snapshots shared with the Oh My Pi Auth Broker client."""

import asyncio
import json
import logging
import math
import os
import re
import secrets
import tempfile
from hashlib import sha256
from pathlib import Path
from time import monotonic, time

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


HEADER = b"OMPS\x02"
NONCE_SIZE = 12
MAX_BYTES = 32 * 1024 * 1024
DEFAULT_TTL_MS = 3_600_000
REMOTE_REFRESH = "__remote__"
logger = logging.getLogger(__name__)


def number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def config_root(profile):
    root = Path.home() / Path(os.getenv("PI_CONFIG_DIR") or ".omp").expanduser()
    if profile == "default":
        return root
    if (not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", profile) or profile.endswith(".")
        or re.fullmatch(r"(?:con|prn|aux|nul|com[0-9]|lpt[0-9])(?:\..*)?", profile)):
        raise ValueError("Invalid Oh My Pi profile name.")
    return root / "profiles" / profile


def snapshot_path(profile):
    override = os.getenv("OMP_AUTH_BROKER_SNAPSHOT_CACHE")
    if override:
        return Path(override).expanduser()
    root = config_root(profile)
    xdg = os.getenv("XDG_CACHE_HOME")
    if os.name != "nt" and xdg and (profile != "default" or not os.getenv("PI_CODING_AGENT_DIR")):
        candidate = Path(xdg) / "omp"
        if profile != "default":
            candidate = candidate / "profiles" / profile
        if candidate.is_dir():
            root = candidate
    return root / "cache" / "auth-broker-snapshot.enc"


def snapshot_ttl():
    raw = os.getenv("OMP_AUTH_BROKER_SNAPSHOT_TTL_MS", "").strip()
    if raw:
        try:
            value = float(raw)
            if number(value) and value >= 0:
                return value
        except ValueError:
            pass
        logger.debug("Invalid Auth Broker snapshot TTL; using default.")
    return DEFAULT_TTL_MS


def valid_snapshot(document):
    if (not isinstance(document, dict) or type(document.get("generation")) is not int
        or document["generation"] < 0 or not number(document.get("generatedAt"))
        or not number(document.get("serverNowMs")) or not isinstance(document.get("refresher"), dict)
        or not isinstance(document.get("credentials"), list)):
        return False
    for row in document["credentials"]:
        if (not isinstance(row, dict) or type(row.get("id")) is not int
            or not isinstance(row.get("provider"), str) or not isinstance(row.get("credential"), dict)):
            return False
        credential = row["credential"]
        if credential.get("type") == "oauth":
            if (credential.get("refresh") != REMOTE_REFRESH or not isinstance(credential.get("access"), str)
                or not credential["access"] or not number(credential.get("expires"))):
                return False
        elif credential.get("type") != "api_key" or not isinstance(credential.get("key"), str):
            return False
    return True


def cache_signature(document):
    entries = [{key: value for key, value in row.items() if key != "rotatesInMs"} for row in document["credentials"]]
    return sha256(json.dumps([document["generation"], entries], sort_keys=True).encode()).digest()


class SnapshotCache:
    def __init__(self, path, token, url, ttl_ms=DEFAULT_TTL_MS):
        self.path, self.ttl_ms = path, ttl_ms
        self.cipher = AESGCM(sha256(token.encode()).digest())
        self.aad = HEADER + url.encode()
        self.pending = self.task = self.written = None
        self.started_at = float("-inf")
        self.flush = asyncio.Event()
        self.swept = False

    @classmethod
    def configured(cls, token, url, profile):
        ttl = snapshot_ttl()
        return cls(snapshot_path(profile), token, url, ttl) if ttl > 0 else None

    def read(self):
        if self.ttl_ms <= 0:
            return None
        try:
            with self.path.open("rb") as file:
                payload = file.read(MAX_BYTES + 1)
            start = len(HEADER)
            end = start + NONCE_SIZE
            if not payload.startswith(HEADER) or not end + 16 < len(payload) <= MAX_BYTES:
                return None
            document = json.loads(self.cipher.decrypt(payload[start:end], payload[end:], self.aad))
            if valid_snapshot(document):
                age = time() * 1000 - document["generatedAt"]
                if age <= self.ttl_ms:
                    if age > 0:
                        for row in document["credentials"]:
                            if number(remaining := row.get("rotatesInMs")):
                                row["rotatesInMs"] = max(0, remaining - age)
                    return document
        except (OSError, ValueError, InvalidTag, RecursionError):
            pass
        return None

    def write(self, document):
        if self.ttl_ms <= 0 or not valid_snapshot(document):
            return
        nonce = secrets.token_bytes(NONCE_SIZE)
        plaintext = json.dumps(document, separators=(",", ":"), allow_nan=False).encode()
        if len(plaintext) + len(HEADER) + NONCE_SIZE + 16 > MAX_BYTES:
            return
        payload = HEADER + nonce + self.cipher.encrypt(nonce, plaintext, self.aad)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f"{self.path.name}.{os.getpid()}.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "wb") as file:
                file.write(payload)
            os.replace(temporary, self.path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        if not self.swept:
            self.swept = True
            cutoff = time() - 3600
            pattern = re.compile(re.escape(self.path.name) + r"\.\d+\.[\w-]+\.tmp\Z")
            for path in self.path.parent.iterdir():
                if pattern.fullmatch(path.name):
                    try:
                        if path.stat().st_mtime <= cutoff:
                            path.unlink(missing_ok=True)
                    except OSError:
                        pass

    def schedule(self, document):
        if self.ttl_ms <= 0 or not valid_snapshot(document):
            return
        self.pending = document
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self.persist())

    async def persist(self):
        while self.pending is not None:
            delay = self.started_at + 2 - monotonic()
            if delay > 0 and not self.flush.is_set():
                try:
                    await asyncio.wait_for(self.flush.wait(), delay)
                except TimeoutError:
                    pass
            document, self.pending = self.pending, None
            try:
                signature = await asyncio.to_thread(cache_signature, document)
                generated = document["generatedAt"]
                if self.written and signature == self.written[0] and generated - self.written[1] < 60_000:
                    continue
                self.started_at = monotonic()
                await asyncio.to_thread(self.write, document)
            except (OSError, ValueError, RecursionError):
                logger.debug("Could not write encrypted Auth Broker snapshot.")
            else:
                self.written = signature, generated

    async def close(self):
        self.flush.set()
        if self.task:
            await self.task
