"""Resolve the content-addressed image data referenced by session journals."""

import base64
import re
import stat
from hashlib import sha256

from session_artifacts import file_mode


BLOB_HASH = re.compile(r"[a-f0-9]{64}")
BLOB_REF = re.compile(r"blob:sha256:([a-f0-9]{64})")
JOURNAL_BLOB = re.compile(rb"\bblob:sha256:([a-f0-9]{64})\b")


def blob_path(directories, digest):
    if not BLOB_HASH.fullmatch(digest):
        return None
    for directory in directories:
        path = directory / digest
        if stat.S_ISREG(file_mode(path)):
            with path.open("rb") as file:
                if sha256_file(file) == digest:
                    return path
    return None


def sha256_file(file):
    digest = sha256()
    while chunk := file.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def references(journal):
    with journal.open("rb") as file:
        return {digest.decode("ascii") for line in file for digest in JOURNAL_BLOB.findall(line)}


def hydrate_images(value, directories, cache, position=None):
    if isinstance(value, list):
        for child in value:
            hydrate_images(child, directories, cache, position)
    if not isinstance(value, dict):
        return
    kind = value.get("type")
    mime = value.get("mimeType")
    image_data = kind == "image" or isinstance(mime, str) and mime.startswith("image/")
    fields = []
    if image_data and position in ("content", "images", "frames"):
        fields.append(("data", False))
    if kind == "image_generation_call":
        fields.append(("result", False))
    fields.append(("image_url", True))
    if position == "image_url":
        fields.append(("url", True))
    for key, data_url in fields:
        ref = value.get(key)
        if not isinstance(ref, str) or not (match := BLOB_REF.fullmatch(ref)):
            continue
        address = (match[1], data_url)
        if address not in cache:
            path = blob_path(directories, match[1])
            cache[address] = None
            if path:
                data = path.read_bytes()
                if sha256(data).hexdigest() == match[1]:
                    cache[address] = data.decode("utf-8", errors="replace") if data_url else base64.b64encode(data).decode("ascii")
        if cache[address] is not None:
            value[key] = cache[address]
    for key, child in value.items():
        if isinstance(child, (dict, list)):
            hydrate_images(child, directories, cache, key)
