"""Just-in-time Apple DeviceCheck envelopes for official Codex OAuth requests."""

import asyncio
import ctypes
import json
import platform
import struct
import sys
from base64 import urlsafe_b64encode
from functools import cache
from threading import Event
from time import perf_counter
from uuid import uuid4

import httpx

from local_oauth import jwt_payload


HEADER = "x-oai-attestation"
BUNDLE_ID = "com.openai.codex"
APP_SESSION_ID = str(uuid4())
SECURITY_FRAMEWORK = "/System/Library/Frameworks/Security.framework/Security"
CALLER_SECURITY_SESSION = 0xFFFFFFFF
SESSION_HAS_GRAPHIC_ACCESS = 0x0010


def cbor_header(major, size):
    if size < 24:
        return bytes((major + size,))
    if size <= 0xFF:
        return bytes((major + 24, size))
    if size <= 0xFFFF:
        return bytes((major + 25,)) + struct.pack(">H", size)
    return bytes((major + 26,)) + struct.pack(">I", size)


def cbor(value):
    if isinstance(value, int):
        return cbor_header(0, value)
    if isinstance(value, float):
        return b"\xfb" + struct.pack(">d", value)
    if isinstance(value, str):
        text = value.encode("utf-8")
        return cbor_header(0x60, len(text)) + text
    if isinstance(value, bytes):
        return cbor_header(0x40, len(value)) + value
    if isinstance(value, list):
        return cbor_header(0x80, len(value)) + b"".join(map(cbor, value))
    if isinstance(value, dict):
        return cbor_header(0xA0, len(value)) + b"".join(cbor(key) + cbor(item) for key, item in value.items())
    raise TypeError("Unsupported DeviceCheck CBOR value.")


def envelope(supported, token, latency_ms, locale, timezone):
    locale, timezone = (locale or "unknown")[:64], (timezone or "unknown")[:64]
    signals = {0: 1, 1: [locale], 2: locale, 3: timezone, 4: 0, 5: 1, 6: APP_SESSION_ID}
    result = {"token": token} if supported and token else {"error_code": 4 if supported else 3}
    result.update(bundle_id=BUNDLE_ID, f=cbor(signals))
    if latency_ms is not None:
        result["t"] = float(latency_ms)
    encoded = urlsafe_b64encode(cbor(result)).decode("ascii").rstrip("=")
    return json.dumps({"v": 1, "s": 0, "t": "v1." + encoded}, separators=(",", ":"))


@cache
def session_get_info():
    method = ctypes.CDLL(SECURITY_FRAMEWORK).SessionGetInfo
    integer, pointer = ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)
    method.argtypes = (integer, pointer, pointer)
    method.restype = ctypes.c_int32
    return method


def has_graphic_access():
    attributes = ctypes.c_uint32()
    try:
        status = session_get_info()(CALLER_SECURITY_SESSION, None, ctypes.byref(attributes))
    except (OSError, AttributeError):
        return False
    return status == 0 and bool(attributes.value & SESSION_HAS_GRAPHIC_ACCESS)


def device_token():
    # Querying DCDevice outside a GUI session can terminate the entire process.
    started = perf_counter()
    if not has_graphic_access():
        return False, None, (perf_counter() - started) * 1000

    from DeviceCheck import DCDevice
    from objc import autorelease_pool

    result = []
    ready = Event()

    def completed(token, error):
        try:
            if token is not None:
                encoded = token.base64EncodedStringWithOptions_(0)
                if encoded:
                    result.append(str(encoded))
        except Exception:
            pass
        finally:
            ready.set()

    with autorelease_pool():
        device = DCDevice.currentDevice()
        supported = device is not None and bool(device.isSupported())
        token = None
        if supported:
            device.generateTokenWithCompletionHandler_(completed)
            if ready.wait(1):
                token = next(iter(result), None)
    return supported, token, (perf_counter() - started) * 1000


def generate():
    try:
        from Foundation import NSLocale, NSTimeZone
        from objc import autorelease_pool

        with autorelease_pool():
            locale = str(NSLocale.currentLocale().localeIdentifier() or "").split("@", 1)[0].replace("_", "-")
            timezone = str(NSTimeZone.localTimeZone().name() or "")
        return envelope(*device_token(), locale, timezone)
    except Exception:
        return None


def native_platform():
    return sys.platform == "darwin" and platform.machine() == "arm64"


def eligible(headers, url):
    # Account proof must never follow a Codex-compatible proxy or caller-supplied header.
    headers.pop(HEADER, None)
    url = httpx.URL(url)
    if (not native_platform() or url.scheme != "https" or url.host != "chatgpt.com" or url.port not in (None, 443)
        or url.userinfo or url.path not in ("/backend-api/codex/responses", "/backend-api/codex/responses/compact")):
        return False
    scheme, _, token = headers.get("authorization", "").partition(" ")
    claims = jwt_payload(token).get("https://api.openai.com/auth") if scheme.lower() == "bearer" else None
    return isinstance(claims, dict) and isinstance(claims.get("chatgpt_account_id"), str) and bool(claims["chatgpt_account_id"])


def authorize_request(request):
    if request.method != "POST":
        request.headers.pop(HEADER, None)
        return
    if eligible(request.headers, request.url) and (proof := generate()):
        request.headers[HEADER] = proof


async def upgrade_headers(headers, url):
    headers = httpx.Headers(headers)
    if eligible(headers, url) and (proof := await asyncio.to_thread(generate)):
        headers[HEADER] = proof
    return list(headers.items())
