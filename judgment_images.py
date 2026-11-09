"""Image inputs and provider bounds for eval judgments."""

import base64
import gzip
import math
from io import BytesIO
from pathlib import Path

from PIL import Image

from model_catalog import JUDGE_APIS, model_policy
from snapcompact_inline import excludes_webp


MAX_INPUT_BYTES = 20 * 1024 * 1024
MIME = {"PNG": "image/png", "JPEG": "image/jpeg", "GIF": "image/gif", "WEBP": "image/webp"}
NATIVE_LIMITS = {
    "typesafe": {"mimeTypes": ("image/png", "image/jpeg")},
    "openai-decisions": {"maxCount": 128},
    "openrouter-decisions": {},
    "workers-ai-systemone": {"mimeTypes": ("image/png", "image/jpeg", "image/webp"), "maxCount": 4,
                            "maxBytes": 4 * 1024**2, "maxTotalBytes": 8 * 1024**2, "maxPixels": 16_000_000},
}


def load_images(value, cwd, cancellation):
    if not isinstance(value, list):
        raise ValueError("invalid judgment image source: images must be an array of paths or image attachments")
    images = []
    for index, source in enumerate(value):
        cancellation.check()
        try:
            detail = None
            if isinstance(source, str) and source:
                path = cwd / Path(source).expanduser()
                if not path.is_file() or path.stat().st_size > MAX_INPUT_BYTES:
                    raise ValueError("expected a regular image file within the size limit")
                with path.open("rb") as file:
                    data = file.read(MAX_INPUT_BYTES + 1)
                if path.suffix.lower() in (".svg", ".svgz"):
                    from resvg_py import svg_to_bytes

                    if path.suffix.lower() == ".svgz":
                        with gzip.GzipFile(fileobj=BytesIO(data)) as file:
                            data = file.read(MAX_INPUT_BYTES + 1)
                    if len(data) > MAX_INPUT_BYTES:
                        raise ValueError("SVG input exceeds the size limit")
                    data = bytes(svg_to_bytes(svg_string=data.decode("utf-8-sig"), width=2048, height=2048))
                expected = None
            elif (isinstance(source, dict) and source.get("type") == "image"
                  and isinstance(source.get("data"), str) and source.get("mimeType") in MIME.values()):
                detail = source.get("detail")
                if "detail" in source and detail not in ("auto", "low", "high", "original"):
                    raise ValueError("unsupported image detail")
                if len(source["data"]) > (MAX_INPUT_BYTES + 2) // 3 * 4:
                    raise ValueError("image attachment exceeds the size limit")
                data = base64.b64decode(source["data"], validate=True)
                expected = source["mimeType"]
            else:
                raise ValueError("expected a local path or an embedded image attachment")
            if not data or len(data) > MAX_INPUT_BYTES:
                raise ValueError("image is empty or exceeds the size limit")
            with Image.open(BytesIO(data)) as image:
                mime = MIME.get(image.format)
                if not mime or expected and mime != expected:
                    raise ValueError("unsupported image format or MIME type mismatch")
                image.verify()
        except (OSError, ValueError, RuntimeError, SyntaxError, Image.DecompressionBombError) as error:
            raise ValueError(f"invalid judgment image source: images[{index}] {error}") from error
        images.append({"type": "image", "mimeType": mime, "data": base64.b64encode(data).decode("ascii"),
                       **({"detail": detail} if detail is not None else {})})
    return images


def policy_for(route):
    model = route.catalog_model
    return model_policy(model) if model else {}


def limits_for(route):
    if route.api not in JUDGE_APIS:
        mime = tuple(value for value in MIME.values() if value != "image/webp" or not excludes_webp(route))
        return {"minSide": 200, "maxSide": min(1568, route.image_max_edge or 1568), "maxBytes": 500 * 1024,
                "mimeTypes": mime}
    limits = dict(NATIVE_LIMITS[route.api])
    policy = policy_for(route).get("judgmentImageInput") or {}
    for key, value in policy.items():
        if key == "mimeTypes":
            allowed = value.split(",")
            limits[key] = tuple(mime for mime in limits.get(key, allowed) if mime in allowed)
        elif key.startswith("min"):
            limits[key] = max(value, limits.get(key, value))
        else:
            limits[key] = min(value, limits.get(key, value))
    return limits


def data_url(image):
    return f"data:{image['mimeType']};base64,{image['data']}"


def image_detail(image, *, original=True):
    detail = image.get("detail")
    return {} if detail is None else {"detail": "auto" if detail == "original" and not original else detail}


def chat_content(text, images):
    if not images:
        return text
    return [{"type": "text", "text": text}, *[
        {"type": "image_url", "image_url": {"url": data_url(image), **image_detail(image)}}
        for image in images
    ]]


def fit_images(images, limits):
    if not images:
        return []
    cap = min(limits.get("maxBytes", math.inf), limits.get("maxTotalBytes", math.inf) // len(images)
              if "maxTotalBytes" in limits else math.inf)
    return [fit(image, limits, cap) for image in images]


def fit(image, limits, cap):
    raw = base64.b64decode(image["data"])
    accepted = limits.get("mimeTypes", MIME.values())
    min_side, max_side = limits.get("minSide", 1), limits.get("maxSide", math.inf)
    min_pixels, max_pixels = limits.get("minPixels", 1), limits.get("maxPixels", math.inf)
    with Image.open(BytesIO(raw)) as source:
        width, height = source.size
        if (min_side <= min(width, height) <= max(width, height) <= max_side
                and min_pixels <= width * height <= max_pixels and len(raw) <= cap and image["mimeType"] in accepted):
            return image
        scale = min(max(1, min_side / min(width, height), math.sqrt(min_pixels / (width * height))),
                    max_side / max(width, height), math.sqrt(max_pixels / (width * height)))
        width, height = (min(max_side, max(1, math.floor(side * scale + .5))) for side in (width, height))
        if width * height > max_pixels:
            if width >= height:
                width = max_pixels // height
            else:
                height = max_pixels // width
        alpha = "A" in source.getbands() or "transparency" in source.info
        formats = ("PNG", "WEBP", "JPEG") if alpha else ("JPEG", "WEBP", "PNG")
        formats = [fmt for fmt in formats if MIME[fmt] in accepted]
        if not formats:
            raise ValueError("judgment image has no supported output format")
        pixels = source.convert("RGBA" if alpha else "RGB")
        while True:
            canvas_width, canvas_height = max(width, min_side), max(height, min_side)
            if canvas_width * canvas_height < min_pixels:
                if canvas_width >= canvas_height:
                    canvas_height = math.ceil(min_pixels / canvas_width)
                else:
                    canvas_width = math.ceil(min_pixels / canvas_height)
            if max(canvas_width, canvas_height) > max_side or canvas_width * canvas_height > max_pixels:
                raise ValueError("judgment image cannot fit the model's size bounds")
            resized = pixels.resize((width, height), Image.Resampling.LANCZOS)
            if resized.size != (canvas_width, canvas_height):
                canvas = Image.new(resized.mode, (canvas_width, canvas_height), resized.getpixel((0, 0)))
                canvas.paste(resized, ((canvas_width - width) // 2, (canvas_height - height) // 2))
                resized = canvas
            for fmt in formats:
                encoded = BytesIO()
                output = resized.convert("RGB") if fmt == "JPEG" else resized
                for quality in (80, 60, 40, 20) if fmt != "PNG" else (80,):
                    encoded.seek(0)
                    encoded.truncate()
                    output.save(encoded, format=fmt, quality=quality, optimize=True)
                    if encoded.tell() <= cap:
                        return {**image, "mimeType": MIME[fmt], "data": base64.b64encode(encoded.getvalue()).decode("ascii")}
            if max(width, height) <= min_side:
                raise ValueError("judgment image cannot fit the model's byte limit")
            width, height = max(1, width * 3 // 4), max(1, height * 3 // 4)
