"""Snapcompact grid planning and PNG rasterization using the upstream fonts."""

import re
from dataclasses import dataclass, replace
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont


FONTS = Path(__file__).with_name("snapcompact_fonts")
DIM_ON, DIM_OFF, NEWLINE = "\x0e", "\x0f", "█"
PALETTE = (
    (255, 255, 255), (109, 2, 2), (109, 53, 2), (24, 109, 2), (2, 109, 109),
    (2, 32, 109), (75, 2, 109), (0, 0, 0), (255, 247, 194), (128, 128, 128),
)
GUTTER = 3
WIDE_RANGES = (
    (0x1100, 0x115F), (0x2E80, 0x2EFF), (0x2F00, 0x2FDF), (0x3000, 0x303E),
    (0x3041, 0x33FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xA000, 0xA4CF),
    (0xAC00, 0xD7A3), (0xF900, 0xFAFF), (0xFE30, 0xFE4F), (0xFF00, 0xFF60),
    (0xFFE0, 0xFFE6), (0x20000, 0x2FFFD), (0x30000, 0x3FFFD),
)
STOPWORDS = frozenset((
    "the a an and or of to in on at as is are was were be been by for with that this it its from had has have not "
    "but he she his her they their them which also who whom when where while will would could should there then "
    "than into over under about after before between during each such these those some most more other only same so"
).split())


@dataclass(frozen=True)
class Shape:
    font: str
    width: int
    height: int
    size: int = 1568
    stretch: bool = False
    colored: bool = False
    dimmed: bool = False
    repeat: int = 1
    columns: int = 1

    @property
    def grid_columns(self):
        return self.size // self.width

    @property
    def cols(self):
        return (self.grid_columns - GUTTER) // 2 if self.columns == 2 else self.grid_columns

    @property
    def rows(self):
        return self.size // self.height // self.repeat

    @property
    def capacity(self):
        return self.cols * self.rows * self.columns


SHAPES = {
    "8x8r-bw": Shape("8x8", 8, 8, repeat=2),
    "8x8u-bw": Shape("8x8", 8, 8),
    "6x6u-bw": Shape("8x8", 6, 6, stretch=True),
    "5x8-bw": Shape("5x8", 5, 8, size=2576),
    "6x12-dim": Shape("6x12", 6, 12, dimmed=True),
    "8x13-bw": Shape("8x13", 8, 13),
    "8on16-bw": Shape("8x13", 8, 16),
    "8on22-bw": Shape("8x13", 8, 22),
    "11on16-bw": Shape("8x13", 11, 16),
    "silver16-bw": Shape("silver", 16, 16),
    "doc-8on16-bw": Shape("8x13", 8, 16, columns=2),
}
for variant in ("8x8r", "8x8u", "6x6u", "5x8", "doc-8on16"):
    SHAPES[f"{variant}-sent"] = replace(SHAPES[f"{variant}-bw"], colored=True)
SHAPES["doc-8on16-sent-dim"] = replace(SHAPES["doc-8on16-sent"], dimmed=True)


@lru_cache(maxsize=1)
def silver_coverage():
    with TTFont(FONTS / "Silver.ttf", lazy=True) as font:
        return frozenset(font.getBestCmap())


@lru_cache(maxsize=4)
def bitmap_font(name):
    glyphs = {}
    if name == "8x8":
        for line in (FONTS / "unscii-8.hex").read_text(encoding="ascii").splitlines():
            code, _, bitmap = line.partition(":")
            if len(bitmap) == 16:
                glyphs[int(code, 16)] = (0, 0, Image.frombytes("1", (8, 8), bytes.fromhex(bitmap)))
        return 7, glyphs

    lines = iter((FONTS / f"{name}.bdf").read_text(encoding="utf-8").splitlines())
    ascent, code = 0, -1
    for line in lines:
        key, _, value = line.partition(" ")
        if key == "FONT_ASCENT":
            ascent = int(value)
        elif key == "ENCODING":
            code = int(value.split()[0])
        elif key == "BBX":
            width, height, left, bottom = map(int, value.split())
        elif key == "BITMAP":
            bitmap = bytearray()
            for row in lines:
                if row == "ENDCHAR":
                    break
                bitmap.append(int(row, 16))
            if code >= 0 and width > 0 and height > 0:
                glyph = Image.frombytes("1", (min(width, 8), height), bytes(bitmap))
                glyphs[code] = (left, ascent - height - bottom, glyph)
    return ascent, glyphs


@lru_cache(maxsize=16384)
def is_wide(char):
    code = ord(char)
    return code >= 0x1100 and any(lo <= code <= hi for lo, hi in WIDE_RANGES)


def cells(char, shape):
    if char in (DIM_ON, DIM_OFF):
        return 0
    return 2 if shape.font != "silver" and is_wide(char) else 1


def wrap(text, shape):
    lines = []
    line, used = "", 0
    for word in text.split():
        count = sum(cells(char, shape) for char in word)
        if line and used + 1 + count > shape.cols:
            lines.append(line)
            line, used = "", 0
        while count > shape.cols:
            end, size = 0, 0
            while end < len(word) and size + cells(word[end], shape) <= shape.cols:
                size += cells(word[end], shape)
                end += 1
            lines.append(word[:end])
            word, count = word[end:], count - size
        if word:
            line, used = (line + " " + word, used + 1 + count) if line else (word, count)
    if line:
        lines.append(line)
    return lines


def paginate(text, shape):
    if shape.columns == 2:
        lines = wrap(text, shape)
        step = shape.rows * 2
        return ["\n".join(lines[index:index + step]) for index in range(0, len(lines), step)]

    pages = []
    start, used = 0, 0
    for index, char in enumerate(text):
        count = cells(char, shape)
        if not count:
            continue
        pad = count == 2 and used % shape.cols == shape.cols - 1
        if used and used + pad + count > shape.capacity:
            pages.append(text[start:index])
            start, used, pad = index, 0, False
        used += pad + count
    if used:
        pages.append(text[start:])
    return pages


def layout(text, shape):
    dim = False
    cursor, line, sentence = 0, 0, 0
    for index, char in enumerate(text):
        if char in (DIM_ON, DIM_OFF):
            dim = char == DIM_ON
            continue
        if shape.columns == 2 and char == "\n":
            line, cursor = line + 1, 0
            if line >= shape.rows * 2:
                break
            continue
        width = cells(char, shape)
        if width == 2 and cursor % shape.cols == shape.cols - 1:
            cursor += 1
        row, column = divmod(cursor, shape.cols)
        cursor += width
        if shape.columns == 2:
            if row:
                continue
            side, row = divmod(line, shape.rows)
            column += side * (shape.cols + GUTTER)
        elif row >= shape.rows:
            break
        ink = 9 if dim else (1 + sentence % 6 if shape.colored else 7)
        yield char, ink, row, column, width
        if char in ".!?" and text[index + 1:index + 2] in (" ", NEWLINE, "\n"):
            sentence += 1


def dim_stopwords(text):
    parts = re.split(r"([\x0e\x0f])", text)
    for index in range(0, len(parts), 2):
        if index and parts[index - 1] == DIM_ON:
            continue
        parts[index] = re.sub(
            r"[a-zA-ZÀ-ÖØ-öø-ÿ]+",
            lambda match: DIM_ON + match[0] + DIM_OFF if match[0].lower() in STOPWORDS else match[0],
            parts[index],
        )
    return "".join(parts)


def render(text, shape):
    if shape.dimmed:
        text = dim_stopwords(text)
    placed = list(layout(text, shape))
    used = 1 + max((row for _, _, row, _, _ in placed), default=0)
    out_height = max(64, used * shape.repeat * shape.height)
    indexed = shape.font != "silver"
    width, height = map(int, shape.font.split("x")) if shape.stretch else (shape.width, shape.height)
    working_size = shape.grid_columns * width if shape.stretch else shape.size
    working_height = used * shape.repeat * height if shape.stretch else out_height
    frame = Image.new("P" if indexed else "RGB", (working_size, working_height), 0 if indexed else PALETTE[0])
    if indexed:
        frame.putpalette(bytes(channel for color in PALETTE for channel in color))
        ascent, bitmap = bitmap_font(shape.font)
    else:
        ascent, bitmap = height - 2, {}
    draw = ImageDraw.Draw(frame)
    for row in range(used):
        for copy in range(1, shape.repeat):
            y = (row * shape.repeat + copy) * height
            draw.rectangle((0, y, working_size - 1, y + height - 1), fill=8 if indexed else PALETTE[8])

    silver = {}
    for char, ink, row, column, units in placed:
        code = ord(char)
        glyph = bitmap.get(code)
        color = ink if indexed else PALETTE[ink]
        x = column * width
        if glyph:
            left, top, mask = glyph
        elif char not in (NEWLINE, " ") and code in silver_coverage():
            px = min(height, width * units, 16)
            if px not in silver:
                silver[px] = ImageFont.truetype(FONTS / "Silver.ttf", px)
            raster, (left, top) = silver[px].getmask2(char, mode="L", anchor="ls")
            mask = Image.frombytes("L", raster.size, bytes(raster))
            left = (width * units - mask.width) // 2
            top += ascent
        elif char != NEWLINE:
            continue
        for copy in range(shape.repeat):
            y = (row * shape.repeat + copy) * height
            if char == NEWLINE:
                draw.rectangle((x, y, x + width - 1, y + height - 1), fill=7 if indexed else PALETTE[7])
            elif mask.width and mask.height:
                box = (x + left, y + top, x + left + mask.width, y + top + mask.height)
                if indexed and mask.mode == "L":
                    if ink == 7:
                        frame.paste(9, box, mask.point(lambda value: 255 if value >= 56 else 0, "1"))
                    frame.paste(ink, box, mask.point(lambda value: 255 if value >= (170 if ink == 7 else 110) else 0, "1"))
                else:
                    frame.paste(color, box, mask)

    if shape.stretch:
        stretched = frame.convert("RGB").resize(
            (shape.grid_columns * shape.width, used * shape.repeat * shape.height), Image.Resampling.LANCZOS,
        )
        frame = Image.new("RGB", (shape.size, out_height), PALETTE[0])
        frame.paste(stretched)
    out = BytesIO()
    png_options = {"bits": 4} if frame.mode == "P" else {}
    frame.save(out, "PNG", compress_level=9, **png_options)
    return out.getvalue(), out_height, used
