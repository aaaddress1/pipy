"""Offline BPE tokenizers using the pinned Oh My Pi rank tables."""

import unicodedata
from functools import cache
from pathlib import Path

import regex


TABLES = Path(__file__).with_name("catalog")
CL100K = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|"
          r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
O200K = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
QWEN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|"
        r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
KIMI = r"[\p{Han}]+|" + O200K.replace(
    r"\p{Lo}\p{M}]", r"\p{Lo}\p{M}&&[^\p{Han}]]",
).replace(r"[\r\n/]*", r"[\r\n]*")
DEEPSEEK_STAGES = tuple(map(regex.compile, (
    r"\p{N}{1,3}", r"[一-龥぀-ゟ゠-ヿ]+",
    r'''[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+|'''
    r" ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
)))
BPE = {
    "cl100k": ("cl100k_base", CL100K), "o200k": ("o200k_base", O200K),
    "qwen3": ("qwen3", QWEN), "deepseek-v3": ("deepseek3", r"(?s:.+)"),
    "kimi-k2": ("kimi_k2", KIMI), "glm5": ("glm5", CL100K),
}


def read_table(name):
    from zstandard import ZstdDecompressor

    with (TABLES / f"{name}.bin.zst").open("rb") as file, ZstdDecompressor().stream_reader(file) as reader:
        return reader.read()


def varint(raw, offset):
    number, shift = 0, 0
    while True:
        byte = raw[offset]
        offset += 1
        number |= (byte & 127) << shift
        if byte < 128:
            return number, offset
        shift += 7


def split_isolated(pattern, text):
    start = 0
    for match in pattern.finditer(text):
        if match.start() > start:
            yield text[start:match.start()]
        yield match.group()
        start = match.end()
    if start < len(text):
        yield text[start:]


class PreparedEncoding:
    def __init__(self, core, *, normalize=False, stages=()):
        self.core, self.normalize, self.stages = core, normalize, stages

    def encode_ordinary(self, text):
        if self.normalize:
            text = unicodedata.normalize("NFC", text)
        if not self.stages:
            return self.core.encode_ordinary(text)
        parts = [text]
        for pattern in self.stages:
            parts = [piece for part in parts for piece in split_isolated(pattern, part)]
        return [token for part in parts for token in self.core.encode_ordinary(part)]


@cache
def encoding(family):
    from tiktoken import Encoding

    name, pattern = BPE[family]
    raw = read_table(name)
    if not raw.startswith(b"UTOK1\n"):
        raise ValueError(f"Invalid token table {name}.")
    count, offset = int.from_bytes(raw[6:10], "little"), 10
    ranks = {}
    for rank in range(count):
        length, offset = varint(raw, offset)
        if length:
            ranks[raw[offset:offset + length]] = rank
        offset += length
    if offset != len(raw):
        raise ValueError(f"Invalid token table size for {name}.")
    core = Encoding(f"pipy-{family}", pat_str=pattern, mergeable_ranks=ranks, special_tokens={})
    if family == "qwen3":
        return PreparedEncoding(core, normalize=True)
    if family == "deepseek-v3":
        return PreparedEncoding(core, stages=DEEPSEEK_STAGES)
    return core
