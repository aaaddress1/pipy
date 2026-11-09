"""Claude content counts from ctok's measured vocabulary and normalization."""

from dataclasses import dataclass
from functools import cache, lru_cache
from struct import unpack_from

from claude_marks import ESCAPED_MARKER_LITERALS, GLYPH_TO_ATOM
from claude_normalize import nfc, raw_head_space, stream_norm
from text_tokenizer import read_table, varint


CLAUDE = {"claude-v3", "claude-v47", "claude-v5", "claude-v5-sonnet"}
MARKERS = str.maketrans({index: glyph for index, glyph in enumerate(GLYPH_TO_ATOM, 1)})
END = ""


@dataclass(frozen=True)
class Frame:
    fold_quotes: bool
    allcaps_min: int | None
    frame_bow: bool
    frame_tail: str
    frame_strip: str


class Vocabulary:
    def __init__(self, pieces, byte_pieces, *, fold_quotes, caps):
        self.fold_quotes, self.caps = fold_quotes, caps
        self.trie = {}
        self.newlines = []
        for piece in pieces:
            node = self.trie
            for character in reversed(piece):
                node = node.setdefault(character, {})
            node[END] = True
            if len(piece) == 1 and piece not in GLYPH_TO_ATOM:
                byte_pieces.add(piece.encode())
            if piece.strip("\n") == "":
                self.newlines.append(len(piece))
        self.byte_pieces = byte_pieces
        self.newlines.sort()

    @lru_cache(maxsize=4096)
    def floor(self, character):
        raw = ESCAPED_MARKER_LITERALS.get(character, character).encode("utf-8", errors="replace")
        best = [0]
        for end in range(1, len(raw) + 1):
            best.append(min([best[end - 1] + 1] + [best[start] + 1 for start in range(end - 1)
                                                   if raw[start:end] in self.byte_pieces]))
        return best[-1]

    def count(self, marked):
        best = [0] * (len(marked) + 1)
        root = self.trie
        for end, character in enumerate(marked, 1):
            node = root.get(character)
            price = best[end - 1] + (1 if character in GLYPH_TO_ATOM or node and END in node else self.floor(character))
            if node:
                for start in range(end - 2, -1, -1):
                    node = node.get(marked[start])
                    if not node:
                        break
                    if END in node:
                        price = min(price, best[start] + 1)
            best[end] = price
        return best[-1]

    @lru_cache(maxsize=128)
    def tail(self, count, appended):
        if not count:
            return 0
        best = [0] * (count + appended + 1)
        for end in range(1, len(best)):
            price = best[end - 1] + 1
            for size in self.newlines:
                if size > end:
                    break
                price = min(price, best[end - size] + 1)
            best[end] = price
        return best[-1] - 1


@cache
def vocabulary(early):
    name = "ctok_v3" if early else "ctok_v4_7"
    raw = read_table(name)
    if not raw.startswith(b"CTOK\x02"):
        raise ValueError(f"Invalid Claude vocabulary {name}.")
    flags, _, caps, byte_count, piece_count = unpack_from("<BBBHI", raw, 5)
    offset = 14
    byte_pieces = set()
    for _ in range(byte_count):
        length = raw[offset]
        offset += 1
        byte_pieces.add(raw[offset:offset + length])
        offset += length
    pieces, previous = [], b""
    for _ in range(piece_count):
        shared, offset = varint(raw, offset)
        length, offset = varint(raw, offset)
        previous = previous[:shared] + raw[offset:offset + length]
        pieces.append(previous.decode().translate(MARKERS))
        offset += length
    if offset != len(raw):
        raise ValueError(f"Invalid Claude vocabulary size for {name}.")
    return Vocabulary(pieces, byte_pieces, fold_quotes=bool(flags & 1), caps=caps or None)


@cache
def family_data(family):
    vocab = vocabulary(family == "claude-v3")
    ladder = family != "claude-v5"
    frame = Frame(vocab.fold_quotes, vocab.caps, family in ("claude-v3", "claude-v47"),
                  "ladder" if ladder else "free", "\n" if ladder else " \t\n\r\f\v")
    return vocab, frame


def content_count(family, text):
    vocab, frame = family_data(family)
    head_space = raw_head_space(text)
    ladder = frame.frame_tail == "ladder"
    if not ladder:
        text = text.rstrip(frame.frame_strip)
    norm = nfc(text, fold_quotes=frame.fold_quotes)
    count = vocab.count(stream_norm(norm, frame, raw_head_space=head_space))
    if ladder:
        tail = len(norm) - len(norm.rstrip("\n"))
        count += vocab.tail(tail, 0 if family == "claude-v5-sonnet" else 2)
    return count
