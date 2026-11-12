"""Text normalisation defense and the character tables shared with the attack.

The same tables define what the attacker's ``confusables`` / ``invisible`` charsets may produce and what the
``unicode`` normalisation step undoes, so the defense is evaluated against exactly the attack surface it
claims to cover — no more.
"""

from __future__ import annotations

import collections
import re
import string
import unicodedata
from typing import Iterable

from ..hashing import sha256_json
from .schemas import TextNormalizationDefense

ASCII_LETTERS = string.ascii_lowercase

# Visually confusable letters (Cyrillic / Greek / Latin extensions) → ASCII. NFKC does NOT map these.
CONFUSABLES: dict[str, str] = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s",
    "ԁ": "d", "ɡ": "g", "һ": "h", "ӏ": "l", "ο": "o", "ν": "v", "κ": "k", "ı": "i",
}
ASCII_TO_CONFUSABLE: dict[str, list[str]] = collections.defaultdict(list)
for _c, _a in CONFUSABLES.items():
    ASCII_TO_CONFUSABLE[_a].append(_c)

# Zero-width / invisible format characters (Unicode category Cf).
INVISIBLE = ("​", "‌", "‍", "⁠")

_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_LEET_RE = re.compile(r"(?<=[a-z])[013457@$](?=[a-z])")
_WORD_RE = re.compile(r"[a-z']+")


def unicode_fold(text: str) -> str:
    """NFKC, drop format (Cf) characters, map confusables to ASCII."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    return "".join(CONFUSABLES.get(ch, ch) for ch in text)


def edit1_neighbours(word: str) -> set[str]:
    out: set[str] = set()
    for i in range(len(word) + 1):
        a, b = word[:i], word[i:]
        if b:
            out.add(a + b[1:])
        if len(b) > 1:
            out.add(a + b[1] + b[0] + b[2:])
        for ch in ASCII_LETTERS:
            if b:
                out.add(a + ch + b[1:])
            out.add(a + ch + b)
    out.discard(word)
    return out


class Normalizer:
    """Deterministic pre-processing; ``spell_snap`` uses ONLY the training vocabulary."""

    def __init__(self, cfg: TextNormalizationDefense, training_texts: Iterable[str]):
        self.cfg = cfg
        self.vocab: collections.Counter[str] = collections.Counter()
        if cfg.spell_snap:
            for t in training_texts:
                self.vocab.update(_WORD_RE.findall(self._pre(t)))
        self._snap_cache: dict[str, str] = {}

    def _pre(self, text: str) -> str:
        if self.cfg.unicode:
            text = unicode_fold(text)
        text = text.lower()
        if self.cfg.leetspeak:
            text = _LEET_RE.sub(lambda m: m.group(0).translate(_LEET), text)
        return text

    def _snap(self, word: str) -> str:
        if len(word) < 3 or self.vocab.get(word, 0) > 0:
            return word
        if word not in self._snap_cache:
            known = [n for n in edit1_neighbours(word) if self.vocab.get(n, 0) >= self.cfg.spell_min_count]
            self._snap_cache[word] = max(known, key=lambda n: (self.vocab[n], n)) if known else word
        return self._snap_cache[word]

    def __call__(self, text: str) -> str:
        text = self._pre(text)
        if self.cfg.spell_snap:
            text = _WORD_RE.sub(lambda m: self._snap(m.group(0)), text)
        return text

    def describe(self) -> dict:
        return {**self.cfg.model_dump(), "spell_vocabulary_size": len(self.vocab),
                "spell_vocabulary_sha256": sha256_json(sorted(self.vocab.items())) if self.cfg.spell_snap else None,
                "confusables": len(CONFUSABLES), "invisible_characters": [f"U+{ord(c):04X}" for c in INVISIBLE]}
