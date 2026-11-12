"""Bundled text corpus for the ML01 spam-filter evasion scenario.

The SMS Spam Collection v.1 (CC BY 4.0, see ``data/sms_spam_collection/NOTICE.md``) ships inside the package.
Nothing is downloaded. Loading:

1. verifies the SHA-256 of the bundled file (a modified corpus is refused, not silently used);
2. removes exact duplicates, compared case- and whitespace-insensitively, keeping the first occurrence, so no
   message can appear in both the training and the test split (the raw file has 415 such duplicates);
3. splits deterministically (stratified, ``random_state=seed``).

Labels: 1 = spam, 0 = ham.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from ..hashing import sha256_json

DATA_DIR = Path(__file__).parent / "data" / "sms_spam_collection"
CORPUS_FILE = DATA_DIR / "SMSSpamCollection"
CORPUS_SHA256 = "7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d"
DATASET_ID = "sms-spam-collection-v1"
LABELS = ("ham", "spam")


class CorpusIntegrityError(RuntimeError):
    """The bundled corpus does not match its recorded SHA-256."""


@dataclass(frozen=True)
class TextDataset:
    seed: int
    test_size: float
    train_texts: tuple[str, ...]
    train_labels: tuple[int, ...]
    test_texts: tuple[str, ...]
    test_labels: tuple[int, ...]
    n_raw: int
    n_duplicates_removed: int

    @property
    def y_train(self) -> np.ndarray:
        return np.array(self.train_labels, dtype=np.int64)

    @property
    def y_test(self) -> np.ndarray:
        return np.array(self.test_labels, dtype=np.int64)

    @property
    def split_id(self) -> str:
        return f"{DATASET_ID}/dedup/seed={self.seed}/test_size={self.test_size}/stratified"

    def fingerprint(self) -> dict[str, str]:
        return {"train_sha256": sha256_json([self.train_texts, self.train_labels]),
                "test_sha256": sha256_json([self.test_texts, self.test_labels])}

    def describe(self) -> dict:
        tr, te = self.y_train, self.y_test
        return {
            "dataset_id": DATASET_ID,
            "source": "SMS Spam Collection v.1 (Almeida & Gómez Hidalgo, 2011), UCI ML Repository",
            "doi": "10.24432/C5CC84",
            "license": "CC BY 4.0",
            "bundled_file_sha256": CORPUS_SHA256,
            "n_raw_messages": self.n_raw,
            "n_duplicates_removed": self.n_duplicates_removed,
            "dedup_rule": "exact duplicates after lower-casing and collapsing whitespace; first occurrence kept",
            "labels": {"0": "ham", "1": "spam"},
            "split_id": self.split_id,
            "n_train": len(tr), "n_test": len(te),
            "n_train_spam": int(tr.sum()), "n_test_spam": int(te.sum()),
            **self.fingerprint(),
        }


def _normalize_key(text: str) -> str:
    return " ".join(text.lower().split())


@lru_cache(maxsize=4)
def load_corpus() -> tuple[tuple[str, ...], tuple[int, ...], int]:
    """Verified, de-duplicated corpus: (texts, labels, number of raw lines)."""
    raw = CORPUS_FILE.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != CORPUS_SHA256:
        raise CorpusIntegrityError(f"bundled corpus sha256 {digest} ≠ expected {CORPUS_SHA256}")
    lines = raw.decode("utf-8").splitlines()
    seen: set[str] = set()
    texts, labels = [], []
    for line in lines:
        label, text = line.split("\t", 1)
        if label not in LABELS:
            raise CorpusIntegrityError(f"unexpected label {label!r}")
        key = _normalize_key(text)
        if key in seen:
            continue
        seen.add(key)
        texts.append(text)
        labels.append(LABELS.index(label))
    return tuple(texts), tuple(labels), len(lines)


@lru_cache(maxsize=8)
def load_text_dataset(seed: int, test_size: float = 0.2) -> TextDataset:
    texts, labels, n_raw = load_corpus()
    idx = np.arange(len(texts))
    y = np.array(labels)
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    return TextDataset(
        seed=seed, test_size=test_size,
        train_texts=tuple(texts[i] for i in tr), train_labels=tuple(int(y[i]) for i in tr),
        test_texts=tuple(texts[i] for i in te), test_labels=tuple(int(y[i]) for i in te),
        n_raw=n_raw, n_duplicates_removed=n_raw - len(texts),
    )
