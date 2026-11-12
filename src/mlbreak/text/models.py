"""Approved spam classifiers: TF-IDF + Logistic Regression (default) or Multinomial Naive Bayes.

Trained in-process from the bundled corpus and an approved configuration — never loaded from pickle/joblib.
The artifact hash covers everything that determines predictions: the TF-IDF vocabulary and IDF weights, the
classifier parameters and the normaliser (including its spell-snap vocabulary).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from ..hashing import sha256_arrays, sha256_json

TFIDF_CONFIG: dict[str, Any] = {
    "analyzer": "word", "lowercase": True, "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True,
    "token_pattern": r"(?u)\b\w\w+\b", "norm": "l2",
}


@dataclass(frozen=True)
class TextModelSpec:
    model_id: str
    estimator: str
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""


TEXT_MODELS: dict[str, TextModelSpec] = {
    "tfidf-word-logreg": TextModelSpec(
        "tfidf-word-logreg", "sklearn.linear_model.LogisticRegression",
        {"C": 10.0, "max_iter": 2000, "class_weight": "balanced", "solver": "lbfgs"},
        "Word uni+bigram TF-IDF + logistic regression (balanced classes)."),
    "tfidf-word-nb": TextModelSpec(
        "tfidf-word-nb", "sklearn.naive_bayes.MultinomialNB", {"alpha": 0.1},
        "Word uni+bigram TF-IDF + multinomial naive Bayes (comparison model)."),
}


class TextClassifier:
    """normaliser (optional) → TF-IDF → classifier. ``p_spam`` is P(class 1 = spam)."""

    def __init__(self, spec: TextModelSpec, vectorizer: TfidfVectorizer, clf: Any, seed: int,
                 normalizer: Callable[[str], str] | None, training_note: str, n_train: int):
        self.spec, self.vectorizer, self.clf, self.seed = spec, vectorizer, clf, seed
        self.normalizer, self.training_note, self.n_train = normalizer, training_note, n_train

    def _features(self, texts: Sequence[str]):
        if self.normalizer is not None:
            texts = [self.normalizer(t) for t in texts]
        return self.vectorizer.transform(texts)

    def p_spam(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.empty(0)
        return self.clf.predict_proba(self._features(texts))[:, 1]

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return (self.p_spam(texts) >= 0.5).astype(np.int64)

    def parameter_arrays(self) -> list[np.ndarray]:
        if isinstance(self.clf, LogisticRegression):
            return [self.vectorizer.idf_, self.clf.coef_, self.clf.intercept_]
        return [self.vectorizer.idf_, self.clf.feature_log_prob_, self.clf.class_log_prior_]

    def artifact_sha256(self) -> str:
        norm = self.normalizer.describe() if self.normalizer is not None else None
        return sha256_json({"vocabulary": sorted(self.vectorizer.vocabulary_.items()),
                            "parameters": sha256_arrays(*self.parameter_arrays()), "normalizer": norm})

    def describe(self) -> dict[str, Any]:
        return {
            "model_id": self.spec.model_id, "estimator": self.spec.estimator,
            "hyperparameters": {**self.spec.params, "random_state": self.seed if "C" in self.spec.params else None},
            "tfidf": {**TFIDF_CONFIG, "ngram_range": list(TFIDF_CONFIG["ngram_range"]),
                      "n_features": len(self.vectorizer.vocabulary_)},
            "normalizer": self.normalizer.describe() if self.normalizer is not None else None,
            "training": self.training_note, "n_train": self.n_train,
            "artifact_sha256": self.artifact_sha256(),
            "artifact_format": "in-memory fitted vectorizer + classifier (no pickle); hash over vocabulary, IDF, "
                               "classifier parameters and normaliser",
        }


def train_text_model(model_id: str, texts: Sequence[str], labels: Sequence[int], seed: int,
                     normalizer: Callable[[str], str] | None = None, training_note: str = "clean") -> TextClassifier:
    spec = TEXT_MODELS.get(model_id)
    if spec is None:
        raise KeyError(f"unknown text model {model_id!r}")
    vec = TfidfVectorizer(**TFIDF_CONFIG)
    docs = [normalizer(t) for t in texts] if normalizer is not None else list(texts)
    X = vec.fit_transform(docs)
    clf = LogisticRegression(random_state=seed, **spec.params) if "C" in spec.params else MultinomialNB(**spec.params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(X, np.asarray(labels))
    return TextClassifier(spec, vec, clf, seed, normalizer, training_note, len(texts))
