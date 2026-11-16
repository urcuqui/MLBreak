"""The black-box prediction service and its output policies (the ML03 defensive controls).

The service takes a FULL record (public features + private attribute) and releases one value per record under
an output policy. The SAME policy is applied to every response — the real target record's response the attacker
observes and every crafted candidate query — because a deployed service cannot tell them apart.

Output policies are restrictions on what is released. They are NOT a formal privacy guarantee: there is no
bound on what an attacker can learn (contrast with differential privacy, which is deliberately not implemented
here because its mechanism and assumptions would need separate validation).

* ``full_probability`` — releases P(high risk) (float64).
* ``quantized_probability`` — releases the midpoint of the equal-width bucket containing P (``bins`` buckets).
* ``label_only`` — releases the decision 1[P ≥ 0.5].

The service makes its decision on the full score in every policy; a consumer of the RELEASED output derives
``1[released ≥ 0.5]``, which can differ from the service decision for quantised scores near 0.5.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..attacks import QueryBudgetExceeded
from .models import LinearModel


def apply_policy(policy: dict[str, Any], p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    kind = policy["kind"]
    if kind == "full_probability":
        return p.copy()
    if kind == "quantized_probability":
        bins = int(policy["bins"])
        k = np.minimum(np.floor(p * bins), bins - 1)
        return (k + 0.5) / bins
    if kind == "label_only":
        return (p >= 0.5).astype(np.float64)
    raise ValueError(f"unknown output policy {kind!r}")


def is_valid_output(policy: dict[str, Any], v: float) -> bool:
    """Whether ``v`` is a value this policy can release (integrity re-validation of stored responses)."""
    kind = policy["kind"]
    if not 0.0 <= v <= 1.0:
        return False
    if kind == "label_only":
        return v in (0.0, 1.0)
    if kind == "quantized_probability":
        bins = int(policy["bins"])
        return abs(v * bins - 0.5 - round(v * bins - 0.5)) < 1e-9
    return True


def describe_policy(policy: dict[str, Any]) -> dict[str, Any]:
    kind = policy["kind"]
    text = {"full_probability": "P(elevated payment risk) as float64",
            "quantized_probability": f"midpoint of one of {policy.get('bins')} equal-width probability buckets",
            "label_only": "decision only, 1[P ≥ 0.5]"}[kind]
    return {**policy, "releases": text, "formal_privacy_guarantee": False}


class PredictionService:
    """Victim model + output policy. Evaluates exactly the records it is given (no lookup of stored records)."""

    def __init__(self, model: LinearModel, policy: dict[str, Any]):
        self._model = model
        self.policy = dict(policy)

    def full_scores(self, X_public: np.ndarray, s: np.ndarray) -> np.ndarray:
        return self._model.proba(np.c_[np.asarray(X_public, dtype=np.float64), np.asarray(s, dtype=np.float64)])

    def respond(self, X_public: np.ndarray, s: np.ndarray) -> np.ndarray:
        return apply_policy(self.policy, self.full_scores(X_public, s))

    def describe(self) -> dict[str, Any]:
        return {"model_artifact_sha256": self._model.artifact_sha256(), "model_id": self._model.model_id,
                "output_policy": describe_policy(self.policy), "decision_policy": "1[P(high risk) ≥ 0.5] on the full score",
                "interface": "black-box: one record (public features + private attribute) in, one released value out"}


def counting_oracle(service: PredictionService, budget: int) -> tuple[Callable[[tuple[float, ...], int], float],
                                                                     Callable[[], int]]:
    """The ONLY handle on the service given to the attack: ``query(public_features, candidate) -> released output``.

    Every call is counted; a call beyond ``budget`` raises :class:`~mlbreak.attacks.QueryBudgetExceeded` (the
    attack is written never to do that; the guard makes a regression in the attack fail loudly). The model,
    its parameters and any stored record are unreachable from the returned closures.
    """
    count = [0]

    def query(public_features: tuple[float, ...], candidate: int) -> float:
        if count[0] >= budget:
            raise QueryBudgetExceeded(f"query budget of {budget} per target exhausted")
        count[0] += 1
        return float(service.respond(np.array([public_features], dtype=np.float64), np.array([candidate]))[0])

    return query, lambda: count[0]
