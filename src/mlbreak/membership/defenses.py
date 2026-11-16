"""The ML04 prediction service and its defenses.

Every victim configuration — undefended, regularised, reduced, decoy or logistic regression — sits behind the same
:class:`MembershipService`: a batch of records in, one released value per record out (``P(high risk)`` under an
output policy). The output policies are those of ML03 (:func:`mlbreak.inversion.defenses.apply_policy`).

Defenses (see :mod:`.schemas`):

* ``regularization`` / ``complexity_reduction`` change how the victim is TRAINED (same training records);
* ``output_restriction`` changes what the service RELEASES (same model).

None of them is a formal privacy guarantee. Output restriction is in particular NOT differential privacy: the
released value is a deterministic function of the model, which is itself a deterministic function of the training
set, so the privacy loss of releasing it is unbounded (ε = ∞) — it removes information that some attacks use, it
does not bound what any attack can learn.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..attacks import QueryBudgetExceeded
from ..inversion.defenses import apply_policy, describe_policy, is_valid_output
from .models import FittedModel

__all__ = ["MembershipService", "counting_oracle", "apply_policy", "describe_policy", "is_valid_output"]


class MembershipService:
    """Victim model + output policy. Evaluates exactly the records it is given (no lookup of stored records)."""

    def __init__(self, model: FittedModel, policy: dict[str, Any]):
        self._model = model
        self.policy = dict(policy)

    def full_scores(self, X: np.ndarray) -> np.ndarray:
        return self._model.proba(X)

    def respond(self, X: np.ndarray) -> np.ndarray:
        return apply_policy(self.policy, self.full_scores(X))

    def describe(self) -> dict[str, Any]:
        return {"model_artifact_sha256": self._model.artifact_sha256(), "model_family": self._model.recipe.family,
                "output_policy": describe_policy(self.policy),
                "interface": "black-box: a batch of records (8 features) in, one released value per record out; "
                             "identical for every victim configuration",
                "decision_policy": "1[P(high risk) ≥ 0.5] on the full score"}


def counting_oracle(service: MembershipService, budget: int) -> tuple[Callable[[np.ndarray], np.ndarray],
                                                                       Callable[[], int]]:
    """The ONLY handle on the victim given to the attack: ``query(X) -> released outputs`` (one count per record).

    A call that would exceed ``budget`` raises :class:`~mlbreak.attacks.QueryBudgetExceeded` before the victim is
    evaluated. The model, its training records and every membership flag are unreachable from the closures.
    """
    count = [0]

    def query(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if count[0] + len(X) > budget:
            raise QueryBudgetExceeded(f"query budget of {budget} victim queries exhausted "
                                      f"({count[0]} used, {len(X)} requested)")
        count[0] += len(X)
        return service.respond(X)

    return query, lambda: count[0]
