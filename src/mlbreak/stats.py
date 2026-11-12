"""Paired comparison of an undefended and a defended evaluation of the SAME attack.

Regression invariants that only bound one posture ("defended success ≤ x") cannot tell a defense
that helps from one whose threshold happened to be met, and they break when numerical noise moves
the single number across a hard-coded line. A paired comparison asks the question the defense
claims to answer: *on the same test samples, does the defense make the attack succeed less often?*

Pairing is per test sample, over the samples eligible in BOTH evaluations:

* ML01 evasion: eligible = attack-eligible outcome (``evaded`` / ``abstained_adversarial`` /
  ``robust``); success = ``evaded``.
* ML01 spam-filter (text) evasion: eligible = attacked messages (``evaded`` / ``failed`` /
  ``budget_exhausted``); success = ``evaded`` (target class reached within every constraint).
* ML02 poisoning: eligible = test samples of the attack's population (target class for targeted
  attacks, all test samples otherwise) that the CLEAN model classifies correctly; success = the
  evaluated model (poisoned / defended) misclassifies them.

Statistics (all over the paired population):

* ``absolute_reduction`` = success rate undefended − success rate defended.
* ``relative_reduction`` = absolute_reduction / success rate undefended (undefined if the latter is 0).
* ``mcnemar_p_value``    = exact one-sided McNemar test (binomial on the discordant pairs) of
  H0 "the defense does not reduce success" vs H1 "it reduces success":
  P(X ≤ c), X ~ Binomial(b + c, ½), where b = samples evaded only without the defense and
  c = samples evaded only with it. 1.0 when there are no discordant pairs.

Undefined values are ``None`` and make any invariant that uses them fail.
"""

from __future__ import annotations

from math import comb
from typing import Any

from .metrics import classify

STATISTICS = ("absolute_reduction", "relative_reduction", "mcnemar_p_value")


def mcnemar_one_sided_p(b: int, c: int) -> float:
    """Exact one-sided McNemar p-value for H1: the second condition has FEWER successes (c < b)."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(comb(n, k) for k in range(c + 1)) / 2 ** n


def paired_success(record: Any) -> dict[int, bool]:
    """Map test_index → attack succeeded, over the samples eligible for the attack in this record."""
    if getattr(record, "family", None) == "text_evasion":
        return {s.test_index: s.outcome == "evaded" for s in record.samples
                if s.outcome in ("evaded", "failed", "budget_exhausted")}
    if getattr(record, "family", None) == "poisoning":
        cfg = record.poison_config or {}
        target = cfg.get("target_class")
        key = "pred_defended" if record.evaluation == "poisoning_replay" else "pred_poisoned"
        return {s.test_index: getattr(s, key) != s.y_true for s in record.test_samples
                if s.pred_clean == s.y_true and (target is None or s.y_true == target)}
    out = {}
    for s in record.samples:
        outcome = s.outcome or classify(s)  # v1 records have no stored outcome
        if outcome in ("evaded", "abstained_adversarial", "robust"):
            out[s.test_index] = outcome == "evaded"
    return out


def paired_comparison(undefended: Any, defended: Any) -> dict[str, Any]:
    u, d = paired_success(undefended), paired_success(defended)
    common = sorted(u.keys() & d.keys())
    n = len(common)
    b = sum(u[i] and not d[i] for i in common)
    c = sum(d[i] and not u[i] for i in common)
    su, sd = sum(u[i] for i in common), sum(d[i] for i in common)
    ru = su / n if n else None
    rd = sd / n if n else None
    absolute = None if ru is None else ru - rd
    relative = None if not ru else (ru - rd) / ru
    return {
        "n_paired": n, "undefended_successes": su, "defended_successes": sd,
        "only_undefended_succeeded_b": b, "only_defended_succeeded_c": c,
        "undefended_rate": ru, "defended_rate": rd,
        "absolute_reduction": absolute, "relative_reduction": relative,
        "mcnemar_p_value": mcnemar_one_sided_p(b, c) if n else None,
        "population": "test samples eligible for the attack in BOTH the undefended and the defended evaluation",
    }
