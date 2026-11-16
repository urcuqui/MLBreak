"""ML03 inversion metrics, derived ONLY from per-target and per-record evidence.

:func:`derive_inversion_metrics` builds a record's metrics and re-derives them during verification
(:mod:`.integrity`). Every ratio carries numerator, denominator, population and definition; an empty denominator
gives ``value: None`` (undefined), never 0.

ML03 does NOT reuse the ML01 "attack success" semantics. Its success notion is defined here:

* a reconstruction is **correct** when the attacker's inferred value equals the target's private attribute;
  an abstention is never correct;
* **attribute_inference_accuracy** is the fraction of targets reconstructed correctly — it is NOT evidence of
  model-induced leakage on its own, because the attribute is partly predictable from public features;
* **inference_advantage** = inversion accuracy − auxiliary-only accuracy on the SAME targets is the leakage
  attributable to the model's outputs; the paired one-sided McNemar test (``leakage_p_value``) asks whether the
  inversion is correct on significantly more targets than the baseline. A leakage claim requires advantage > 0
  AND p < 0.05.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from ..metrics import ratio
from ..stats import mcnemar_one_sided_p

LEAKAGE_ALPHA = 0.05

# Flat metrics (``record.metrics[k]["value"]``) that regression invariants may reference.
INVARIANT_METRICS = (
    "attribute_inference_accuracy", "baseline_inference_accuracy", "inference_advantage", "leakage_p_value",
    "ambiguity_rate", "unambiguous_accuracy", "abstention_rate", "budget_exhaustion_rate",
    "query_budget_utilisation", "recall_private_0", "recall_private_1", "victim_prediction_accuracy",
    "consumer_decision_accuracy", "undefended_attribute_inference_accuracy", "defense_accuracy_reduction",
    "defense_reduction_p_value", "evidence_consistency",
)


def _g(o: Any, k: str):
    return o.get(k) if isinstance(o, dict) else getattr(o, k)


def mean_block(values: Sequence[float], definition: str, population: str) -> dict[str, Any]:
    v = [float(x) for x in values]
    return {"value": (sum(v) / len(v)) if v else None, "n": len(v), "definition": definition, "population": population}


def _service_metrics(service: Sequence[Any], evaluation: str) -> dict[str, Any]:
    pop = "every record of the held-out evaluation split, served under this experiment's output policy"
    y = np.array([_g(r, "y_true") for r in service], dtype=np.int64)
    released = np.array([_g(r, "released") for r in service], dtype=np.float64)
    full = np.array([_g(r, "p_full") for r in service], dtype=np.float64)
    n = len(service)
    out: dict[str, Any] = {
        "victim_prediction_accuracy": ratio(sum(_g(r, "decision") == _g(r, "y_true") for r in service), n,
                                            "records whose service decision 1[P ≥ 0.5] (made on the full score) "
                                            "equals the true label / records", pop),
        "consumer_decision_accuracy": ratio(sum(_g(r, "consumer_decision") == _g(r, "y_true") for r in service), n,
                                            "records whose decision derived from the RELEASED output, "
                                            "1[released ≥ 0.5], equals the true label / records (utility of the "
                                            "policy to a consumer)", pop),
        "released_brier_score": mean_block(((released - y) ** 2).tolist(),
                                           "mean squared error between the released output and the true label "
                                           "(lower is better; label-only releases 0/1)", pop),
        "score_distortion": mean_block(np.abs(released - full).tolist(),
                                       "mean |released output − full probability| (information removed by the "
                                       "policy; 0 for full probability)", pop),
    }
    auc = roc_auc_score(y, released) if 0 < y.sum() < n else None
    out["released_ranking_auc"] = {"value": None if auc is None else float(auc), "n": n, "population": pop,
                                   "definition": "ROC AUC of the released output for the true label (ties count ½; "
                                                 "label-only can only rank two groups)"}
    if evaluation == "clean" and all(_g(r, "p_public_only") is not None for r in service) and n:
        pub = np.array([_g(r, "p_public_only") for r in service], dtype=np.float64)
        out["accuracy_without_private_attribute"] = ratio(
            int(((pub >= 0.5).astype(int) == y).sum()), n,
            "records correctly decided by a reference model trained WITHOUT the private attribute / records", pop)
        out["log_loss_with_private_attribute"] = {"value": float(log_loss(y, full, labels=[0, 1])), "n": n,
                                                  "population": pop, "definition": "cross-entropy of the victim"}
        out["log_loss_without_private_attribute"] = {"value": float(log_loss(y, pub, labels=[0, 1])), "n": n,
                                                     "population": pop,
                                                     "definition": "cross-entropy of the public-only reference"}
    return out


def derive_inversion_metrics(evaluation: str, targets: Sequence[Any], service: Sequence[Any]) -> dict[str, Any]:
    pop = "selected held-out evaluation records (inversion targets)"
    n = len(targets)
    m: dict[str, Any] = {"n_targets": n}
    base_ok = [bool(_g(t, "baseline_correct")) for t in targets]
    truth = [_g(_g(t, "ground_truth"), "private_attribute") for t in targets]
    m["baseline_inference_accuracy"] = ratio(sum(base_ok), n, "targets whose private attribute equals the "
                                             "auxiliary-only guess (no victim output, no query) / targets", pop)
    for c in (0, 1):
        idx = [k for k, v in enumerate(truth) if v == c]
        m[f"baseline_recall_private_{c}"] = ratio(sum(base_ok[k] for k in idx), len(idx),
                                                  f"targets with attribute {c} guessed as {c} by the auxiliary-only "
                                                  f"attacker / targets with attribute {c}", pop + f", attribute = {c}")
    m.update(_service_metrics(service, evaluation))
    if evaluation == "clean":
        return m

    traces = [_g(t, "trace") for t in targets]
    inv_ok = [bool(_g(t, "inversion_correct")) for t in targets]
    amb = [bool(_g(tr, "ambiguous")) for tr in traces]
    m["attribute_inference_accuracy"] = ratio(sum(inv_ok), n, "targets whose reconstructed attribute equals the "
                                              "true attribute (abstention = not correct) / targets", pop)
    for c in (0, 1):
        idx = [k for k, v in enumerate(truth) if v == c]
        m[f"recall_private_{c}"] = ratio(sum(inv_ok[k] for k in idx), len(idx),
                                         f"targets with attribute {c} reconstructed as {c} / targets with attribute {c}",
                                         pop + f", attribute = {c}")
    m["inference_advantage"] = ratio(sum(inv_ok) - sum(base_ok), n,
                                     "(targets correct by inversion − targets correct by the auxiliary-only attacker) "
                                     "/ targets — signed; the information attributable to the model's outputs", pop)
    b = sum(i and not a for i, a in zip(inv_ok, base_ok))
    c = sum(a and not i for i, a in zip(inv_ok, base_ok))
    p = mcnemar_one_sided_p(b, c) if n else None
    m["leakage_p_value"] = {
        "value": p, "n": n, "inversion_only_correct": b, "baseline_only_correct": c,
        "alpha": LEAKAGE_ALPHA, "leakage_claimed": bool(p is not None and p < LEAKAGE_ALPHA and b > c),
        "population": pop,
        "definition": "exact one-sided McNemar test on the paired per-target correctness of inversion vs auxiliary-"
                      "only attacker: P(X ≤ baseline_only_correct), X ~ Bin(discordant pairs, ½). Leakage is claimed "
                      "only if p < alpha and the inversion is correct on more targets than the baseline"}
    m["ambiguity_rate"] = ratio(sum(amb), n, "targets with more than one candidate consistent with the observed "
                                "output after the attacker's queries / targets", pop)
    unamb = [k for k in range(n) if not amb[k]]
    m["unambiguous_accuracy"] = ratio(sum(inv_ok[k] for k in unamb), len(unamb),
                                      "unambiguous targets reconstructed correctly / unambiguous targets",
                                      pop + ", unambiguous only")
    m["abstention_rate"] = ratio(sum(_g(tr, "inferred") is None for tr in traces), n,
                                 "targets with no inference (tie_policy = abstain on an ambiguous target) / targets", pop)
    m["budget_exhaustion_rate"] = ratio(sum(bool(_g(tr, "budget_exhausted")) for tr in traces), n,
                                        "targets left ambiguous because the per-target query budget ran out before "
                                        "every candidate was queried / targets", pop)
    q = [int(_g(tr, "n_queries")) for tr in traces]
    budget = max((int(_g(tr, "query_budget")) for tr in traces), default=0)
    m["queries"] = {"mean": (sum(q) / n) if n else None, "max": max(q, default=0), "total": sum(q), "n": n,
                    "budget_per_target": budget,
                    "definition": "victim queries sent by the attacker per target (the observed output of the real "
                                  "record is not a query)"}
    m["query_budget_utilisation"] = ratio(sum(q), budget * n, "queries sent / (per-target budget × targets)", pop)
    rules: dict[str, int] = {}
    for tr in traces:
        rules[_g(tr, "rule")] = rules.get(_g(tr, "rule"), 0) + 1
    m["inference_rules"] = dict(sorted(rules.items()))
    if evaluation == "adaptive_inversion":
        und = [bool(_g(t, "undefended_inversion_correct")) for t in targets]
        m["undefended_attribute_inference_accuracy"] = ratio(
            sum(und), n, "the SAME targets reconstructed correctly against the undefended (full-probability) "
                         "service in the parent experiment / targets", pop)
        b2 = sum(u and not d for u, d in zip(und, inv_ok))
        c2 = sum(d and not u for u, d in zip(und, inv_ok))
        m["defense_accuracy_reduction"] = ratio(sum(und) - sum(inv_ok), n,
                                                "(targets correct against full probability − targets correct against "
                                                "this policy) / targets — paired, signed", pop)
        m["defense_reduction_p_value"] = {
            "value": mcnemar_one_sided_p(b2, c2) if n else None, "n": n, "only_undefended_correct": b2,
            "only_defended_correct": c2, "population": pop,
            "definition": "exact one-sided McNemar test that the output policy reduces correct reconstructions on "
                          "the same targets"}
    return m
