"""Contracts for the ML03:2023 Model Inversion lab (private-attribute inversion).

A separate record family (``mlbreak.inversion/v1``): the attack surface (a prediction service queried with
crafted records), the per-sample evidence (an attacker trace + evaluation-only ground truth) and the metrics
(attribute-inference accuracy and advantage over an auxiliary-only baseline) share nothing with the evasion or
poisoning records. They share the evidence store, the ID format, the execution limits and the integrity approach.

Request schemas are strict (unknown fields are rejected) and contain NO field through which a private attribute
value could be supplied or returned: the attacker-facing inputs are public features, an output policy and an
effort budget only.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

from ..config import LIMITS
from ..schemas import ExperimentId, Strict

INVERSION_SCHEMA = "mlbreak.inversion/v1"
SCENARIO_ID = "ml03-private-attribute-inversion"
ModelId = Literal["logreg", "logreg-public-only"]


# --------------------------------------------------------------------------
# Output policies (what the prediction service releases)
# --------------------------------------------------------------------------
class FullProbability(Strict):
    """The service releases P(high risk) as a float64 — the undefended service."""

    kind: Literal["full_probability"] = "full_probability"


class QuantizedProbability(Strict):
    """The service releases the MIDPOINT of the equal-width probability bucket that contains P(high risk):
    ``bins`` buckets over [0, 1], bucket ``k = min(floor(p · bins), bins − 1)``, released value ``(k + ½) / bins``.
    """

    kind: Literal["quantized_probability"] = "quantized_probability"
    bins: int = Field(4, ge=2, le=1000)


class LabelOnly(Strict):
    """The service releases only the decision ``1[p ≥ 0.5]`` (0 or 1), no score."""

    kind: Literal["label_only"] = "label_only"


OutputPolicyConfig = Annotated[Union[FullProbability, QuantizedProbability, LabelOnly], Field(discriminator="kind")]
RestrictivePolicy = Annotated[Union[QuantizedProbability, LabelOnly], Field(discriminator="kind")]


# --------------------------------------------------------------------------
# Attack
# --------------------------------------------------------------------------
class InversionAttackConfig(Strict):
    """Black-box candidate-matching reconstruction of a binary private attribute.

    For each target the attacker submits candidate records ``(public features, s = c)`` to the prediction service,
    in a deterministic order (the candidate its auxiliary prior prefers first), and keeps the candidates whose
    response is within ``tolerance`` of the observed response. It stops as soon as only one candidate remains
    consistent, or when ``query_budget`` is spent. Several consistent candidates = an AMBIGUOUS reconstruction,
    resolved by ``tie_policy``: ``auxiliary_prior`` (the auxiliary-only guess — never better than the baseline on
    those targets) or ``abstain`` (no inference; counted as not correct).
    """

    attack: Literal["candidate_matching"] = "candidate_matching"
    query_budget: int = Field(2, ge=1, le=LIMITS.max_inversion_query_budget)
    tie_policy: Literal["auxiliary_prior", "abstain"] = "auxiliary_prior"
    tolerance: float = Field(1e-9, ge=0, le=1e-3)


class TargetSelection(Strict):
    """Which held-out evaluation records are attacked (same selection ⇒ same targets, so results are paired)."""

    max_targets: int = Field(200, ge=1, le=LIMITS.max_inversion_targets)
    order: Literal["first", "random"] = "first"
    indices: list[Annotated[int, Field(ge=0, lt=LIMITS.max_inversion_targets)]] | None = Field(
        None, max_length=LIMITS.max_inversion_targets,
        description="explicit evaluation-split indices (overrides max_targets/order); must be unique")


class _Base(Strict):
    scenario_id: Literal["ml03-private-attribute-inversion"] = SCENARIO_ID
    model_id: ModelId = "logreg"
    seed: int = Field(7, ge=0, le=LIMITS.max_inversion_seed)
    targets: TargetSelection = Field(default_factory=TargetSelection)


class InversionBaselineRequest(_Base):
    """Victim utility on the evaluation split + the auxiliary-only attacker on the selected targets (no queries)."""


class InversionAttackRequest(_Base):
    """Model inversion against the UNDEFENDED service (full probability output)."""

    attack: InversionAttackConfig = Field(default_factory=InversionAttackConfig)


class InversionDefendRequest(Strict):
    """A NEW (adaptive) reconstruction attempt against a restricted output policy, with the parent attack's
    model, seed, targets and attack configuration. The observed target response and every candidate query go
    through the same policy."""

    parent_experiment_id: ExperimentId
    output_policy: RestrictivePolicy = Field(default_factory=LabelOnly)


class InversionTargetsRequest(Strict):
    """Attacker-facing view of the evaluation targets (public features + observed output only)."""

    model_id: ModelId = "logreg"
    seed: int = Field(7, ge=0, le=LIMITS.max_inversion_seed)
    output_policy: OutputPolicyConfig = Field(default_factory=FullProbability)
    targets: TargetSelection = Field(default_factory=lambda: TargetSelection(max_targets=50))


class InversionQueryRequest(Strict):
    """One query to the black-box prediction service: a full (possibly crafted) record in, the released output
    out. Nothing is looked up — the service evaluates exactly the record it is given."""

    model_id: ModelId = "logreg"
    seed: int = Field(7, ge=0, le=LIMITS.max_inversion_seed)
    output_policy: OutputPolicyConfig = Field(default_factory=FullProbability)
    public_features: list[Annotated[float, Field(ge=-10, le=10)]] = Field(min_length=4, max_length=4)
    private_attribute: Literal[0, 1]


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------
InferenceRule = Literal["unique_match", "elimination", "tie_auxiliary_prior", "tie_abstain",
                        "no_consistent_candidate"]


class CandidateQuery(BaseModel):
    query_index: int  # 1-based order in which the attacker sent it
    candidate: int  # private-attribute value placed in the crafted record
    response: float  # output released by the service under the policy
    distance: float  # |response − observed_output|


class AttackerTrace(BaseModel):
    """Everything the ADVERSARY had or computed for one target — and nothing else."""

    public_features: list[float]
    observed_output: float  # the service's released output for the real target record
    candidate_domain: list[int]
    auxiliary_p1: float  # the attacker's own prior P(s = 1 | public features) from auxiliary data
    queries: list[CandidateQuery]
    n_queries: int
    query_budget: int
    budget_exhausted: bool  # some candidate could not be queried within the budget
    consistent_candidates: list[int]
    ambiguous: bool
    rule: InferenceRule
    inferred: int | None
    separation: float | None  # |response(s=1) − response(s=0)| when both were queried


class GroundTruth(BaseModel):
    """Evaluation-only. Never passed to the attack code."""

    private_attribute: int
    target_label: int


class TargetResult(BaseModel):
    target_index: int  # index into the evaluation split
    auxiliary_p1: float  # auxiliary-only attacker: P(s = 1 | public features)
    baseline_inferred: int  # 1[auxiliary_p1 > 0.5]
    baseline_correct: bool
    ground_truth: GroundTruth
    trace: AttackerTrace | None = None  # None in a baseline experiment
    inversion_correct: bool | None = None  # inferred == ground truth (abstention = False)
    undefended_inversion_correct: bool | None = None  # defend only: the parent (full-probability) result


class ServiceRow(BaseModel):
    """One evaluation-split record served by the prediction service (utility of the output policy)."""

    eval_index: int
    y_true: int
    p_full: float  # P(high risk) of the real record before the output policy
    released: float  # what the policy releases
    decision: int  # 1[p_full ≥ 0.5] — the service's decision is made on the full score
    consumer_decision: int  # decision a consumer derives from the released output: 1[released ≥ 0.5]
    p_public_only: float | None = None  # baseline only: reference model trained WITHOUT the private attribute


class InversionRecord(BaseModel):
    schema_version: Literal["mlbreak.inversion/v1"] = INVERSION_SCHEMA
    family: Literal["inversion"] = "inversion"
    experiment_id: str
    created_at: str
    mode: Literal["baseline", "attack", "defend"]
    evaluation: Literal["clean", "inversion", "adaptive_inversion"]
    status: Literal["completed", "failed"]
    error: str | None = None
    intentionally_vulnerable: bool
    scenario_id: str
    owasp: dict[str, Any]
    attacker_capabilities: dict[str, Any] | None
    model: dict[str, Any]  # the victim model
    auxiliary_model: dict[str, Any] | None  # the attacker's own auxiliary-only model
    target_system: dict[str, Any]
    dataset: dict[str, Any]
    seed: int
    output_policy: dict[str, Any]  # the policy the service applied in this experiment
    attack_config: dict[str, Any] | None
    defense_config: dict[str, Any] | None  # the restrictive output policy (None = undefended full probability)
    target_selection: dict[str, Any]
    parent_experiment_id: str | None = None
    replay_of: str | None = None
    lineage: dict[str, Any] | None = None
    request: dict[str, Any]
    metrics: dict[str, Any]
    targets: list[TargetResult]
    service: list[ServiceRow]
    reproducibility: dict[str, Any]
    duration_ms: float
