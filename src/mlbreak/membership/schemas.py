"""Contracts for the ML04:2023 Membership Inference lab (training-set membership inference).

A separate record family (``mlbreak.membership/v1``). It shares the evidence store, the experiment-ID format, the
execution limits, the integrity approach and the output policies of ML03 (:mod:`mlbreak.inversion.schemas`); its
attack surface, per-candidate evidence and metrics are its own.

Separation of roles inside one record:

* :class:`AttackerObservation` and :class:`AttackDecision` — what the ADVERSARY had (candidate features, the label
  when the configuration grants it, the victim's released output) and what it concluded (score, member/non-member);
* :class:`MembershipTruth` — EVALUATOR-only ground truth (designated group, real training membership, source
  population). The attack code never receives it; it is attached after the attack returns.

Schema versions (both load and verify; neither is reinterpreted):

* ``mlbreak.membership/v1`` — metrics are computed against the DESIGNATED group (``truth.group``). For the decoy
  victim this made them placebo population-discrimination metrics labelled as membership metrics. v1 records keep
  their stored meaning and hashes; they are verified under v1 rules and flagged when they are decoy records.
* ``mlbreak.membership/v2`` — every record declares an ``evaluation_contract``:

  - ``membership`` — candidates contain actual training members and actual non-members; ordinary membership metrics
    use the ACTUAL membership ``truth.in_training``;
  - ``placebo`` — no candidate is a training member of the victim (decoy). Ordinary membership metrics still use the
    actual membership (so positive-class metrics such as ROC AUC and TPR are UNDEFINED, and only false-positive
    metrics on real non-members are defined); a separate PLACEBO analysis (``*_placebo_*`` metrics) measures whether
    the attack separates the two designated source pools, placebo group A (drawn from the pool the real victim was
    trained on) and group B (the evaluation pool). It never claims that group A contains members.

Every request schema is strict and contains no field through which a membership flag can be supplied: the inputs are
a victim configuration, candidate counts, attacker knowledge switches and effort budgets.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from ..config import LIMITS
from ..inversion.schemas import LabelOnly, QuantizedProbability, RestrictivePolicy
from ..schemas import ExperimentId, Strict
from .data import N_EVAL_NON_MEMBERS, N_TRAIN

MEMBERSHIP_SCHEMA_V1 = "mlbreak.membership/v1"
MEMBERSHIP_SCHEMA = "mlbreak.membership/v2"
EvaluationContract = Literal["membership", "placebo"]
SCENARIO_ID = "ml04-training-membership-inference"
ModelId = Literal["random-forest-overfit", "logreg-l2", "random-forest-decoy"]
Signal = Literal["true_label_confidence", "negative_log_likelihood", "max_confidence"]
LABEL_SIGNALS = ("true_label_confidence", "negative_log_likelihood")
ATTACKS = ("threshold", "shadow")


# --------------------------------------------------------------------------
# Defenses
# --------------------------------------------------------------------------
class Regularization(Strict):
    """Stronger regularisation of the victim's training. Tree models: every leaf must hold at least
    ``min_samples_leaf`` training records (no leaf can memorise one record). Logistic regression: inverse L2
    strength ``C`` (smaller = stronger). Only the parameter of the victim's model family is applied."""

    kind: Literal["regularization"] = "regularization"
    min_samples_leaf: int = Field(25, ge=2, le=200)
    C: float = Field(0.01, gt=0, le=10)


class ComplexityReduction(Strict):
    """Smaller tree ensemble: depth limit and fewer trees. Not applicable to logistic regression (already linear)."""

    kind: Literal["complexity_reduction"] = "complexity_reduction"
    max_depth: int = Field(4, ge=1, le=32)
    n_estimators: int = Field(30, ge=1, le=100)


class OutputRestriction(Strict):
    """Restrict what the prediction service releases (quantised probability or label only). The model is unchanged.
    Output restriction is NOT differential privacy (see docs/ml04-membership-inference.md)."""

    kind: Literal["output_restriction"] = "output_restriction"
    policy: RestrictivePolicy = Field(default_factory=LabelOnly)


MembershipDefense = Annotated[Union[Regularization, ComplexityReduction, OutputRestriction],
                              Field(discriminator="kind")]


# --------------------------------------------------------------------------
# Candidates and attack
# --------------------------------------------------------------------------
class CandidateSelection(Strict):
    """Final membership-evaluation candidates: ``n_members`` records drawn from the victim's training set and
    ``n_non_members`` from the disjoint evaluation non-member population (deterministic in the seed; a larger count
    extends a smaller one). Unequal counts set the evaluation prevalence; ``operational_prevalences`` are additional
    prevalences for which precision / false-discovery rate are PROJECTED from the measured TPR and FPR."""

    n_members: int = Field(250, ge=1, le=N_TRAIN)
    n_non_members: int = Field(250, ge=1, le=N_EVAL_NON_MEMBERS)
    operational_prevalences: list[Annotated[float, Field(gt=0, lt=1)]] = Field(
        default_factory=lambda: [0.5, 0.1, 0.01], max_length=5)

    @field_validator("operational_prevalences")
    @classmethod
    def _unique(cls, v: list[float]) -> list[float]:
        if len(set(v)) != len(v):
            raise ValueError("operational prevalences must be unique")
        return v


class MembershipAttackConfig(Strict):
    """Both attacks run in every attack experiment, against the same candidates and the same calibration data.

    * Threshold attack: one model-output signal per candidate, compared with thresholds calibrated on the attacker's
      REFERENCE models (trained by the attacker on its own calibration population with a known split).
    * Shadow-model attack: ``n_shadow_models`` models trained on the attacker's shadow population with known
      member/non-member splits; a gradient-boosting meta-classifier learns membership from their released outputs.
      Its operating thresholds are calibrated on the same reference models as the threshold attack.

    ``label_known`` grants the attacker each candidate's true label. Without it only ``max_confidence`` is valid for
    the threshold attack, and the shadow meta-classifier sees the released output only.
    """

    label_known: bool = True
    threshold_signal: Signal = "true_label_confidence"
    threshold_mode: Literal["global", "class_conditional"] = "global"
    min_class_calibration: int = Field(50, ge=10, le=1000,
                                       description="members AND non-members of a class needed for its own threshold")
    target_fpr: float = Field(0.01, ge=0.001, le=0.2, description="low-FPR operating point, calibrated on reference "
                                                                  "models only")
    n_shadow_models: int = Field(8, ge=1, le=LIMITS.max_shadow_models)
    n_reference_models: int = Field(2, ge=1, le=LIMITS.max_reference_models)
    query_budget: int = Field(LIMITS.max_membership_query_budget, ge=1, le=LIMITS.max_membership_query_budget,
                              description="victim queries for the whole experiment (one per candidate)")

    @model_validator(mode="after")
    def _label(self):
        if not self.label_known and self.threshold_signal in LABEL_SIGNALS:
            raise ValueError(f"threshold_signal {self.threshold_signal!r} needs the candidate's label "
                             "(label_known = true); use 'max_confidence'")
        return self


class _Base(Strict):
    scenario_id: Literal["ml04-training-membership-inference"] = SCENARIO_ID
    model_id: ModelId = "random-forest-overfit"
    seed: int = Field(7, ge=0, le=LIMITS.max_membership_seed)
    candidates: CandidateSelection = Field(default_factory=CandidateSelection)


class MembershipBaselineRequest(_Base):
    """Victim utility (training vs held-out accuracy, generalisation gap) and the no-signal baseline. No queries."""


class MembershipAttackRequest(_Base):
    """Threshold and shadow-model membership inference against the victim's released outputs."""

    attack: MembershipAttackConfig = Field(default_factory=MembershipAttackConfig)


class MembershipDefendRequest(Strict):
    """A defended victim (retrained on the SAME training records, or the same model behind an output policy), attacked
    by a RECALIBRATED attack: the attacker's reference and shadow models are retrained with the defended recipe and see
    outputs through the same policy. Same candidates and attack configuration as the parent attack."""

    parent_experiment_id: ExperimentId
    defense: MembershipDefense = Field(default_factory=Regularization)


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------
class AttackerObservation(BaseModel):
    """What the ADVERSARY had for one candidate — and nothing else."""

    features: list[float]
    label: int | None  # None when the configuration does not grant the label
    released_output: float  # the victim's released output for this record (one counted query)
    query_index: int  # 1-based position in the attacker's single batch of queries


class AttackDecision(BaseModel):
    """One attack's output for one candidate (adversary-side)."""

    score: float  # higher = more member-like (threshold: the oriented signal; shadow: meta-classifier P(member))
    class_key: int | None  # class whose threshold applied (class-conditional mode), None for a global threshold
    member_balanced: bool  # decision at the balanced (Youden) operating point
    member_low_fpr: bool | None  # decision at the low-FPR operating point; None if that point is undefined


class MembershipTruth(BaseModel):
    """EVALUATOR-only. Attached after the attack returns; never passed to attack code."""

    group: Literal["member", "non_member"]  # designated evaluation group (the label metrics are computed against)
    in_training: bool  # whether the record is in THIS victim's training set (False for every decoy-control record)
    population: Literal["victim_train", "evaluation_non_members"]
    population_index: int


class CandidateResult(BaseModel):
    candidate_id: str  # opaque, attacker-facing identifier (order of the candidate list)
    truth: MembershipTruth
    observation: AttackerObservation | None = None  # None in a baseline experiment
    threshold: AttackDecision | None = None
    shadow: AttackDecision | None = None
    undefended_correct: dict[str, bool] | None = None  # defend only: parent's correctness per attack (balanced point)


class UtilityRow(BaseModel):
    """One record served by the victim, for utility and the generalisation gap (training split and held-out test)."""

    split: Literal["train", "test"]
    index: int
    y_true: int
    p_full: float  # P(high risk) before the output policy
    released: float  # what the service releases under the experiment's policy


class OperatingPoint(BaseModel):
    name: Literal["balanced", "low_fpr"]
    thresholds: dict[str, float | None]  # "all" or per class ("0", "1"); None = +inf (nobody is called a member)
    calibration_tpr: float
    calibration_fpr: float
    status: Literal["calibrated", "approximate", "undefined"]
    note: str


class AttackCalibration(BaseModel):
    """Threshold calibration of one attack, entirely on the attacker's reference models (no evaluation labels)."""

    attack: Literal["threshold", "shadow"]
    signal: str
    mode: Literal["global", "class_conditional"]
    class_modes: dict[str, str]  # per class: "own threshold" or the reason for falling back to the global threshold
    target_fpr: float
    scores: list[float]  # calibration scores of the reference models' records
    is_member: list[bool]  # ATTACKER-known membership of those records in the attacker's own reference models
    class_keys: list[int]
    operating_points: list[OperatingPoint]
    calibration_auc: float | None


class MembershipRecord(BaseModel):
    """Schema v1 (historical). Unchanged so that stored v1 records keep their dumps and hashes; new records are
    :class:`MembershipRecordV2`."""
    schema_version: Literal["mlbreak.membership/v1"] = MEMBERSHIP_SCHEMA_V1
    family: Literal["membership"] = "membership"
    experiment_id: str
    created_at: str
    mode: Literal["baseline", "attack", "defend"]
    evaluation: Literal["clean", "membership_inference", "adaptive_membership_inference"]
    status: Literal["completed", "failed"]
    error: str | None = None
    intentionally_vulnerable: bool
    scenario_id: str
    owasp: dict[str, Any]
    attacker_capabilities: dict[str, Any] | None
    model: dict[str, Any]  # the victim model (configuration, artifact hash, training population)
    target_system: dict[str, Any]
    dataset: dict[str, Any]  # population sizes, fingerprints, disjointness checks
    seed: int
    output_policy: dict[str, Any]
    attack_config: dict[str, Any] | None
    defense_config: dict[str, Any] | None
    candidate_selection: dict[str, Any]
    calibration: dict[str, Any] | None = None  # reference-model provenance + per-attack AttackCalibration
    shadow: dict[str, Any] | None = None  # shadow-model fingerprints + meta-classifier
    parent_experiment_id: str | None = None
    replay_of: str | None = None
    lineage: dict[str, Any] | None = None
    request: dict[str, Any]
    metrics: dict[str, Any]
    candidates: list[CandidateResult]
    utility: list[UtilityRow]
    reproducibility: dict[str, Any]
    duration_ms: float


class MembershipTruthV2(MembershipTruth):
    """v2 ground truth with the three notions kept apart:

    * ``in_training`` — ACTUAL membership in this victim's training set (the label of ordinary membership metrics);
    * ``group`` / ``population`` — the DESIGNATED candidate group and its source pool (``member`` = drawn from the
      ``victim_train`` pool, ``non_member`` = drawn from ``evaluation_non_members``);
    * ``placebo_group`` — ``"A"`` (``victim_train`` pool) / ``"B"`` (evaluation pool) under the placebo contract only,
      ``None`` under the membership contract.
    """

    placebo_group: Literal["A", "B"] | None


class CandidateResultV2(CandidateResult):
    truth: MembershipTruthV2


class MembershipRecordV2(MembershipRecord):
    """Schema v2: explicit evaluation contract; ordinary membership metrics use actual membership."""

    schema_version: Literal["mlbreak.membership/v2"] = MEMBERSHIP_SCHEMA  # type: ignore[assignment]
    evaluation_contract: EvaluationContract
    candidates: list[CandidateResultV2]  # type: ignore[assignment]


def contract_of(rec) -> str | None:
    """``membership`` / ``placebo`` for v2 records, ``None`` (designated-group semantics) for v1 records."""
    return getattr(rec, "evaluation_contract", None)


def actual_member(rec, c) -> bool:
    """Label of ordinary membership metrics: actual membership (v2) or the stored designated group (v1)."""
    return c.truth.in_training if contract_of(rec) else c.truth.group == "member"


def policy_of(defense: dict[str, Any] | None) -> dict[str, Any]:
    """Output policy implied by a defense (full probability unless the defense is an output restriction)."""
    if defense and defense.get("kind") == "output_restriction":
        return dict(defense["policy"])
    return {"kind": "full_probability"}


__all__ = ["MEMBERSHIP_SCHEMA", "MEMBERSHIP_SCHEMA_V1", "MembershipRecordV2", "CandidateResultV2", "MembershipTruthV2",
           "EvaluationContract", "contract_of", "actual_member", "SCENARIO_ID", "ATTACKS", "LABEL_SIGNALS", "CandidateSelection",
           "MembershipAttackConfig", "MembershipBaselineRequest", "MembershipAttackRequest", "MembershipDefendRequest",
           "MembershipDefense", "Regularization", "ComplexityReduction", "OutputRestriction", "QuantizedProbability",
           "LabelOnly", "MembershipRecord", "CandidateResult", "MembershipTruth", "AttackerObservation",
           "AttackDecision", "UtilityRow", "OperatingPoint", "AttackCalibration", "policy_of"]
