"""Contracts for the ML02:2023 Data Poisoning lab (label flipping).

Kept separate from the ML01 evasion contracts (:mod:`mlbreak.schemas`) on purpose: the attack surface
(training labels vs test-time inputs), the per-sample evidence (training rows + test rows vs perturbed
test rows) and the metrics differ, so the two families share no record type. They do share the evidence
store, the ID format, the execution limits and the integrity approach.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ..schemas import ExperimentId, Strict

POISONING_SCHEMA = "mlbreak.poisoning/v1"
SCENARIO_ID = "ml02-moons-label-flip"
MAX_POISON_FRACTION = 0.5


class LabelFlipConfig(Strict):
    """Label-flipping poisoning of the training set.

    * ``random`` — flip the labels of ``fraction`` of ALL training samples, chosen uniformly at random
      (seeded). Attacker goal: degrade overall accuracy (availability). Needs no knowledge of the model.
    * ``targeted`` — flip only samples whose true label is ``target_class`` (relabelled as the other
      class). Attacker goal: make the model miss ``target_class`` at test time. ``selection`` picks
      *which* target-class samples:

      - ``random``: uniformly at random among them (needs only the labels);
      - ``boundary``: those with the smallest margin |logit| under the CLEAN model, i.e. nearest the clean
        decision boundary (needs query access to the clean model or a faithful surrogate; here the
        attacker is given the clean model itself — a perfect-knowledge surrogate).

    ``fraction`` is always a fraction of the WHOLE training set, so random and targeted attacks with the
    same fraction flip exactly the same number of labels.
    """

    strategy: Literal["random", "targeted"] = "random"
    fraction: float = Field(0.2, gt=0, le=MAX_POISON_FRACTION)
    target_class: Literal[0, 1] | None = None
    selection: Literal["random", "boundary"] = "random"

    @model_validator(mode="after")
    def _check(self) -> "LabelFlipConfig":
        if self.strategy == "targeted" and self.target_class is None:
            raise ValueError("targeted label flipping requires target_class (0 or 1)")
        if self.strategy == "random":
            if self.target_class is not None:
                raise ValueError("random label flipping has no target_class; use strategy='targeted'")
            if self.selection != "random":
                raise ValueError("selection='boundary' only applies to targeted label flipping")
        return self

    @property
    def knowledge(self) -> str:
        if self.strategy == "random":
            return "none (blind label noise)"
        if self.selection == "random":
            return "training labels"
        return "training labels + clean-model margins (perfect-knowledge surrogate)"


class KnnLabelSanitization(Strict):
    """kNN label sanitisation (after Paudice et al., 2018, "Label Sanitization against Label Flipping
    Poisoning Attacks").

    For every training sample, look at its ``k`` nearest neighbours in feature space (excluding itself).
    If the fraction of neighbours that share its (possibly poisoned) label is below
    ``agreement_threshold``, the label is *suspicious*. ``relabel`` replaces it with the neighbourhood
    majority label; ``remove`` drops the sample before retraining.
    """

    kind: Literal["knn_label_sanitization"] = "knn_label_sanitization"
    k: int = Field(10, ge=3, le=25)
    agreement_threshold: float = Field(0.5, gt=0, lt=1)
    action: Literal["relabel", "remove"] = "relabel"


class _PoisonBase(Strict):
    scenario_id: Literal["ml02-moons-label-flip"] = SCENARIO_ID
    model_id: Literal["mlp-tanh-16x16", "logreg"] = "mlp-tanh-16x16"
    seed: int = Field(7, ge=0, le=2**31 - 1)


class PoisonBaselineRequest(_PoisonBase):
    """Train on the clean training set and evaluate on the clean test set."""


class PoisonAttackRequest(_PoisonBase):
    """Flip training labels, retrain, and compare with the clean model."""

    poison: LabelFlipConfig = Field(default_factory=LabelFlipConfig)


class PoisonDefendRequest(Strict):
    """Defensive replay: apply a label-validation defense to the EXACT poisoned training set of a stored
    attack experiment, retrain, and re-evaluate."""

    parent_experiment_id: ExperimentId
    defense: KnnLabelSanitization = Field(default_factory=KnnLabelSanitization)


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------
TestOutcome = Literal["broken", "kept", "fixed", "still_wrong"]
Detection = Literal["true_positive", "false_positive", "false_negative", "true_negative"]


class TrainingSampleResult(BaseModel):
    train_index: int
    x: list[float]
    y_true: int
    y_observed: int  # label the learner was given (differs from y_true iff poisoned)
    poisoned: bool
    pred_clean: int  # predictions on the training inputs, for training accuracy
    pred_poisoned: int | None = None
    pred_defended: int | None = None
    # Defensive replay only ----------------------------------------------------
    knn_agreement: float | None = None  # fraction of k neighbours sharing y_observed
    flagged: bool | None = None
    y_sanitized: int | None = None  # label after sanitisation; None if the sample was removed
    removed: bool | None = None
    detection: Detection | None = None
    flagged_on_clean: bool | None = None  # the same defense applied to the CLEAN labels (utility cost)


class TestSampleResult(BaseModel):
    __test__ = False  # not a pytest test class, despite the name
    test_index: int
    x: list[float]
    y_true: int
    pred_clean: int
    p1_clean: float
    pred_poisoned: int | None = None
    p1_poisoned: float | None = None
    pred_defended: int | None = None
    p1_defended: float | None = None
    pred_clean_sanitized: int | None = None  # defense applied to clean labels (utility cost)
    outcome: TestOutcome | None = None  # clean model vs poisoned model
    defended_outcome: TestOutcome | None = None  # clean model vs defended model


class PoisoningRecord(BaseModel):
    schema_version: Literal["mlbreak.poisoning/v1"] = POISONING_SCHEMA
    family: Literal["poisoning"] = "poisoning"
    experiment_id: str
    created_at: str
    mode: Literal["baseline", "attack", "defend"]
    evaluation: Literal["clean_training", "poisoning", "poisoning_replay"]
    status: Literal["completed", "failed"]
    error: str | None = None
    intentionally_vulnerable: bool
    scenario_id: str
    owasp: dict[str, Any]
    attacker_capabilities: dict[str, Any] | None
    model: dict[str, Any]  # the CLEAN model (shared reference point for every ML02 record)
    models: dict[str, Any]  # {"clean": ..., "poisoned": ... | None, "defended": ... | None}
    target_system: dict[str, Any]  # the model under evaluation (summary/UI compatibility with ML01)
    dataset: dict[str, Any]
    seed: int
    poison_config: dict[str, Any] | None
    # How the poisoned rows were chosen, their indices, label-vector hashes and the verified budget.
    poisoning: dict[str, Any] | None = None
    defense_config: dict[str, Any] | None
    parent_experiment_id: str | None = None
    replay_of: str | None = None
    lineage: dict[str, Any] | None = None
    request: dict[str, Any]
    metrics: dict[str, Any]
    training_samples: list[TrainingSampleResult]
    test_samples: list[TestSampleResult]
    reproducibility: dict[str, Any]
    duration_ms: float
