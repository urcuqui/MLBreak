"""Pydantic contracts for experiment requests and evidence records.

Request models are strict (``extra="forbid"``) and bounded by
:data:`mlbreak.config.LIMITS`: they are the only way to configure an
experiment, through the API or the CLI. There is no free-form code path.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import LIMITS

EXPERIMENT_ID_PATTERN = r"^MLB-EXP-[0-9a-f]{12}$"
ExperimentId = Annotated[str, Field(pattern=EXPERIMENT_ID_PATTERN)]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------
# Attack configuration
# --------------------------------------------------------------------------
class AttackConfig(Strict):
    """Test-time evasion attack.

    * ``fgsm`` — Fast Gradient Sign Method (Goodfellow et al., 2015). White-box,
      single step, L∞ only.
    * ``pgd`` — Projected Gradient Descent (Madry et al., 2018). White-box,
      iterative, L∞ or L2.
    * ``random_search`` — decision-based black-box search. Uses only hard labels
      returned by the deployed system and a fixed query budget.
    """

    attack: Literal["fgsm", "pgd", "random_search"] = "pgd"
    norm: Literal["linf", "l2"] = "linf"
    epsilon: float = Field(0.3, gt=0, le=LIMITS.max_epsilon)
    steps: int = Field(40, ge=1, le=LIMITS.max_steps)
    step_size: float | None = Field(None, gt=0, le=LIMITS.max_epsilon)
    confidence: float = Field(
        0.0, ge=0, le=10,
        description="Required logit margin past the boundary (Carlini-Wagner style κ). 0 = just cross it.",
    )
    random_start: bool = False
    query_budget: int = Field(500, ge=1, le=LIMITS.max_query_budget)

    @model_validator(mode="after")
    def _check(self) -> "AttackConfig":
        if self.attack == "fgsm":
            if self.norm != "linf":
                raise ValueError("FGSM is defined for the L∞ norm only; use PGD for L2")
            self.steps = 1
            self.step_size = self.epsilon
            self.random_start = False
        return self

    @property
    def effective_step_size(self) -> float:
        return self.step_size if self.step_size is not None else 2.5 * self.epsilon / self.steps

    @property
    def knowledge(self) -> Literal["white-box", "black-box"]:
        return "black-box" if self.attack == "random_search" else "white-box"


# --------------------------------------------------------------------------
# Defense configuration
# --------------------------------------------------------------------------
class AdversarialTrainingDefense(Strict):
    """Retrain on clean + PGD adversarial copies of the training set (Madry et al., 2018)."""

    kind: Literal["adversarial_training"] = "adversarial_training"
    epsilon: float = Field(0.2, gt=0, le=LIMITS.max_epsilon)
    norm: Literal["linf", "l2"] = "linf"
    steps: int = Field(20, ge=1, le=50)
    rounds: int = Field(3, ge=1, le=LIMITS.max_adv_training_rounds)


class ConfidenceRejectionDefense(Strict):
    """Abstain when the model's top-class probability is below ``threshold``."""

    kind: Literal["confidence_rejection"] = "confidence_rejection"
    threshold: float = Field(0.8, gt=0.5, lt=1.0)


DefenseConfig = Annotated[
    Union[AdversarialTrainingDefense, ConfidenceRejectionDefense], Field(discriminator="kind")
]


# --------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------
class SampleSelection(Strict):
    """Which test-split samples to attack. ``indices`` wins over ``max_samples``."""

    indices: list[Annotated[int, Field(ge=0)]] | None = Field(None, max_length=LIMITS.max_samples)
    max_samples: int = Field(LIMITS.max_samples, ge=1, le=LIMITS.max_samples)

    @model_validator(mode="after")
    def _unique(self) -> "SampleSelection":
        if self.indices is not None:
            if not self.indices:
                raise ValueError("indices must not be empty")
            if len(set(self.indices)) != len(self.indices):
                raise ValueError("indices must be unique")
        return self


class _Base(Strict):
    scenario_id: Literal["ml01-moons-evasion"] = "ml01-moons-evasion"
    model_id: Literal["mlp-tanh-16x16", "logreg"] = "mlp-tanh-16x16"
    seed: int = Field(7, ge=0, le=2**31 - 1)


class BaselineRequest(_Base):
    defense: DefenseConfig | None = None


class AttackRequest(_Base):
    attack: AttackConfig = Field(default_factory=AttackConfig)
    samples: SampleSelection = Field(default_factory=SampleSelection)
    target_defense: DefenseConfig | None = Field(
        None,
        description="When set, this is an ADAPTIVE attack: a new attack generated against the "
        "defended system by an attacker who knows the defense.",
    )
    parent_experiment_id: ExperimentId | None = None


class DefendRequest(Strict):
    """Replay the perturbations of an existing attack experiment against a defended system."""

    parent_experiment_id: ExperimentId
    defense: DefenseConfig


class BoundaryRequest(_Base):
    defense: DefenseConfig | None = None
    resolution: int = Field(100, ge=20, le=LIMITS.max_grid_resolution)


# --------------------------------------------------------------------------
# Evidence record
# --------------------------------------------------------------------------
Mode = Literal["baseline", "attack", "defend"]
Status = Literal["completed", "failed"]


class Ratio(BaseModel):
    numerator: int
    denominator: int
    value: float | None
    definition: str


class SampleResult(BaseModel):
    test_index: int
    y_true: int
    x_original: list[float]
    pred_original: int  # -1 = abstained (confidence rejection)
    p1_original: float
    confidence_original: float
    x_adversarial: list[float] | None = None
    pred_adversarial: int | None = None
    p1_adversarial: float | None = None
    confidence_adversarial: float | None = None
    perturbation_linf: float | None = None
    perturbation_l2: float | None = None
    clean_correct: bool
    attack_success: bool | None = None
    queries: int | None = None


class ExperimentRecord(BaseModel):
    schema_version: Literal["mlbreak.experiment/v1"] = "mlbreak.experiment/v1"
    experiment_id: str
    created_at: str
    mode: Mode
    evaluation: Literal["clean", "attack", "adaptive_attack", "replay"]
    status: Status
    error: str | None = None
    intentionally_vulnerable: bool
    scenario_id: str
    owasp: dict[str, Any]
    attacker_capabilities: dict[str, Any] | None
    model: dict[str, Any]
    target_system: dict[str, Any]
    dataset: dict[str, Any]
    seed: int
    attack_config: dict[str, Any] | None
    defense_config: dict[str, Any] | None
    parent_experiment_id: str | None = None
    replay_of: str | None = None
    request: dict[str, Any]
    metrics: dict[str, Any]
    samples: list[SampleResult]
    reproducibility: dict[str, Any]
    duration_ms: float
