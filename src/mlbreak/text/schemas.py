"""Contracts for the ML01 spam-filter evasion scenario (character-level adversarial text).

Same OWASP category as the 2-D evasion lab (ML01:2023 Input Manipulation), different input modality, so the
records are a separate family (``mlbreak.text-evasion/v1``) with their own per-sample evidence: original and
adversarial text, the exact edit script, queries, edit distance and per-constraint validation.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, model_validator

from ..config import LIMITS
from ..schemas import ExperimentId, Strict

TEXT_SCHEMA = "mlbreak.text-evasion/v1"
SCENARIO_ID = "ml01-spam-filter-evasion"

Operation = Literal["swap", "substitute", "delete", "insert"]
Charset = Literal["ascii_letters", "confusables", "invisible"]
ModelId = Literal["tfidf-word-logreg", "tfidf-word-nb"]


class TextAttackConfig(Strict):
    """DeepWordBug-style greedy character-level attack (see ``mlbreak.text.attacks``).

    Search: score every modifiable word by how much deleting it lowers the source-class probability
    (Gao et al.'s "Replace-1" score; deletion is equivalent to an unknown token for a bag-of-words TF-IDF
    model), then visit words in that order and apply the best of ``candidates_per_word`` random one-character
    edits per visit, until the goal is met, the constraints leave no candidate, or the query budget is spent.

    Constraints (all enforced during the search AND re-validated on the final text):

    * ``max_edit_distance`` — Levenshtein distance (code points) between original and adversarial text;
    * ``max_char_ratio`` — edit distance / original length;
    * ``max_modified_word_ratio`` — modified words / modifiable words;
    * ``max_edits_per_word`` — 1 reproduces DeepWordBug's one edit per word;
    * ``operations`` / ``charsets`` — the only permitted edits and characters;
    * ``protect_numbers`` — tokens containing digits (phone numbers, short codes, prices), URLs and e-mail
      addresses are never modified: changing them would break the spam's call to action;
    * ``protect_stopwords`` — stopwords are never modified (TextAttack's StopwordModification).
    """

    attack: Literal["deepwordbug_greedy"] = "deepwordbug_greedy"
    goal: Literal["spam_to_ham", "ham_to_spam"] = "spam_to_ham"
    min_target_confidence: float = Field(0.5, ge=0.5, lt=1.0, description="target-class probability required")
    operations: list[Operation] = Field(default_factory=lambda: ["swap", "substitute", "delete", "insert"],
                                        min_length=1, max_length=4)
    charsets: list[Charset] = Field(default_factory=lambda: ["ascii_letters"], min_length=1, max_length=3)
    query_budget: int = Field(200, ge=1, le=LIMITS.max_text_query_budget)
    max_edit_distance: int = Field(10, ge=1, le=50)
    max_char_ratio: float = Field(0.15, gt=0, le=0.5)
    max_modified_word_ratio: float = Field(0.5, gt=0, le=1.0)
    max_edits_per_word: int = Field(1, ge=1, le=3)
    candidates_per_word: int = Field(8, ge=1, le=32)
    protect_numbers: bool = True
    protect_stopwords: bool = True
    min_word_length: int = Field(2, ge=1, le=10)

    @model_validator(mode="after")
    def _check(self) -> "TextAttackConfig":
        if len(set(self.operations)) != len(self.operations) or len(set(self.charsets)) != len(self.charsets):
            raise ValueError("operations and charsets must not repeat")
        if set(self.charsets) == {"invisible"} and not ({"insert"} & set(self.operations)):
            raise ValueError("the 'invisible' charset can only be used by the insert operation")
        if set(self.charsets) == {"confusables"} and not ({"substitute"} & set(self.operations)):
            raise ValueError("the 'confusables' charset can only be used by the substitute operation")
        return self

    @property
    def source_class(self) -> int:
        return 1 if self.goal == "spam_to_ham" else 0

    @property
    def target_class(self) -> int:
        return 1 - self.source_class


class TextNormalizationDefense(Strict):
    """Pre-processing applied by the deployed system to every message (training AND inference).

    * ``unicode`` — NFKC normalisation, removal of format characters (zero-width etc.), mapping of common
      confusable letters to ASCII;
    * ``leetspeak`` — digits/symbols between letters mapped to letters (``fr3e`` → ``free``);
    * ``spell_snap`` — a token not in the TRAINING vocabulary is replaced by the most frequent training word at
      edit distance 1 (frequency ≥ ``spell_min_count``); a robust-word-recognition style defense
      (cf. Pruthi et al., 2019).
    """

    kind: Literal["text_normalization"] = "text_normalization"
    unicode: bool = True
    leetspeak: bool = True
    spell_snap: bool = True
    spell_min_count: int = Field(2, ge=1, le=20)


class TextAdversarialTrainingDefense(Strict):
    """Data augmentation with adversarial examples generated from TRAINING spam only (no test data), labelled
    spam, then the model is retrained on clean + adversarial training messages."""

    kind: Literal["text_adversarial_training"] = "text_adversarial_training"
    max_train_samples: int = Field(300, ge=10, le=600)
    query_budget: int = Field(100, ge=10, le=500)
    candidates_per_word: int = Field(6, ge=1, le=16)
    max_edit_distance: int = Field(10, ge=1, le=50)
    max_char_ratio: float = Field(0.15, gt=0, le=0.5)


TextDefense = Annotated[Union[TextNormalizationDefense, TextAdversarialTrainingDefense], Field(discriminator="kind")]


class TextSampleSelection(Strict):
    """Test-split messages to attack. ``indices`` (test-split indices) win over ``max_samples`` (the first
    messages of the goal's source class, in test-split order)."""

    indices: list[Annotated[int, Field(ge=0)]] | None = Field(None, max_length=LIMITS.max_text_samples)
    max_samples: int = Field(100, ge=1, le=LIMITS.max_text_samples)

    @model_validator(mode="after")
    def _unique(self) -> "TextSampleSelection":
        if self.indices is not None and (not self.indices or len(set(self.indices)) != len(self.indices)):
            raise ValueError("indices must be non-empty and unique")
        return self


class _TextBase(Strict):
    scenario_id: Literal["ml01-spam-filter-evasion"] = SCENARIO_ID
    model_id: ModelId = "tfidf-word-logreg"
    seed: int = Field(7, ge=0, le=2**31 - 1)


class TextBaselineRequest(_TextBase):
    defense: TextDefense | None = None


class TextAttackRequest(_TextBase):
    attack: TextAttackConfig = Field(default_factory=TextAttackConfig)
    samples: TextSampleSelection = Field(default_factory=TextSampleSelection)
    target_defense: TextDefense | None = Field(
        None, description="When set, this is an ADAPTIVE attack: a new search run against the defended system.")
    parent_experiment_id: ExperimentId | None = None


class TextDefendRequest(Strict):
    """Non-adaptive replay: the exact adversarial messages of a stored attack, re-submitted to a defended system."""

    parent_experiment_id: ExperimentId
    defense: TextDefense


class TextClassifyRequest(_TextBase):
    text: str = Field(min_length=1, max_length=LIMITS.max_text_length)
    defense: TextDefense | None = None


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------
TextOutcome = Literal["evaded", "failed", "budget_exhausted", "skipped_misclassified", "not_perturbed"]


class TextEdit(BaseModel):
    word_index: int  # index of the modified word segment
    op: Operation
    position: int  # character position inside the word BEFORE the edit
    char_before: str | None  # removed / replaced character (None for insert)
    char_after: str | None  # inserted / new character (None for delete)
    word_before: str
    word_after: str


class TextSampleResult(BaseModel):
    test_index: int
    y_true: int
    text_original: str
    pred_original: int
    p_spam_original: float
    text_adversarial: str | None = None
    pred_adversarial: int | None = None
    p_spam_adversarial: float | None = None
    queries: int | None = None
    edit_distance: int | None = None
    char_ratio: float | None = None
    words_modifiable: int | None = None
    words_modified: int | None = None
    edits: list[TextEdit] | None = None
    constraints: dict[str, bool] | None = None
    outcome: TextOutcome


class TestPrediction(BaseModel):
    __test__ = False  # not a pytest class
    test_index: int
    y_true: int
    pred: int
    p_spam: float


class TextEvasionRecord(BaseModel):
    schema_version: Literal["mlbreak.text-evasion/v1"] = TEXT_SCHEMA
    family: Literal["text_evasion"] = "text_evasion"
    experiment_id: str
    created_at: str
    mode: Literal["baseline", "attack", "defend"]
    evaluation: Literal["clean", "attack", "adaptive_attack", "replay"]
    status: Literal["completed", "failed"]
    error: str | None = None
    intentionally_vulnerable: bool
    scenario_id: str
    owasp: dict[str, Any]
    attacker_capabilities: dict[str, Any] | None
    model: dict[str, Any]  # undefended base classifier
    target_system: dict[str, Any]  # the deployed system that was evaluated
    dataset: dict[str, Any]
    seed: int
    attack_config: dict[str, Any] | None
    defense_config: dict[str, Any] | None
    parent_experiment_id: str | None = None
    replay_of: str | None = None
    lineage: dict[str, Any] | None = None
    request: dict[str, Any]
    metrics: dict[str, Any]
    samples: list[TextSampleResult]
    test_predictions: list[TestPrediction]
    reproducibility: dict[str, Any]
    duration_ms: float
