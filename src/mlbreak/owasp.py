"""Registry of the OWASP Machine Learning Security Top 10 (2023) categories.

Only categories whose ``lab_status`` is ``implemented`` have a working,
tested laboratory. Every other entry is a metadata contract only: there is no
placeholder attack behind it and the UI/API report it as not implemented.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TAXONOMY_NAME = "OWASP Machine Learning Security Top 10"
TAXONOMY_VERSION = "2023"
TAXONOMY_URL = "https://owasp.org/www-project-machine-learning-security-top-10/"
_DOCS = "https://owasp.org/www-project-machine-learning-security-top-10/docs/"

LabStatus = Literal["implemented", "not_implemented"]


class OwaspCategory(BaseModel):
    id: str = Field(pattern=r"^ML(0[1-9]|10):2023$")
    code: str = Field(pattern=r"^ML(0[1-9]|10)$")
    name: str
    summary: str
    reference_url: str
    taxonomy: str = TAXONOMY_NAME
    taxonomy_version: str = TAXONOMY_VERSION
    lab_status: LabStatus = "not_implemented"
    scenario_ids: list[str] = Field(default_factory=list)


def _cat(n: int, name: str, slug: str, summary: str, **kw) -> OwaspCategory:
    code = f"ML{n:02d}"
    return OwaspCategory(
        id=f"{code}:2023",
        code=code,
        name=name,
        summary=summary,
        reference_url=f"{_DOCS}{code}_2023-{slug}",
        **kw,
    )


CATEGORIES: tuple[OwaspCategory, ...] = (
    _cat(
        1,
        "Input Manipulation Attack",
        "Input_Manipulation_Attack",
        "An attacker deliberately alters model inputs at inference time so the "
        "model produces an incorrect output (adversarial examples / evasion).",
        lab_status="implemented",
        scenario_ids=["ml01-moons-evasion", "ml01-spam-filter-evasion"],
    ),
    _cat(2, "Data Poisoning Attack", "Data_Poisoning_Attack",
         "Training data is manipulated so the trained model behaves as the attacker intends.",
         lab_status="implemented",
         scenario_ids=["ml02-moons-label-flip"]),
    _cat(3, "Model Inversion Attack", "Model_Inversion_Attack",
         "Model outputs are used to reconstruct sensitive information about training inputs."),
    _cat(4, "Membership Inference Attack", "Membership_Inference_Attack",
         "An attacker determines whether a specific record was part of the training set."),
    _cat(5, "Model Theft", "Model_Theft",
         "An attacker replicates or extracts a model's functionality or parameters."),
    _cat(6, "AI Supply Chain Attacks", "AI_Supply_Chain_Attacks",
         "Compromise of third-party libraries, models, datasets or tooling used to build ML systems."),
    _cat(7, "Transfer Learning Attack", "Transfer_Learning_Attack",
         "A pre-trained model is manipulated so the weakness persists after fine-tuning on a new task."),
    _cat(8, "Model Skewing", "Model_Skewing",
         "Feedback or retraining data is manipulated to shift model behaviour over time."),
    _cat(9, "Output Integrity Attack", "Output_Integrity_Attack",
         "Model outputs are tampered with between the model and the consumer that relies on them."),
    _cat(10, "Model Poisoning", "Model_Poisoning",
         "Model parameters are directly manipulated to make the model behave maliciously."),
)

_BY_ID = {c.id: c for c in CATEGORIES}
_BY_CODE = {c.code: c for c in CATEGORIES}


def get_category(key: str) -> OwaspCategory:
    """Look up a category by ``ML01:2023`` id or ``ML01`` code."""
    cat = _BY_ID.get(key) or _BY_CODE.get(key.upper())
    if cat is None:
        raise KeyError(key)
    return cat
