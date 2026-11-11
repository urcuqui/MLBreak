"""OWASP ML02:2023 — Data Poisoning Attack lab (label flipping + kNN label sanitisation).

Separate from the ML01 evasion family: own request/record schemas, metrics, integrity checks and report.
"""

from .schemas import POISONING_SCHEMA, SCENARIO_ID, PoisoningRecord  # noqa: F401
