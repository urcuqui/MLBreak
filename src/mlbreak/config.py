"""Runtime configuration: storage location and hard resource limits.

Limits are enforced twice: by the Pydantic request schemas (so the API rejects
oversized requests with 422) and by the experiment engine (wall-clock deadline
and a concurrency gate), so the CLI cannot bypass them either. The gate and the
deadline also cover non-experiment work that can train models (decision-boundary
grids, dataset views), and only one regression suite may run at a time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Directory where experiment evidence is written (``MLBREAK_DATA_DIR``)."""
    return Path(os.environ.get("MLBREAK_DATA_DIR", PROJECT_ROOT / "var")).resolve()


def cases_dir() -> Path:
    """Directory holding the declared security regression cases."""
    return Path(os.environ.get("MLBREAK_CASES_DIR", PROJECT_ROOT / "security_cases")).resolve()


@dataclass(frozen=True)
class Limits:
    max_epsilon: float = 1.0
    max_steps: int = 200
    max_samples: int = 300
    max_query_budget: int = 2000
    max_adv_training_rounds: int = 5
    max_grid_resolution: int = 150
    # Wall-clock limit per experiment. Covers base-model training, adversarial training, attack loops,
    # poisoning retraining and evaluation. COOPERATIVE, not preemptive: the deadline is only checked between
    # uninterruptible segments (fits, rounds, attack steps, samples). Overshoot is bounded by the longest
    # segment that starts before the deadline (typically one scikit-learn fit) plus the short unchecked
    # tail after the last check. See docs/reproducibility.md "Execution limits and timeouts".
    max_experiment_seconds: float = 60.0
    max_concurrent_experiments: int = 2
    # Wall-clock limit for a whole security regression suite (cases not started in time FAIL).
    max_regression_seconds: float = 600.0


LIMITS = Limits()
