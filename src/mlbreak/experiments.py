"""Experiment engine: Baseline → Attack → Defend, replay and evidence.

Three distinct evaluations are kept deliberately separate:

* ``attack`` — a new attack generated against the **undefended** model.
* ``replay`` (mode ``defend``) — the *exact* adversarial inputs of an earlier
  attack experiment are re-submitted to a **defended** system. This measures a
  non-adaptive attacker (perturbations crafted without knowledge of the defense).
* ``adaptive_attack`` (mode ``attack`` with ``target_defense``) — a **new**
  attack generated against the defended system by an attacker who knows the
  defense. Stored as its own experiment, linked to its parent.

:meth:`ExperimentEngine.replay` is something else again: it re-executes a
stored experiment from its recorded request and checks the results are
bit-for-bit identical (reproducibility, not robustness).
"""

from __future__ import annotations

import platform
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import sklearn

from . import __version__
from .attacks import (ABSTAIN, BudgetViolation, ExperimentTimeout, QueryBudgetExceeded,
                      enforce_constraints, gradient_attack, lp_norm, random_search_attack)
from .config import LIMITS
from .data import load_dataset
from .defenses import DeployedSystem, base_model, build_system, logit_of
from .evidence import EvidenceStore
from .hashing import sha256_json
from .owasp import get_category
from .schemas import (AttackRequest, BaselineRequest, BoundaryRequest, DefendRequest,
                      ExperimentRecord, SampleResult)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    title: str
    owasp_id: str
    dataset_id: str
    intentionally_vulnerable: bool
    description: str


SCENARIOS: dict[str, Scenario] = {
    "ml01-moons-evasion": Scenario(
        scenario_id="ml01-moons-evasion",
        title="Evasion of a 2-D classifier with bounded test-time perturbations",
        owasp_id="ML01:2023",
        dataset_id="moons-v1",
        intentionally_vulnerable=True,
        description=(
            "A classifier trained on a synthetic two-moons dataset is attacked at "
            "inference time. The attacker perturbs a correctly classified input "
            "within an Lp budget ε so the deployed system returns the wrong class."
        ),
    )
}


class EngineBusy(RuntimeError):
    """Raised when the concurrent-experiment limit is reached."""


class InvalidExperiment(ValueError):
    """The request is well-formed but cannot be executed (e.g. bad parent)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return f"MLB-EXP-{uuid.uuid4().hex[:12]}"


def _ratio(num: int, den: int, definition: str) -> dict[str, Any]:
    return {"numerator": int(num), "denominator": int(den),
            "value": (num / den) if den else None, "definition": definition}


def _stats(linf: np.ndarray, l2: np.ndarray) -> dict[str, Any]:
    if len(linf) == 0:
        return {"n": 0}
    return {"n": int(len(linf)), "linf_mean": float(np.mean(linf)), "linf_max": float(np.max(linf)),
            "linf_median": float(np.median(linf)), "l2_mean": float(np.mean(l2)),
            "l2_max": float(np.max(l2)), "l2_median": float(np.median(l2))}


def reproducibility_info() -> dict[str, Any]:
    return {
        "mlbreak_version": __version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "deterministic": True,
        "note": "Bit-identical replay is expected on the same platform and library versions.",
    }


def results_fingerprint(samples: list[SampleResult], metrics: dict[str, Any]) -> str:
    return sha256_json({"samples": [s.model_dump() for s in samples], "metrics": metrics})


class ExperimentEngine:
    def __init__(self, store: EvidenceStore, max_concurrent: int = LIMITS.max_concurrent_experiments,
                 time_limit: float = LIMITS.max_experiment_seconds):
        self.store = store
        self._gate = threading.BoundedSemaphore(max_concurrent)
        self.time_limit = time_limit

    # ------------------------------------------------------------------ API
    def run_baseline(self, req: BaselineRequest, replay_of: str | None = None) -> ExperimentRecord:
        return self._run("baseline", req, self._baseline, replay_of)

    def run_attack(self, req: AttackRequest, replay_of: str | None = None) -> ExperimentRecord:
        return self._run("attack", req, self._attack, replay_of)

    def run_defend(self, req: DefendRequest, replay_of: str | None = None) -> ExperimentRecord:
        parent = self.store.load(req.parent_experiment_id)  # raises ExperimentNotFound
        if parent.mode != "attack" or parent.evaluation != "attack" or parent.status != "completed":
            raise InvalidExperiment(
                "defend/replay requires a completed attack experiment against the undefended model")
        return self._run("defend", req, lambda r, deadline: self._defend(r, parent, deadline), replay_of)

    def replay(self, experiment_id: str) -> dict[str, Any]:
        """Re-execute a stored experiment from its request and compare results exactly."""
        original = self.store.load(experiment_id)
        if original.status != "completed":
            raise InvalidExperiment("only completed experiments can be replayed")
        if original.mode == "baseline":
            new = self.run_baseline(BaselineRequest.model_validate(original.request), replay_of=experiment_id)
        elif original.mode == "attack":
            new = self.run_attack(AttackRequest.model_validate(original.request), replay_of=experiment_id)
        else:
            new = self.run_defend(DefendRequest.model_validate(original.request), replay_of=experiment_id)

        pairs = [
            ("status", original.status, new.status),
            ("model_artifact_sha256", original.model["artifact_sha256"], new.model.get("artifact_sha256")),
            ("target_system_artifact_sha256", original.target_system["model_artifact_sha256"],
             new.target_system.get("model_artifact_sha256")),
            ("train_split_sha256", original.dataset["train_sha256"], new.dataset.get("train_sha256")),
            ("test_split_sha256", original.dataset["test_sha256"], new.dataset.get("test_sha256")),
            ("results_sha256", original.reproducibility["results_sha256"],
             new.reproducibility.get("results_sha256")),
        ]
        checks = [{"check": n, "original": a, "replay": b, "match": a == b} for n, a, b in pairs]
        return {"original_experiment_id": experiment_id, "replay_experiment_id": new.experiment_id,
                "identical": all(c["match"] for c in checks), "checks": checks}

    def boundary(self, req: BoundaryRequest) -> dict[str, Any]:
        scen = SCENARIOS[req.scenario_id]
        ds = load_dataset(scen.dataset_id, req.seed)
        system = build_system(req.model_id, scen.dataset_id, req.seed, req.defense)
        xs = np.linspace(*ds.spec.domain[0], req.resolution)
        ys = np.linspace(*ds.spec.domain[1], req.resolution)
        gx, gy = np.meshgrid(xs, ys)
        d = system.decide(np.c_[gx.ravel(), gy.ravel()])
        shape = gx.shape
        return {
            "x": xs.round(5).tolist(), "y": ys.round(5).tolist(),
            "p1": d.p1.reshape(shape).round(5).tolist(),
            "abstain": (d.labels == ABSTAIN).reshape(shape).astype(int).tolist(),
            "target_system": system.describe(),
            "model": system.model.describe(),
        }

    # ------------------------------------------------------------ internals
    def _run(self, mode: str, req, fn, replay_of: str | None) -> ExperimentRecord:
        if not self._gate.acquire(blocking=False):
            raise EngineBusy("too many concurrent experiments; retry shortly")
        started = time.monotonic()
        exp_id = _new_id()
        try:
            try:
                body = fn(req, started + self.time_limit)
                status, error = "completed", None
            except (BudgetViolation, ExperimentTimeout, QueryBudgetExceeded) as exc:
                body, status, error = self._failure_body(req), "failed", f"{type(exc).__name__}: {exc}"
            body["metrics"] = body.get("metrics", {})
            body["samples"] = body.get("samples", [])
            repro = reproducibility_info()
            repro["config_sha256"] = sha256_json(req.model_dump(mode="json"))
            repro["results_sha256"] = results_fingerprint(body["samples"], body["metrics"]) \
                if status == "completed" else None
            repro["replay_command"] = f"mlbreak replay {exp_id}"
            record = ExperimentRecord(
                experiment_id=exp_id, created_at=_now(), mode=mode, status=status, error=error,
                replay_of=replay_of, request=req.model_dump(mode="json"), reproducibility=repro,
                duration_ms=round((time.monotonic() - started) * 1000, 2), **body,
            )
            self.store.save(record)
            return record
        finally:
            self._gate.release()

    def _context(self, scenario_id: str, model_id: str, seed: int):
        scen = SCENARIOS[scenario_id]
        cat = get_category(scen.owasp_id)
        ds = load_dataset(scen.dataset_id, seed)
        model = base_model(model_id, scen.dataset_id, seed)
        common = {
            "scenario_id": scen.scenario_id,
            "owasp": {"id": cat.id, "name": cat.name, "taxonomy": cat.taxonomy,
                      "taxonomy_version": cat.taxonomy_version, "reference_url": cat.reference_url},
            "model": model.describe(),
            "dataset": ds.describe(),
            "seed": seed,
        }
        return scen, ds, common

    def _failure_body(self, req) -> dict[str, Any]:
        r = req
        if isinstance(req, DefendRequest):
            parent = self.store.load(req.parent_experiment_id)
            r = AttackRequest.model_validate(parent.request)
        _, ds, common = self._context(r.scenario_id, r.model_id, r.seed)
        return {**common, "evaluation": "attack", "intentionally_vulnerable": True,
                "attacker_capabilities": None, "target_system": {"model_artifact_sha256": None},
                "attack_config": getattr(r, "attack", None) and r.attack.model_dump(),
                "defense_config": None}

    @staticmethod
    def _select(ds, selection) -> np.ndarray:
        n = len(ds.y_test)
        if selection.indices is not None:
            idx = np.array(selection.indices, dtype=int)
            if np.any(idx >= n):
                raise InvalidExperiment(f"sample indices must be < {n} (test split size)")
            return idx
        return np.arange(min(selection.max_samples, n))

    @staticmethod
    def _clean_metrics(system: DeployedSystem, ds) -> dict[str, Any]:
        d = system.decide(ds.X_test)
        n = len(ds.y_test)
        correct = int(np.sum(d.labels == ds.y_test))
        accepted = d.labels != ABSTAIN
        out = {"clean_accuracy": _ratio(correct, n, "correct decisions / all test samples "
                                                   "(abstentions count as not correct)")}
        if system.reject_threshold is not None:
            out["coverage"] = _ratio(int(accepted.sum()), n, "non-abstained decisions / all test samples")
            out["selective_accuracy"] = _ratio(correct, int(accepted.sum()),
                                               "correct decisions / non-abstained decisions")
        return out

    def _baseline(self, req: BaselineRequest, deadline: float) -> dict[str, Any]:
        scen, ds, common = self._context(req.scenario_id, req.model_id, req.seed)
        system = build_system(req.model_id, scen.dataset_id, req.seed, req.defense)
        d = system.decide(ds.X_test)
        samples = [
            SampleResult(test_index=i, y_true=int(ds.y_test[i]), x_original=ds.X_test[i].tolist(),
                         pred_original=int(d.labels[i]), p1_original=float(d.p1[i]),
                         confidence_original=float(d.confidence[i]),
                         clean_correct=bool(d.labels[i] == ds.y_test[i]))
            for i in range(len(ds.y_test))
        ]
        metrics = self._clean_metrics(system, ds)
        metrics["train_accuracy"] = _ratio(int(np.sum(system.model.predict(ds.X_train) == ds.y_train)),
                                           len(ds.y_train), "model argmax correct / training samples")
        return {**common, "evaluation": "clean", "intentionally_vulnerable": req.defense is None,
                "attacker_capabilities": None, "target_system": system.describe(),
                "attack_config": None, "defense_config": system.defense,
                "metrics": metrics, "samples": samples}

    def _attack(self, req: AttackRequest, deadline: float) -> dict[str, Any]:
        scen, ds, common = self._context(req.scenario_id, req.model_id, req.seed)
        if req.parent_experiment_id:
            self.store.load(req.parent_experiment_id)  # must exist
        system = build_system(req.model_id, scen.dataset_id, req.seed, req.target_defense)
        cfg = req.attack
        idx = self._select(ds, req.samples)
        X0, y = ds.X_test[idx], ds.y_test[idx]
        clean = system.decide(X0)
        correct = clean.labels == y
        att = np.flatnonzero(correct)  # only correctly-decided inputs are attacked
        lo, hi = ds.domain_low, ds.domain_high
        rngs = [np.random.default_rng([req.seed, int(i)]) for i in idx[att]]
        adaptive = req.target_defense is not None

        if cfg.knowledge == "white-box":
            min_conf = 0.0
            if system.reject_threshold is not None:
                # Adaptive: land past the rejection threshold, not just past the boundary.
                min_conf = logit_of(system.reject_threshold) + 1e-3
            out = gradient_attack(system.model, X0[att], y[att], cfg, lo, hi, rngs, deadline, min_conf)
        else:
            out = random_search_attack(lambda X: system.decide(X).labels, X0[att], y[att], cfg,
                                       lo, hi, rngs, deadline)
        budget_ratio = enforce_constraints(X0[att], out.x_adv, cfg.epsilon, cfg.norm, lo, hi)
        adv = system.decide(out.x_adv) if len(att) else None

        caps = {
            "knowledge": cfg.knowledge,
            "access": ("model parameters and exact input gradients" if cfg.knowledge == "white-box"
                       else f"hard-label queries to the deployed system (budget {cfg.query_budget}/sample)"),
            "knows_defense": adaptive,
            "can_modify": "the 2 input features of a test-time sample",
            "constraint": f"‖δ‖_{cfg.norm} ≤ {cfg.epsilon}, input stays in the feature domain",
            "cannot": "modify training data, model parameters, or the decision policy",
        }
        if cfg.knowledge == "white-box":
            caps["effective_confidence_kappa"] = out.effective_confidence
        samples, success = self._samples(idx, X0, y, clean, att, out.x_adv, adv, out.queries)
        metrics = self._attack_metrics(system, ds, cfg, idx, y, correct, att, success, out.x_adv, X0,
                                       adv, budget_ratio)
        metrics["queries"] = {
            "total": int(out.queries.sum()),
            "mean_per_attacked_sample": float(out.queries.mean()) if len(att) else 0.0,
            "kind": "black-box label queries" if cfg.knowledge == "black-box"
                    else "white-box model evaluations (forward + gradient)",
        }
        return {**common, "evaluation": "adaptive_attack" if adaptive else "attack",
                "intentionally_vulnerable": not adaptive, "attacker_capabilities": caps,
                "target_system": system.describe(), "attack_config": cfg.model_dump(),
                "defense_config": system.defense, "parent_experiment_id": req.parent_experiment_id,
                "metrics": metrics, "samples": samples}

    def _defend(self, req: DefendRequest, parent: ExperimentRecord, deadline: float) -> dict[str, Any]:
        preq = AttackRequest.model_validate(parent.request)
        scen, ds, common = self._context(preq.scenario_id, preq.model_id, preq.seed)
        if common["model"]["artifact_sha256"] != parent.model["artifact_sha256"]:
            raise InvalidExperiment("model artifact hash differs from the parent experiment; "
                                    "the perturbations cannot be replayed faithfully")
        system = build_system(preq.model_id, scen.dataset_id, preq.seed, req.defense)
        cfg = preq.attack
        idx = np.array([s.test_index for s in parent.samples], dtype=int)
        X0, y = ds.X_test[idx], ds.y_test[idx]
        if not np.array_equal(X0, np.array([s.x_original for s in parent.samples])):
            raise InvalidExperiment("parent experiment inputs do not match the regenerated dataset")
        # Only inputs the parent actually perturbed are replayed.
        att = np.array([i for i, s in enumerate(parent.samples) if s.x_adversarial is not None], dtype=int)
        x_adv = (np.array([parent.samples[i].x_adversarial for i in att]) if len(att)
                 else np.empty((0, X0.shape[1])))
        budget_ratio = enforce_constraints(X0[att], x_adv, cfg.epsilon, cfg.norm,
                                           ds.domain_low, ds.domain_high)
        clean = system.decide(X0)
        correct = clean.labels == y
        # Replay success is only meaningful where the defended system was right on the clean input.
        keep = att[correct[att]]
        keep_pos = np.flatnonzero(correct[att])
        adv = system.decide(x_adv[keep_pos]) if len(keep) else None
        queries = np.zeros(len(keep), dtype=np.int64)
        samples, success = self._samples(idx, X0, y, clean, keep, x_adv[keep_pos], adv, queries)
        metrics = self._attack_metrics(system, ds, cfg, idx, y, correct, keep, success, x_adv[keep_pos],
                                       X0, adv, budget_ratio)
        pm = parent.metrics
        metrics["comparison"] = {
            "clean_accuracy": {"undefended": pm["clean_accuracy"], "defended": metrics["clean_accuracy"]},
            "adversarial_accuracy": {"undefended": pm["adversarial_accuracy"],
                                     "defended": metrics["adversarial_accuracy"]},
            "attack_success_rate": {"undefended": pm["attack_success_rate"],
                                    "defended": metrics["attack_success_rate"]},
        }
        metrics["notes"] = [
            "Non-adaptive replay: the adversarial inputs were crafted against the undefended model "
            f"in {parent.experiment_id} and re-submitted unchanged.",
            "A low replay success rate does not establish robustness. Run an adaptive attack "
            "(a new attack against the defended system) before drawing conclusions.",
        ]
        caps = dict(parent.attacker_capabilities or {})
        caps["knows_defense"] = False
        caps["note"] = "perturbations replayed from the parent experiment; no new queries/gradients"
        return {**common, "evaluation": "replay", "intentionally_vulnerable": False,
                "attacker_capabilities": caps, "target_system": system.describe(),
                "attack_config": cfg.model_dump(), "defense_config": system.defense,
                "parent_experiment_id": parent.experiment_id, "metrics": metrics, "samples": samples}

    @staticmethod
    def _samples(idx, X0, y, clean, att, x_adv, adv, queries):
        """Build per-sample results. ``att`` are positions (into idx) that were attacked."""
        pos = {int(p): k for k, p in enumerate(att)}
        samples, success = [], np.zeros(len(idx), dtype=bool)
        for p, ti in enumerate(idx):
            s = SampleResult(
                test_index=int(ti), y_true=int(y[p]), x_original=X0[p].tolist(),
                pred_original=int(clean.labels[p]), p1_original=float(clean.p1[p]),
                confidence_original=float(clean.confidence[p]),
                clean_correct=bool(clean.labels[p] == y[p]),
            )
            if p in pos:
                k = pos[p]
                delta = x_adv[k] - X0[p]
                label = int(adv.labels[k])
                ok = label != int(y[p]) and label != ABSTAIN
                success[p] = ok
                s.x_adversarial = x_adv[k].tolist()
                s.pred_adversarial = label
                s.p1_adversarial = float(adv.p1[k])
                s.confidence_adversarial = float(adv.confidence[k])
                s.perturbation_linf = float(lp_norm(delta, "linf")[0])
                s.perturbation_l2 = float(lp_norm(delta, "l2")[0])
                s.attack_success = bool(ok)
                s.queries = int(queries[k])
            samples.append(s)
        return samples, success

    def _attack_metrics(self, system, ds, cfg, idx, y, correct, att, success, x_adv, X0, adv,
                        budget_ratio) -> dict[str, Any]:
        n = len(idx)
        m = self._clean_metrics(system, ds)
        m["clean_accuracy_selected"] = _ratio(int(correct.sum()), n, "correct clean decisions / selected samples")
        # Correct samples that were not attacked (or not replayed) remain correct.
        m["adversarial_accuracy"] = _ratio(
            int(np.sum(correct) - np.sum(success)), n,
            "selected samples still decided correctly after the attack / selected samples")
        m["attack_success_rate"] = _ratio(
            int(success.sum()), len(att),
            "attacked samples whose clean input was decided correctly and whose adversarial input "
            "is decided as a different class (not an abstention) / attacked samples decided correctly "
            "on their clean input")
        if adv is not None and system.reject_threshold is not None:
            m["abstained_on_adversarial"] = _ratio(int(np.sum(adv.labels == ABSTAIN)), len(att),
                                                   "adversarial inputs the system abstained on / attacked samples")
        if len(att):
            deltas = x_adv - X0[att]
            linf, l2, succ_mask = lp_norm(deltas, "linf"), lp_norm(deltas, "l2"), success[att]
        else:
            linf, l2, succ_mask = np.array([]), np.array([]), np.array([], dtype=bool)
        m["perturbation"] = {"norm": cfg.norm, "epsilon": cfg.epsilon, "max_budget_ratio": budget_ratio,
                             "attacked": _stats(linf, l2),
                             "successful": _stats(linf[succ_mask], l2[succ_mask])}
        return m
