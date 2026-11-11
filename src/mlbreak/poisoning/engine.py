"""ML02 experiment engine: clean training → label-flipping poisoning → defensive replay.

``PoisoningLab`` reuses the ML01 :class:`~mlbreak.experiments.ExperimentEngine` only for what is common
to every experiment family — the concurrency gate + wall-clock deadline (``engine.gated``), the evidence
store, the clean base-model cache and the reproducibility block — and implements its own evaluations,
metrics and records. ML01 code paths are unchanged.

Evaluations:

* ``clean_training`` (mode ``baseline``) — train on the clean labels, evaluate on the clean test set.
* ``poisoning`` (mode ``attack``) — flip training labels (random or targeted), RETRAIN, evaluate, and
  compare with the clean model trained on the same features and seed.
* ``poisoning_replay`` (mode ``defend``) — take the EXACT poisoned label vector of a stored attack
  experiment (integrity-verified; the regenerated poisoned model must reproduce the parent's artifact
  hash), apply kNN label sanitisation, retrain, and re-evaluate. The same defense is also applied to the
  clean labels to measure its cost when there is no attack.

Exact replay (``replay``) re-executes a stored record from its request and compares every reproducible
fingerprint; it first requires the stored record to pass the integrity check.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ..attacks import ExperimentTimeout, _check_deadline
from ..data import Dataset, load_dataset
from ..defenses import base_model
from ..hashing import sha256_arrays, sha256_json
from ..integrity import record_sha256
from ..models import DifferentiableClassifier, train_model
from ..owasp import get_category
from .attacks import (InvalidPoisonConfig, PoisonBudgetViolation, enforce_poison_budget, flip_labels,
                      knn_label_check, n_flips, sanitize, select_poison_indices)
from .integrity import label_hashes, results_fingerprint, verify_poisoning_record
from .metrics import classify_test_outcome, derive_metrics
from .schemas import (SCENARIO_ID, KnnLabelSanitization, LabelFlipConfig, PoisonAttackRequest,
                      PoisonBaselineRequest, PoisonDefendRequest, PoisoningRecord, TestSampleResult,
                      TrainingSampleResult)

if TYPE_CHECKING:  # pragma: no cover
    from ..experiments import ExperimentEngine

OWASP_ID = "ML02:2023"

# Trained-on-labels model cache: (model_id, dataset_id, seed, sha256(X, y)) → model. Only completed fits are
# stored, so a timed-out experiment never leaves a partial model behind.
_FIT_CACHE_SIZE = 32
_fit_cache: "OrderedDict[tuple, DifferentiableClassifier]" = OrderedDict()
_fit_lock = threading.Lock()


def clear_fit_cache() -> None:
    with _fit_lock:
        _fit_cache.clear()


def fit_on(model_id: str, ds: Dataset, seed: int, X: np.ndarray, y: np.ndarray, note: str) -> DifferentiableClassifier:
    key = (model_id, ds.spec.dataset_id, seed, sha256_arrays(np.asarray(X, float), np.asarray(y, np.int64)))
    with _fit_lock:
        if key in _fit_cache:
            _fit_cache.move_to_end(key)
            return _fit_cache[key]
    model = train_model(model_id, X, y, seed, training_note=note)
    with _fit_lock:
        _fit_cache[key] = model
        while len(_fit_cache) > _FIT_CACHE_SIZE:
            _fit_cache.popitem(last=False)
    return model


class PoisoningLab:
    def __init__(self, engine: "ExperimentEngine"):
        self.engine = engine
        self.store = engine.store

    # ------------------------------------------------------------------ API
    def run_baseline(self, req: PoisonBaselineRequest, replay_of: str | None = None) -> PoisoningRecord:
        return self._execute("baseline", req, self._baseline, replay_of)

    def run_attack(self, req: PoisonAttackRequest, replay_of: str | None = None) -> PoisoningRecord:
        return self._execute("attack", req, self._attack, replay_of)

    def run_defend(self, req: PoisonDefendRequest, replay_of: str | None = None) -> PoisoningRecord:
        parent = self._load_parent(req.parent_experiment_id)
        return self._execute("defend", req, lambda r, d: self._defend(r, parent, d), replay_of)

    def verify(self, rec: PoisoningRecord) -> dict[str, Any]:
        return verify_poisoning_record(rec)

    def require_integrity(self, rec: PoisoningRecord) -> dict[str, Any]:
        from ..experiments import EvidenceIntegrityError
        report = verify_poisoning_record(rec)
        if not report["ok"]:
            raise EvidenceIntegrityError(rec.experiment_id, report)
        return report

    def replay(self, original: PoisoningRecord) -> dict[str, Any]:
        """Re-execute a stored ML02 record (integrity-verified first) and compare its fingerprints."""
        from ..experiments import ENVIRONMENT_KEYS, InvalidExperiment
        if original.status != "completed":
            raise InvalidExperiment("only completed experiments can be replayed")
        integrity = self.require_integrity(original)
        oid = original.experiment_id
        if original.mode == "baseline":
            new = self.run_baseline(PoisonBaselineRequest.model_validate(original.request), replay_of=oid)
        elif original.mode == "attack":
            new = self.run_attack(PoisonAttackRequest.model_validate(original.request), replay_of=oid)
        else:
            new = self.run_defend(PoisonDefendRequest.model_validate(original.request), replay_of=oid)
        o, n = original.reproducibility, new.reproducibility
        pairs = [("status", original.status, new.status),
                 ("config_sha256", o.get("config_sha256"), n.get("config_sha256")),
                 ("train_split_sha256", original.dataset["train_sha256"], new.dataset.get("train_sha256")),
                 ("test_split_sha256", original.dataset["test_sha256"], new.dataset.get("test_sha256"))]
        for which in ("clean", "poisoned", "defended", "clean_sanitized"):
            a, b = original.models.get(which), new.models.get(which)
            if a is not None or b is not None:
                pairs.append((f"{which}_model_artifact_sha256", (a or {}).get("artifact_sha256"),
                              (b or {}).get("artifact_sha256")))
        for key in ("observed_labels_sha256", "sanitized_labels_sha256", "poisoned_indices"):
            a, b = (original.poisoning or {}).get(key), (new.poisoning or {}).get(key)
            if a is not None or b is not None:
                pairs.append((key, a, b))
        pairs += [("metrics_sha256", o.get("metrics_sha256"), n.get("metrics_sha256")),
                  ("results_sha256", o.get("results_sha256"), n.get("results_sha256"))]
        checks = [{"check": k, "original": a, "replay": b, "match": a == b} for k, a, b in pairs]
        env = {k: {"original": o.get(k), "replay": n.get(k)} for k in ENVIRONMENT_KEYS if o.get(k) != n.get(k)}
        notes = (["The runtime environment differs from the original; bit-identical results are only expected on "
                  "the same platform and library versions."] if env else [])
        return {"original_experiment_id": oid, "replay_experiment_id": new.experiment_id,
                "identical": all(c["match"] for c in checks), "checks": checks, "original_integrity": integrity,
                "environment_differences": env, "notes": notes}

    def boundary(self, experiment_id: str, which: str, resolution: int) -> dict[str, Any]:
        """Decision-boundary grid of one of the models of a stored ML02 record.

        The model is RE-TRAINED from the record's own label vectors and must reproduce the artifact hash
        stored in the record, so the plot is provably of the model the experiment evaluated.
        """
        from ..experiments import InvalidExperiment
        rec = self.store.load(experiment_id)
        if not isinstance(rec, PoisoningRecord):
            raise InvalidExperiment(f"{experiment_id} is not an ML02 poisoning experiment")
        if rec.status != "completed":
            raise InvalidExperiment("the experiment did not complete")
        expected = (rec.models.get(which) or {}).get("artifact_sha256")
        if expected is None:
            raise InvalidExperiment(f"experiment {experiment_id} has no '{which}' model")
        self.require_integrity(rec)
        preq_model, seed = rec.model["model_id"], rec.seed

        def work(deadline: float) -> dict[str, Any]:
            ds = load_dataset(rec.dataset["dataset_id"], seed)
            X = ds.X_train
            rows = rec.training_samples
            if not np.array_equal(X, np.array([r.x for r in rows])):
                raise InvalidExperiment("the record's training inputs do not match the regenerated dataset")
            if which == "clean":
                model = base_model(preq_model, ds.spec.dataset_id, seed)
            elif which == "poisoned":
                model = fit_on(preq_model, ds, seed, X, np.array([r.y_observed for r in rows]), "poisoned labels")
            elif which == "defended":
                keep = np.array([not r.removed for r in rows])
                y = np.array([r.y_sanitized if r.y_sanitized is not None else -1 for r in rows])
                model = fit_on(preq_model, ds, seed, X[keep], y[keep], "sanitised labels")
            else:  # clean_sanitized
                cfg = KnnLabelSanitization.model_validate(rec.defense_config)
                _, fl, maj = knn_label_check(X, ds.y_train, cfg)
                keep, y = sanitize(ds.y_train, fl, maj, cfg.action)
                model = fit_on(preq_model, ds, seed, X[keep], y[keep], "sanitised CLEAN labels")
            _check_deadline(deadline)
            if model.artifact_sha256() != expected:
                raise InvalidExperiment(f"regenerated '{which}' model does not match the recorded artifact hash")
            xs = np.linspace(*ds.spec.domain[0], resolution)
            ys = np.linspace(*ds.spec.domain[1], resolution)
            gx, gy = np.meshgrid(xs, ys)
            p1 = model.proba(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
            return {"x": xs.round(5).tolist(), "y": ys.round(5).tolist(), "p1": p1.round(5).tolist(),
                    "abstain": np.zeros(gx.shape, dtype=int).tolist(), "which": which,
                    "experiment_id": experiment_id,
                    "target_system": {"defense": rec.defense_config if which == "defended" else None,
                                      "model_artifact_sha256": expected, "decision_policy": "argmax"},
                    "model": model.describe()}
        return self.engine.gated(work)

    # ------------------------------------------------------------ internals
    def _execute(self, mode: str, req, fn: Callable[[Any, float], dict[str, Any]],
                 replay_of: str | None) -> PoisoningRecord:
        from ..experiments import _new_id, _now, reproducibility_info

        def work(deadline: float) -> PoisoningRecord:
            started = time.monotonic()
            exp_id = _new_id()
            try:
                body = fn(req, deadline)
                status, error = "completed", None
            except (ExperimentTimeout, PoisonBudgetViolation) as exc:
                body, status, error = self._failure_body(req, mode), "failed", f"{type(exc).__name__}: {exc}"
            body.setdefault("metrics", {})
            body.setdefault("training_samples", [])
            body.setdefault("test_samples", [])
            done = status == "completed"
            repro = reproducibility_info()
            repro["config_sha256"] = sha256_json(req.model_dump(mode="json"))
            repro["metrics_sha256"] = sha256_json(body["metrics"]) if done else None
            repro["results_sha256"] = (results_fingerprint((body["training_samples"], body["test_samples"],
                                                            body["metrics"])) if done else None)
            repro["replay_command"] = f"mlbreak replay {exp_id}"
            repro["verify_command"] = f"mlbreak check {exp_id}"
            rec = PoisoningRecord(experiment_id=exp_id, created_at=_now(), mode=mode, status=status, error=error,
                                  replay_of=replay_of, request=req.model_dump(mode="json"), reproducibility=repro,
                                  duration_ms=round((time.monotonic() - started) * 1000, 2), **body)
            rec.reproducibility["record_sha256"] = record_sha256(rec)
            self.store.save(rec)
            return rec
        return self.engine.gated(work)

    def _context(self, model_id: str, seed: int, deadline: float | None):
        from ..experiments import SCENARIOS
        scen = SCENARIOS[SCENARIO_ID]
        cat = get_category(OWASP_ID)
        ds = load_dataset(scen.dataset_id, seed)
        clean = base_model(model_id, scen.dataset_id, seed)  # identical config/data/seed → shared with ML01
        _check_deadline(deadline)
        common = {
            "scenario_id": scen.scenario_id,
            "owasp": {"id": cat.id, "name": cat.name, "taxonomy": cat.taxonomy,
                      "taxonomy_version": cat.taxonomy_version, "reference_url": cat.reference_url},
            "model": clean.describe(), "dataset": ds.describe(), "seed": seed,
        }
        return ds, clean, common

    def _failure_body(self, req, mode: str) -> dict[str, Any]:
        r = req
        if isinstance(req, PoisonDefendRequest):
            r = PoisonAttackRequest.model_validate(self.store.load(req.parent_experiment_id).request)
        ds, clean, common = self._context(r.model_id, r.seed, None)
        evaluation = {"baseline": "clean_training", "attack": "poisoning", "defend": "poisoning_replay"}[mode]
        return {**common, "evaluation": evaluation, "intentionally_vulnerable": mode == "attack",
                "attacker_capabilities": None, "models": {"clean": clean.describe()},
                "target_system": {"model_artifact_sha256": None},
                "poison_config": getattr(r, "poison", None) and r.poison.model_dump(),
                "defense_config": getattr(req, "defense", None) and req.defense.model_dump(),
                "parent_experiment_id": getattr(req, "parent_experiment_id", None)}

    def _load_parent(self, parent_id: str) -> PoisoningRecord:
        from ..experiments import InvalidExperiment
        parent = self.store.load(parent_id)
        if not isinstance(parent, PoisoningRecord):
            raise InvalidExperiment(f"parent {parent_id} is not an ML02 poisoning experiment")
        if parent.status != "completed":
            raise InvalidExperiment(f"parent {parent_id} did not complete (status {parent.status})")
        if parent.mode != "attack" or parent.evaluation != "poisoning":
            raise InvalidExperiment(f"parent {parent_id} must be a poisoning attack experiment "
                                    f"(got mode={parent.mode}, evaluation={parent.evaluation})")
        self.require_integrity(parent)
        return parent

    @staticmethod
    def _capabilities(cfg: LabelFlipConfig, n_train: int) -> dict[str, Any]:
        goal = ("degrade overall test accuracy (availability)" if cfg.strategy == "random" else
                f"make the model misclassify class-{cfg.target_class} inputs at test time (targeted integrity)")
        return {
            "attack_family": "data poisoning — label flipping",
            "can_modify": f"the LABELS of up to {n_flips(cfg, n_train)}/{n_train} training samples "
                          f"({cfg.fraction:.0%} of the training set) before training",
            "cannot": "modify features, test data, the learning algorithm, hyper-parameters, the seed or the "
                      "trained model; cannot add or remove training samples",
            "knowledge": cfg.knowledge,
            "selection": cfg.strategy if cfg.strategy == "random" else f"targeted/{cfg.selection}",
            "goal": goal,
            "timing": "before training (the victim retrains on the poisoned set)",
        }

    # ---------------------------------------------------------- evaluations
    def _baseline(self, req: PoisonBaselineRequest, deadline: float) -> dict[str, Any]:
        ds, clean, common = self._context(req.model_id, req.seed, deadline)
        tr_pred = clean.predict(ds.X_train)
        te_pred, te_p1 = clean.predict(ds.X_test), clean.proba(ds.X_test)
        train = [TrainingSampleResult(train_index=i, x=ds.X_train[i].tolist(), y_true=int(ds.y_train[i]),
                                      y_observed=int(ds.y_train[i]), poisoned=False, pred_clean=int(tr_pred[i]))
                 for i in range(len(ds.y_train))]
        test = [TestSampleResult(test_index=i, x=ds.X_test[i].tolist(), y_true=int(ds.y_test[i]),
                                 pred_clean=int(te_pred[i]), p1_clean=float(te_p1[i])) for i in range(len(ds.y_test))]
        return {**common, "evaluation": "clean_training", "intentionally_vulnerable": True,
                "attacker_capabilities": None, "models": {"clean": clean.describe(), "poisoned": None,
                                                          "defended": None},
                "target_system": {"model_artifact_sha256": clean.artifact_sha256(), "defense": None,
                                  "decision_policy": "argmax", "trained_on": "clean labels"},
                "poison_config": None, "poisoning": label_hashes(train), "defense_config": None,
                "metrics": derive_metrics("clean_training", train, test, None, None),
                "training_samples": train, "test_samples": test}

    def _attack(self, req: PoisonAttackRequest, deadline: float) -> dict[str, Any]:
        from ..experiments import InvalidExperiment
        ds, clean, common = self._context(req.model_id, req.seed, deadline)
        cfg = req.poison
        margin = np.abs(clean.logit(ds.X_train)) if cfg.selection == "boundary" else None
        try:
            idx, rule = select_poison_indices(cfg, ds.y_train, req.seed, margin)
        except InvalidPoisonConfig as exc:
            raise InvalidExperiment(str(exc)) from exc
        y_obs = flip_labels(ds.y_train, idx)
        enforce_poison_budget(ds.y_train, y_obs, idx, cfg)
        poisoned = fit_on(req.model_id, ds, req.seed, ds.X_train, y_obs,
                          f"poisoned labels ({cfg.strategy}, {len(idx)} flips)")
        _check_deadline(deadline)
        train, test = self._rows(ds, clean, y_obs, poisoned)
        info = {"selection_rule": rule, "poisoned_indices": [int(i) for i in idx], "n_flips": int(len(idx)),
                "budget": n_flips(cfg, len(ds.y_train)), "budget_verified": True, **label_hashes(train)}
        return {**common, "evaluation": "poisoning", "intentionally_vulnerable": True,
                "attacker_capabilities": self._capabilities(cfg, len(ds.y_train)),
                "models": {"clean": clean.describe(), "poisoned": poisoned.describe(), "defended": None},
                "target_system": {"model_artifact_sha256": poisoned.artifact_sha256(), "defense": None,
                                  "decision_policy": "argmax", "trained_on": "poisoned labels"},
                "poison_config": cfg.model_dump(), "poisoning": info, "defense_config": None,
                "metrics": derive_metrics("poisoning", train, test, cfg.model_dump(), None),
                "training_samples": train, "test_samples": test}

    @staticmethod
    def _rows(ds, clean, y_obs, poisoned, defended=None, clean_san=None, extra=None):
        tr_clean, tr_pois = clean.predict(ds.X_train), poisoned.predict(ds.X_train)
        tr_def = defended.predict(ds.X_train) if defended is not None else None
        te_clean, te_p1c = clean.predict(ds.X_test), clean.proba(ds.X_test)
        te_pois, te_p1p = poisoned.predict(ds.X_test), poisoned.proba(ds.X_test)
        te_def = defended.predict(ds.X_test) if defended is not None else None
        te_p1d = defended.proba(ds.X_test) if defended is not None else None
        te_cs = clean_san.predict(ds.X_test) if clean_san is not None else None
        train = []
        for i in range(len(ds.y_train)):
            r = TrainingSampleResult(train_index=i, x=ds.X_train[i].tolist(), y_true=int(ds.y_train[i]),
                                     y_observed=int(y_obs[i]), poisoned=bool(y_obs[i] != ds.y_train[i]),
                                     pred_clean=int(tr_clean[i]), pred_poisoned=int(tr_pois[i]))
            if defended is not None:
                r.pred_defended = int(tr_def[i])
                for k, v in extra(i).items():
                    setattr(r, k, v)
            train.append(r)
        test = []
        for i in range(len(ds.y_test)):
            y = int(ds.y_test[i])
            t = TestSampleResult(test_index=i, x=ds.X_test[i].tolist(), y_true=y, pred_clean=int(te_clean[i]),
                                 p1_clean=float(te_p1c[i]), pred_poisoned=int(te_pois[i]), p1_poisoned=float(te_p1p[i]))
            t.outcome = classify_test_outcome(y, t.pred_clean, t.pred_poisoned)
            if defended is not None:
                t.pred_defended, t.p1_defended = int(te_def[i]), float(te_p1d[i])
                t.pred_clean_sanitized = int(te_cs[i])
                t.defended_outcome = classify_test_outcome(y, t.pred_clean, t.pred_defended)
            test.append(t)
        return train, test

    def _defend(self, req: PoisonDefendRequest, parent: PoisoningRecord, deadline: float) -> dict[str, Any]:
        from ..experiments import InvalidExperiment
        preq = PoisonAttackRequest.model_validate(parent.request)
        ds, clean, common = self._context(preq.model_id, preq.seed, deadline)
        # ---- lineage: the defense must see EXACTLY the parent's poisoned training set ----
        problems = []
        fp = ds.fingerprint()
        if (parent.dataset.get("train_sha256"), parent.dataset.get("test_sha256")) != (fp["train_sha256"],
                                                                                      fp["test_sha256"]):
            problems.append("dataset split hashes")
        if parent.model.get("artifact_sha256") != clean.artifact_sha256():
            problems.append("clean-model artifact hash")
        rows = parent.training_samples
        if not np.array_equal(ds.X_train, np.array([r.x for r in rows])) or \
                not np.array_equal(ds.y_train, np.array([r.y_true for r in rows])):
            problems.append("training inputs / clean labels")
        if problems:
            raise InvalidExperiment(f"parent {parent.experiment_id} is incompatible: differs in " + ", ".join(problems))
        y_obs = np.array([r.y_observed for r in rows], dtype=np.int64)
        poisoned = fit_on(preq.model_id, ds, preq.seed, ds.X_train, y_obs, "poisoned labels (replayed)")
        _check_deadline(deadline)
        if poisoned.artifact_sha256() != parent.models["poisoned"]["artifact_sha256"]:
            raise InvalidExperiment("the regenerated poisoned model does not reproduce the parent's artifact hash")
        cfg = req.defense
        agreement, flagged, majority = knn_label_check(ds.X_train, y_obs, cfg)
        keep, y_san = sanitize(y_obs, flagged, majority, cfg.action)
        defended = fit_on(preq.model_id, ds, preq.seed, ds.X_train[keep], y_san[keep],
                          f"sanitised labels (kNN k={cfg.k}, threshold={cfg.agreement_threshold}, {cfg.action})")
        _check_deadline(deadline)
        _, flagged_c, majority_c = knn_label_check(ds.X_train, ds.y_train, cfg)
        keep_c, y_c = sanitize(ds.y_train, flagged_c, majority_c, cfg.action)
        clean_san = fit_on(preq.model_id, ds, preq.seed, ds.X_train[keep_c], y_c[keep_c],
                           "sanitised CLEAN labels (defense cost without attack)")
        _check_deadline(deadline)
        pois_mask = y_obs != ds.y_train

        def extra(i: int) -> dict[str, Any]:
            det = {(True, True): "true_positive", (True, False): "false_positive",
                   (False, True): "false_negative", (False, False): "true_negative"}[(bool(flagged[i]),
                                                                                      bool(pois_mask[i]))]
            return {"knn_agreement": float(agreement[i]), "flagged": bool(flagged[i]),
                    "y_sanitized": int(y_san[i]) if keep[i] else None, "removed": bool(not keep[i]),
                    "detection": det, "flagged_on_clean": bool(flagged_c[i])}

        train, test = self._rows(ds, clean, y_obs, poisoned, defended, clean_san, extra)
        pcfg = parent.poison_config
        info = {**{k: v for k, v in (parent.poisoning or {}).items()}, **label_hashes(train)}
        metrics = derive_metrics("poisoning_replay", train, test, pcfg, cfg.model_dump())
        lineage = {
            "relation": "defensive_replay_of", "parent_experiment_id": parent.experiment_id,
            "parent_evaluation": parent.evaluation, "parent_schema_version": parent.schema_version,
            "parent_config_sha256": parent.reproducibility.get("config_sha256"),
            "parent_results_sha256": parent.reproducibility.get("results_sha256"),
            "parent_record_sha256": parent.reproducibility.get("record_sha256"),
            "shared": {"scenario_id": parent.scenario_id, "model_id": preq.model_id, "seed": preq.seed,
                       "dataset_train_sha256": fp["train_sha256"], "dataset_test_sha256": fp["test_sha256"],
                       "clean_model_artifact_sha256": clean.artifact_sha256(),
                       "poisoned_model_artifact_sha256": poisoned.artifact_sha256(),
                       "observed_labels_sha256": info.get("observed_labels_sha256")},
            "checks_passed": ["parent completed", "parent is a poisoning attack", "parent evidence integrity",
                              "dataset split hashes", "clean-model artifact hash", "training inputs / clean labels",
                              "regenerated poisoned model reproduces the parent's artifact hash"],
        }
        caps = dict(parent.attacker_capabilities or {})
        caps["knows_defense"] = False
        caps["note"] = "poisoned labels replayed from the parent experiment; the attacker did not adapt"
        return {**common, "evaluation": "poisoning_replay", "intentionally_vulnerable": False,
                "attacker_capabilities": caps,
                "models": {"clean": clean.describe(), "poisoned": poisoned.describe(),
                           "defended": defended.describe(), "clean_sanitized": clean_san.describe()},
                "target_system": {"model_artifact_sha256": defended.artifact_sha256(), "defense": cfg.model_dump(),
                                  "decision_policy": "argmax", "trained_on": "sanitised poisoned labels"},
                "poison_config": pcfg, "poisoning": info, "defense_config": cfg.model_dump(),
                "parent_experiment_id": parent.experiment_id, "lineage": lineage, "metrics": metrics,
                "training_samples": train, "test_samples": test}
