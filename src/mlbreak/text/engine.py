"""ML01 spam-filter evasion engine: clean evaluation → character-level attack → defense replay / adaptive attack.

Evaluations (same names and meaning as the 2-D ML01 lab):

* ``clean`` (mode ``baseline``) — the (optionally defended) classifier on the clean test split;
* ``attack`` — a new DeepWordBug-style attack against the UNDEFENDED classifier;
* ``replay`` (mode ``defend``) — the EXACT adversarial messages of a stored attack re-submitted to a defended
  system (non-adaptive attacker);
* ``adaptive_attack`` — a NEW attack run against the defended system (it queries the defended pipeline, so
  normalisation is inside its loop), with the same selected messages and the same text-preservation
  constraints as its parent.

All model access during an attack goes through a counting, budget-enforcing oracle. Shares the concurrency
gate, deadline, evidence store, lineage verification and paired statistics with the other labs.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ..attacks import ExperimentTimeout, QueryBudgetExceeded, _check_deadline
from ..hashing import sha256_json
from ..integrity import record_sha256
from ..owasp import get_category
from .attacks import TextConstraintViolation, deepwordbug_attack, validate_constraints
from .data import load_text_dataset
from .defenses import base_text_model, build_text_system
from .integrity import results_fingerprint, verify_text_record
from .metrics import derive_text_metrics
from .schemas import (SCENARIO_ID, TestPrediction, TextAttackConfig, TextAttackRequest, TextBaselineRequest,
                      TextClassifyRequest, TextDefendRequest, TextEvasionRecord, TextSampleResult)

if TYPE_CHECKING:  # pragma: no cover
    from ..experiments import ExperimentEngine

# Attack fields that define text preservation and the goal: an adaptive attack must keep them identical to its
# parent so the comparison is like for like. Effort knobs (budget, candidates, edits per word) may differ.
PRESERVATION_FIELDS = ("goal", "min_target_confidence", "operations", "charsets", "max_edit_distance",
                       "max_char_ratio", "max_modified_word_ratio", "protect_numbers", "protect_stopwords",
                       "min_word_length")


class TextLab:
    def __init__(self, engine: "ExperimentEngine"):
        self.engine = engine
        self.store = engine.store

    # ------------------------------------------------------------------ API
    def run_baseline(self, req: TextBaselineRequest, replay_of: str | None = None) -> TextEvasionRecord:
        return self._execute("baseline", req, self._baseline, replay_of)

    def run_attack(self, req: TextAttackRequest, replay_of: str | None = None) -> TextEvasionRecord:
        from ..experiments import InvalidExperiment
        if req.parent_experiment_id and req.target_defense is None:
            raise InvalidExperiment("parent_experiment_id is only valid for adaptive attacks (set target_defense)")
        parent = self._load_parent(req.parent_experiment_id) if req.parent_experiment_id else None
        return self._execute("attack", req, lambda r, d: self._attack(r, d, parent), replay_of)

    def run_defend(self, req: TextDefendRequest, replay_of: str | None = None) -> TextEvasionRecord:
        parent = self._load_parent(req.parent_experiment_id)
        return self._execute("defend", req, lambda r, d: self._defend(r, parent, d), replay_of)

    def verify(self, rec: TextEvasionRecord) -> dict[str, Any]:
        return verify_text_record(rec)

    def classify(self, req: TextClassifyRequest) -> dict[str, Any]:
        """Classify one message with the bundled (optionally defended) classifier. Not stored as evidence."""
        def work(deadline: float) -> dict[str, Any]:
            system = build_text_system(req.model_id, req.seed, req.defense, deadline)
            p = float(system.p_spam([req.text])[0])
            norm = system.model.normalizer(req.text) if system.model.normalizer is not None else None
            return {"text": req.text, "normalized_text": norm, "p_spam": p, "p_ham": 1 - p,
                    "prediction": int(p >= 0.5), "label": "spam" if p >= 0.5 else "ham",
                    "target_system": {k: v for k, v in system.describe().items() if k != "model"},
                    "model": system.model.describe()}
        return self.engine.gated(work)

    def test_messages(self, model_id: str, seed: int, label: int | None, limit: int) -> dict[str, Any]:
        """Held-out test messages with the undefended classifier's prediction (for choosing what to attack)."""
        def work(deadline: float) -> dict[str, Any]:
            ds = load_text_dataset(seed)
            model = base_text_model(model_id, seed)
            _check_deadline(deadline)
            idx = [i for i, y in enumerate(ds.test_labels) if label is None or y == label][:limit]
            p = model.p_spam([ds.test_texts[i] for i in idx])
            return {"dataset": ds.describe(), "model": model.describe(),
                    "messages": [{"test_index": i, "text": ds.test_texts[i], "y_true": ds.test_labels[i],
                                  "p_spam": float(pi), "pred": int(pi >= 0.5)} for i, pi in zip(idx, p)]}
        return self.engine.gated(work)

    def replay(self, original: TextEvasionRecord) -> dict[str, Any]:
        from ..experiments import InvalidExperiment, environment_differences, environment_note
        if original.status != "completed":
            raise InvalidExperiment("only completed experiments can be replayed")
        integrity = self.engine.require_verified(original)
        oid = original.experiment_id
        if original.mode == "baseline":
            new = self.run_baseline(TextBaselineRequest.model_validate(original.request), replay_of=oid)
        elif original.mode == "attack":
            new = self.run_attack(TextAttackRequest.model_validate(original.request), replay_of=oid)
        else:
            new = self.run_defend(TextDefendRequest.model_validate(original.request), replay_of=oid)
        o, n = original.reproducibility, new.reproducibility
        pairs = [("status", original.status, new.status),
                 ("config_sha256", o.get("config_sha256"), n.get("config_sha256")),
                 ("train_split_sha256", original.dataset["train_sha256"], new.dataset.get("train_sha256")),
                 ("test_split_sha256", original.dataset["test_sha256"], new.dataset.get("test_sha256")),
                 ("model_artifact_sha256", original.model["artifact_sha256"], new.model.get("artifact_sha256")),
                 ("target_system_artifact_sha256", original.target_system.get("model_artifact_sha256"),
                  new.target_system.get("model_artifact_sha256")),
                 ("adversarial_texts_sha256", sha256_json([s.text_adversarial for s in original.samples]),
                  sha256_json([s.text_adversarial for s in new.samples])),
                 ("metrics_sha256", o.get("metrics_sha256"), n.get("metrics_sha256")),
                 ("results_sha256", o.get("results_sha256"), n.get("results_sha256"))]
        checks = [{"check": k, "original": a, "replay": b, "match": a == b} for k, a, b in pairs]
        env = environment_differences(o, n)
        identical = all(c["match"] for c in checks)
        return {"original_experiment_id": oid, "replay_experiment_id": new.experiment_id, "identical": identical,
                "checks": checks, "original_integrity": integrity, "environment_differences": env,
                "notes": environment_note(env, identical)}

    # ------------------------------------------------------------ internals
    def _execute(self, mode: str, req, fn: Callable[[Any, float], dict[str, Any]],
                 replay_of: str | None) -> TextEvasionRecord:
        from ..experiments import _new_id, _now, reproducibility_info

        def work(deadline: float) -> TextEvasionRecord:
            started, exp_id = time.monotonic(), _new_id()
            try:
                body = fn(req, deadline)
                status, error = "completed", None
            except (ExperimentTimeout, QueryBudgetExceeded, TextConstraintViolation) as exc:
                body, status, error = self._failure_body(req, mode), "failed", f"{type(exc).__name__}: {exc}"
            body.setdefault("metrics", {})
            body.setdefault("samples", [])
            body.setdefault("test_predictions", [])
            done = status == "completed"
            repro = reproducibility_info()
            repro["config_sha256"] = sha256_json(req.model_dump(mode="json"))
            repro["metrics_sha256"] = sha256_json(body["metrics"]) if done else None
            repro["results_sha256"] = (results_fingerprint(body["samples"], body["test_predictions"], body["metrics"])
                                       if done else None)
            repro["replay_command"] = f"mlbreak replay {exp_id}"
            repro["verify_command"] = f"mlbreak check {exp_id}"
            rec = TextEvasionRecord(experiment_id=exp_id, created_at=_now(), mode=mode, status=status, error=error,
                                    replay_of=replay_of, request=req.model_dump(mode="json"), reproducibility=repro,
                                    duration_ms=round((time.monotonic() - started) * 1000, 2), **body)
            rec.reproducibility["record_sha256"] = record_sha256(rec)
            self.store.save(rec)
            return rec
        return self.engine.gated(work)

    def _context(self, model_id: str, seed: int, deadline: float | None):
        from ..experiments import SCENARIOS
        scen = SCENARIOS[SCENARIO_ID]
        cat = get_category(scen.owasp_id)
        ds = load_text_dataset(seed)
        base = base_text_model(model_id, seed)
        _check_deadline(deadline)
        return ds, base, {
            "scenario_id": scen.scenario_id,
            "owasp": {"id": cat.id, "name": cat.name, "taxonomy": cat.taxonomy,
                      "taxonomy_version": cat.taxonomy_version, "reference_url": cat.reference_url},
            "model": base.describe(), "dataset": ds.describe(), "seed": seed}

    def _failure_body(self, req, mode: str) -> dict[str, Any]:
        r = req
        if isinstance(req, TextDefendRequest):
            r = TextAttackRequest.model_validate(self.store.load(req.parent_experiment_id).request)
        _, _, common = self._context(r.model_id, r.seed, None)
        evaluation = {"baseline": "clean", "defend": "replay"}.get(
            mode, "adaptive_attack" if getattr(req, "target_defense", None) is not None else "attack")
        defense = getattr(req, "defense", None) or getattr(req, "target_defense", None)
        return {**common, "evaluation": evaluation, "intentionally_vulnerable": evaluation in ("attack", "clean"),
                "attacker_capabilities": None, "target_system": {"model_artifact_sha256": None},
                "attack_config": getattr(r, "attack", None) and r.attack.model_dump(),
                "defense_config": defense.model_dump() if defense is not None else None,
                "parent_experiment_id": getattr(req, "parent_experiment_id", None)}

    def _load_parent(self, parent_id: str) -> TextEvasionRecord:
        from ..experiments import InvalidExperiment
        parent = self.store.load(parent_id)
        if not isinstance(parent, TextEvasionRecord):
            raise InvalidExperiment(f"parent {parent_id} is not an ML01 spam-filter evasion experiment")
        if parent.status != "completed":
            raise InvalidExperiment(f"parent {parent_id} did not complete (status {parent.status})")
        if parent.mode != "attack" or parent.evaluation != "attack":
            raise InvalidExperiment(f"parent {parent_id} must be a non-adaptive attack against the undefended "
                                    f"classifier (got mode={parent.mode}, evaluation={parent.evaluation})")
        self.engine.require_verified(parent)
        return parent

    def _lineage(self, parent: TextEvasionRecord, relation: str, model_id: str, seed: int, ds, base,
                 extra: list[str]) -> dict[str, Any]:
        from ..experiments import InvalidExperiment
        fp = ds.fingerprint()
        problems = [name for name, ok in (
            ("model", parent.model.get("model_id") == model_id), ("seed", parent.seed == seed),
            ("dataset split hashes", (parent.dataset.get("train_sha256"), parent.dataset.get("test_sha256")) ==
             (fp["train_sha256"], fp["test_sha256"])),
            ("base-model artifact hash", parent.model.get("artifact_sha256") == base.artifact_sha256())) if not ok]
        if problems:
            raise InvalidExperiment(f"parent {parent.experiment_id} is incompatible: differs in " + ", ".join(problems))
        return {"relation": relation, "parent_experiment_id": parent.experiment_id,
                "parent_evaluation": parent.evaluation, "parent_schema_version": parent.schema_version,
                "parent_config_sha256": parent.reproducibility.get("config_sha256"),
                "parent_results_sha256": parent.reproducibility.get("results_sha256"),
                "parent_record_sha256": parent.reproducibility.get("record_sha256"),
                "shared": {"scenario_id": SCENARIO_ID, "model_id": model_id, "seed": seed, **fp,
                           "base_model_artifact_sha256": base.artifact_sha256()},
                "checks_passed": ["parent completed", "parent is a non-adaptive attack on the undefended classifier",
                                  "parent evidence integrity", "model", "seed", "dataset split hashes",
                                  "base-model artifact hash", *extra]}

    @staticmethod
    def _select(ds, cfg: TextAttackConfig, selection) -> list[int]:
        from ..experiments import InvalidExperiment
        src = cfg.source_class
        if selection.indices is not None:
            bad = [i for i in selection.indices if i >= len(ds.test_labels)]
            if bad:
                raise InvalidExperiment(f"test indices must be < {len(ds.test_labels)} (test split size)")
            wrong = [i for i in selection.indices if ds.test_labels[i] != src]
            if wrong:
                raise InvalidExperiment(f"test indices {wrong[:5]} are not of the goal's source class "
                                        f"({'spam' if src else 'ham'})")
            return list(selection.indices)
        return [i for i, y in enumerate(ds.test_labels) if y == src][:selection.max_samples]

    @staticmethod
    def _test_predictions(system, ds) -> list[TestPrediction]:
        p = system.p_spam(list(ds.test_texts))
        return [TestPrediction(test_index=i, y_true=y, pred=int(pi >= 0.5), p_spam=float(pi))
                for i, (y, pi) in enumerate(zip(ds.test_labels, p))]

    @staticmethod
    def _capabilities(cfg: TextAttackConfig, adaptive: bool, defense: dict | None) -> dict[str, Any]:
        return {
            "knowledge": "black-box, score-based: class probabilities of the deployed system only",
            "access": f"≤ {cfg.query_budget} model queries per message (counted and enforced)",
            "knows_defense": adaptive,
            "queries_defended_pipeline": adaptive,
            "can_modify": f"characters of non-protected words of a test-time message: {', '.join(cfg.operations)} "
                          f"using {', '.join(cfg.charsets)}",
            "constraints": f"Levenshtein ≤ {cfg.max_edit_distance}, ≤ {cfg.max_char_ratio:.0%} of characters, "
                           f"≤ {cfg.max_modified_word_ratio:.0%} of modifiable words, ≤ {cfg.max_edits_per_word} "
                           f"edit(s) per word",
            "protected": ", ".join(p for p, on in (("digits/phone numbers/URLs/e-mails", cfg.protect_numbers),
                                                   ("function words", cfg.protect_stopwords)) if on) or "nothing",
            "goal": f"{cfg.goal.replace('_', ' → ')} with target-class probability ≥ {cfg.min_target_confidence}",
            "cannot": "modify training data, the model, the normaliser or the decision threshold",
        }

    def _baseline(self, req: TextBaselineRequest, deadline: float) -> dict[str, Any]:
        ds, base, common = self._context(req.model_id, req.seed, deadline)
        system = build_text_system(req.model_id, req.seed, req.defense, deadline)
        preds = self._test_predictions(system, ds)
        _check_deadline(deadline)
        return {**common, "evaluation": "clean", "intentionally_vulnerable": req.defense is None,
                "attacker_capabilities": None, "target_system": system.describe(), "attack_config": None,
                "defense_config": system.defense, "metrics": derive_text_metrics([], preds),
                "samples": [], "test_predictions": preds}

    def _attack(self, req: TextAttackRequest, deadline: float, parent: TextEvasionRecord | None) -> dict[str, Any]:
        from ..experiments import InvalidExperiment
        ds, base, common = self._context(req.model_id, req.seed, deadline)
        cfg = req.attack
        idx = self._select(ds, cfg, req.samples)
        lineage = None
        if parent is not None:
            pcfg = parent.attack_config or {}
            diff = [k for k in PRESERVATION_FIELDS if pcfg.get(k) != getattr(cfg, k)]
            if [s.test_index for s in parent.samples] != idx:
                diff.append("selected test messages")
            if diff:
                raise InvalidExperiment(f"adaptive attack is not comparable with parent {parent.experiment_id}: "
                                        "differs in " + ", ".join(diff))
            lineage = self._lineage(parent, "adaptive_attack_of", req.model_id, req.seed, ds, base,
                                    ["same goal and text-preservation constraints", "same selected messages"])
        system = build_text_system(req.model_id, req.seed, req.target_defense, deadline)
        samples = []
        for i in idx:
            _check_deadline(deadline)
            text, y = ds.test_texts[i], ds.test_labels[i]
            r = deepwordbug_attack(text, y, system.p_spam, cfg, np.random.default_rng([req.seed, 1001, i]), deadline)
            # pred_original / p_spam_original are filled from one clean batch below
            s = TextSampleResult(test_index=i, y_true=y, text_original=text, pred_original=-1, p_spam_original=0.0,
                                 outcome=r.outcome, queries=r.queries)
            if r.outcome != "skipped_misclassified":
                checks, stats = validate_constraints(text, r.text, cfg, r.edits)
                if r.outcome == "evaded" and not all(checks.values()):
                    raise TextConstraintViolation(f"test {i}: evasion violates {[k for k, v in checks.items() if not v]}")
                s.text_adversarial, s.p_spam_adversarial = r.text, r.p_spam
                s.pred_adversarial = int(r.p_spam >= 0.5)
                s.edits, s.constraints = r.edits, checks
                s.edit_distance, s.char_ratio = stats["edit_distance"], stats["char_ratio"]
                s.words_modifiable, s.words_modified = stats["words_modifiable"], stats["words_modified"]
            samples.append(s)
        # clean probabilities of the selected messages (one batch; not counted against the attack budget)
        p_clean = system.p_spam([ds.test_texts[i] for i in idx])
        for s, p in zip(samples, p_clean):
            s.p_spam_original = float(p)
            s.pred_original = int(p >= 0.5)
        preds = self._test_predictions(system, ds)
        _check_deadline(deadline)
        adaptive = req.target_defense is not None
        return {**common, "evaluation": "adaptive_attack" if adaptive else "attack",
                "intentionally_vulnerable": not adaptive,
                "attacker_capabilities": self._capabilities(cfg, adaptive, system.defense),
                "target_system": system.describe(), "attack_config": cfg.model_dump(),
                "defense_config": system.defense, "parent_experiment_id": req.parent_experiment_id,
                "lineage": lineage, "metrics": derive_text_metrics(samples, preds), "samples": samples,
                "test_predictions": preds}

    def _defend(self, req: TextDefendRequest, parent: TextEvasionRecord, deadline: float) -> dict[str, Any]:
        preq = TextAttackRequest.model_validate(parent.request)
        ds, base, common = self._context(preq.model_id, preq.seed, deadline)
        lineage = self._lineage(parent, "replay_against_defense_of", preq.model_id, preq.seed, ds, base,
                                ["parent messages match the regenerated test split"])
        from ..experiments import InvalidExperiment
        if any(ds.test_texts[s.test_index] != s.text_original for s in parent.samples):
            raise InvalidExperiment("parent messages do not match the regenerated test split")
        cfg = preq.attack
        system = build_text_system(preq.model_id, preq.seed, req.defense, deadline)
        clean = system.p_spam([s.text_original for s in parent.samples])
        replayable = [k for k, s in enumerate(parent.samples) if s.text_adversarial is not None]
        p_adv = dict(zip(replayable, system.p_spam([parent.samples[k].text_adversarial for k in replayable])))
        samples = []
        for k, ps in enumerate(parent.samples):
            pc = float(clean[k])
            correct = int(pc >= 0.5) == ps.y_true
            s = TextSampleResult(test_index=ps.test_index, y_true=ps.y_true, text_original=ps.text_original,
                                 pred_original=int(pc >= 0.5), p_spam_original=pc, outcome="skipped_misclassified")
            if correct and k in p_adv:
                pa = float(p_adv[k])
                checks, stats = validate_constraints(ps.text_original, ps.text_adversarial, cfg, ps.edits or [])
                reached = (pa if cfg.target_class == 1 else 1 - pa) >= cfg.min_target_confidence
                s.text_adversarial, s.p_spam_adversarial, s.pred_adversarial = ps.text_adversarial, pa, int(pa >= 0.5)
                s.edits, s.constraints, s.queries = ps.edits, checks, 0
                s.edit_distance, s.char_ratio = stats["edit_distance"], stats["char_ratio"]
                s.words_modifiable, s.words_modified = stats["words_modifiable"], stats["words_modified"]
                s.outcome = "evaded" if reached and all(checks.values()) else "failed"
            elif correct:
                s.outcome = "not_perturbed"
            samples.append(s)
        preds = self._test_predictions(system, ds)
        _check_deadline(deadline)
        metrics = derive_text_metrics(samples, preds)
        pm = parent.metrics
        metrics["comparison"] = {k: {"undefended": pm.get(k), "defended": metrics.get(k)}
                                 for k in ("clean_accuracy", "spam_recall", "ham_false_positive_rate",
                                           "attack_success_rate", "accuracy_under_attack")}
        caps = dict(parent.attacker_capabilities or {})
        caps.update(knows_defense=False, queries_defended_pipeline=False,
                    note="adversarial messages replayed from the parent experiment; no new queries")
        return {**common, "evaluation": "replay", "intentionally_vulnerable": False, "attacker_capabilities": caps,
                "target_system": system.describe(), "attack_config": cfg.model_dump(),
                "defense_config": system.defense, "parent_experiment_id": parent.experiment_id,
                "lineage": lineage, "metrics": metrics, "samples": samples, "test_predictions": preds}
