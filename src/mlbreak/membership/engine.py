"""ML04 experiment engine: victim utility → membership inference → recalibrated attack against a defense.

Evaluations:

* ``clean`` (mode ``baseline``) — victim training vs held-out utility (generalisation gap) and the no-signal
  baseline on the evaluation candidates. No queries.
* ``membership_inference`` (mode ``attack``) — threshold and shadow-model attacks against the victim's released
  outputs, calibrated on the attacker's own reference models.
* ``adaptive_membership_inference`` (mode ``defend``) — the defended victim (retrained on the SAME training records
  with a regularised / reduced recipe, or the same model behind an output policy) attacked by a RECALIBRATED attack:
  reference and shadow models are retrained with the defended recipe and see outputs through the same policy. Same
  candidates and attack configuration as the parent; paired per candidate.

Ground-truth isolation: this engine is the only place that knows which candidate is a member. It builds an
:class:`~.attacks.AttackerKnowledge` (candidate features, optional labels, auxiliary populations, the training recipe)
and a counting oracle, runs the attacks, and attaches :class:`~.schemas.MembershipTruth` only afterwards. Candidate
order is shuffled so that position does not reveal the group.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ..attacks import ExperimentTimeout, QueryBudgetExceeded, _check_deadline
from ..hashing import sha256_json
from ..integrity import record_sha256
from ..owasp import get_category
from .attacks import AttackerKnowledge, query_candidates, reference_data, threshold_attack
from .data import N_TRAIN, load_membership_dataset, row_hashes, select_candidates
from .defenses import MembershipService, counting_oracle, describe_policy
from .integrity import attacker_side_sha256, results_fingerprint
from .metrics import ATTACKS, derive_membership_metrics
from .models import VICTIMS, Recipe, apply_defense, victim_model, victim_recipe
from .schemas import (SCENARIO_ID, AttackCalibration, AttackDecision, AttackerObservation, CandidateResult,
                      MembershipAttackConfig, MembershipAttackRequest, MembershipBaselineRequest,
                      MembershipDefendRequest, MembershipRecord, MembershipTruth, UtilityRow, policy_of)
from .shadow import shadow_attack

if TYPE_CHECKING:  # pragma: no cover
    from ..experiments import ExperimentEngine

FULL = {"kind": "full_probability"}


class MembershipLab:
    def __init__(self, engine: "ExperimentEngine"):
        self.engine = engine
        self.store = engine.store

    # ------------------------------------------------------------------ API
    def run_baseline(self, req: MembershipBaselineRequest, replay_of: str | None = None) -> MembershipRecord:
        return self._execute("baseline", req, self._baseline, replay_of)

    def run_attack(self, req: MembershipAttackRequest, replay_of: str | None = None) -> MembershipRecord:
        return self._execute("attack", req, self._attack, replay_of)

    def run_defend(self, req: MembershipDefendRequest, replay_of: str | None = None) -> MembershipRecord:
        parent = self._load_parent(req.parent_experiment_id)
        return self._execute("defend", req, lambda r, d: self._defend(r, parent, d), replay_of)

    def replay(self, original: MembershipRecord) -> dict[str, Any]:
        from ..experiments import InvalidExperiment, environment_differences, environment_note
        if original.status != "completed":
            raise InvalidExperiment("only completed experiments can be replayed")
        integrity = self.engine.require_verified(original)
        oid = original.experiment_id
        if original.mode == "baseline":
            new = self.run_baseline(MembershipBaselineRequest.model_validate(original.request), replay_of=oid)
        elif original.mode == "attack":
            new = self.run_attack(MembershipAttackRequest.model_validate(original.request), replay_of=oid)
        else:
            new = self.run_defend(MembershipDefendRequest.model_validate(original.request), replay_of=oid)
        o, n = original.reproducibility, new.reproducibility
        pairs = [("status", original.status, new.status),
                 ("config_sha256", o.get("config_sha256"), n.get("config_sha256")),
                 ("dataset_sha256", sha256_json({k: v for k, v in original.dataset.items() if k.endswith("_sha256")}),
                  sha256_json({k: v for k, v in new.dataset.items() if k.endswith("_sha256")})),
                 ("victim_artifact_sha256", original.model["artifact_sha256"], new.model.get("artifact_sha256")),
                 ("calibration_sha256", sha256_json(original.calibration), sha256_json(new.calibration)),
                 ("shadow_sha256", sha256_json(original.shadow), sha256_json(new.shadow)),
                 ("attacker_outputs_sha256", attacker_side_sha256(original.candidates),
                  attacker_side_sha256(new.candidates)),
                 ("metrics_sha256", o.get("metrics_sha256"), n.get("metrics_sha256")),
                 ("results_sha256", o.get("results_sha256"), n.get("results_sha256"))]
        checks = [{"check": k, "original": a, "replay": b, "match": a == b} for k, a, b in pairs]
        env = environment_differences(o, n)
        identical = all(c["match"] for c in checks)
        return {"original_experiment_id": oid, "replay_experiment_id": new.experiment_id, "identical": identical,
                "checks": checks, "original_integrity": integrity, "environment_differences": env,
                "notes": environment_note(env, identical)}

    # ------------------------------------------------- multi-seed study (evaluator-facing)
    def study(self, req):
        """Run, store and return a multi-seed study (:mod:`.study`); each seed runs in its own gated call."""
        from .study import StudyStore, run_study
        rep = run_study(self, req)
        StudyStore(self.store.root).save(rep)
        return rep

    def load_study(self, report_id: str):
        from .study import StudyStore
        return StudyStore(self.store.root).load(report_id)

    def verify_study(self, report_id: str) -> dict[str, Any]:
        from .study import verify_study
        return verify_study(self.load_study(report_id))

    def replay_study(self, report_id: str) -> dict[str, Any]:
        """Re-run a stored study from its request (refused if it fails verification) and compare fingerprints."""
        from ..experiments import EvidenceIntegrityError
        from .study import StudyRequest
        original = self.load_study(report_id)
        check = self.verify_study(report_id)
        if not check["ok"]:
            raise EvidenceIntegrityError(report_id, check)
        new = self.study(StudyRequest.model_validate(original.request))
        o, n = original.reproducibility, new.reproducibility
        checks = [{"check": k, "original": o.get(k), "replay": n.get(k), "match": o.get(k) == n.get(k)}
                  for k in ("config_sha256", "results_sha256")]
        return {"original_report_id": report_id, "replay_report_id": new.report_id,
                "identical": all(c["match"] for c in checks), "checks": checks}

    def attacker_independence(self, rec: MembershipRecord) -> dict[str, Any]:
        """Executable check that the attack learned nothing from evaluation membership flags or victim data.

        1. Re-run the attack of a stored record with the evaluation ground truth replaced by a random permutation of
           the groups: every attacker-side value (released outputs, scores, thresholds, calibration, shadow models,
           decisions) must be bit-identical.
        2. The attacker's populations (calibration, shadow) must share no record with the victim's training set or
           with the evaluation candidates.
        """
        from ..experiments import InvalidExperiment
        if rec.status != "completed" or rec.mode == "baseline":
            raise InvalidExperiment("attacker independence needs a completed attack or defend record")
        self.engine.require_verified(rec)
        req = MembershipAttackRequest.model_validate(
            rec.request if rec.mode == "attack" else self.store.load(rec.parent_experiment_id).request)

        def work(deadline: float) -> dict[str, Any]:
            ds = load_membership_dataset(req.seed)
            rec_recipe, _ = apply_defense(victim_recipe(req.model_id), rec.defense_config)
            victim = victim_model(req.model_id, req.seed, rec_recipe)
            cands, cal, shadow, _ = self._run_attack(ds, req, rec_recipe, rec.output_policy, victim, deadline,
                                                     permute_truth=True)
            truth_changed = [c.truth.group for c in cands] != [c.truth.group for c in rec.candidates]
            same = (attacker_side_sha256(cands) == attacker_side_sha256(rec.candidates)
                    and sha256_json(cal) == sha256_json(rec.calibration) and sha256_json(shadow) == sha256_json(rec.shadow))
            X_c, _, _, _ = self._candidate_arrays(ds, req)
            victim_rows = set(row_hashes(ds[VICTIMS[req.model_id]["population"]].X)) | set(row_hashes(X_c))
            overlaps = {p: len(set(row_hashes(ds[p].X)) & victim_rows) for p in ("calibration", "shadow")}
            checks = [{"check": "ground truth permuted for the re-run", "ok": truth_changed},
                      {"check": "attacker outputs identical with permuted ground truth", "ok": same},
                      *({"check": f"{p} population ∩ (victim training ∪ candidates) = ∅", "ok": v == 0,
                         "overlap": v} for p, v in overlaps.items())]
            return {"experiment_id": rec.experiment_id, "ok": all(c["ok"] for c in checks), "checks": checks}
        return self.engine.gated(work)

    # ------------------------------------------------------------ internals
    def _execute(self, mode: str, req, fn: Callable[[Any, float], dict[str, Any]],
                 replay_of: str | None) -> MembershipRecord:
        from ..experiments import _new_id, _now, reproducibility_info

        def work(deadline: float) -> MembershipRecord:
            started, exp_id = time.monotonic(), _new_id()
            try:
                body = fn(req, deadline)
                status, error = "completed", None
            except (ExperimentTimeout, QueryBudgetExceeded) as exc:
                body, status, error = self._failure_body(req, mode), "failed", f"{type(exc).__name__}: {exc}"
            body.setdefault("metrics", {})
            body.setdefault("candidates", [])
            body.setdefault("utility", [])
            done = status == "completed"
            repro = reproducibility_info()
            repro["config_sha256"] = sha256_json(req.model_dump(mode="json"))
            repro["metrics_sha256"] = sha256_json(body["metrics"]) if done else None
            repro["results_sha256"] = (results_fingerprint(body["candidates"], body["utility"], body["metrics"],
                                                           body.get("calibration"), body.get("shadow"))
                                       if done else None)
            repro["replay_command"] = f"mlbreak replay {exp_id}"
            repro["verify_command"] = f"mlbreak check {exp_id}"
            rec = MembershipRecord(experiment_id=exp_id, created_at=_now(), mode=mode, status=status, error=error,
                                   replay_of=replay_of, request=req.model_dump(mode="json"), reproducibility=repro,
                                   duration_ms=round((time.monotonic() - started) * 1000, 2), **body)
            rec.reproducibility["record_sha256"] = record_sha256(rec)
            self.store.save(rec)
            return rec
        return self.engine.gated(work)

    def _common(self, model_id: str, seed: int, victim, policy: dict[str, Any], ds) -> dict[str, Any]:
        from ..experiments import SCENARIOS
        scen = SCENARIOS[SCENARIO_ID]
        cat = get_category(scen.owasp_id)
        v = VICTIMS[model_id]
        return {"scenario_id": scen.scenario_id,
                "owasp": {"id": cat.id, "name": cat.name, "taxonomy": cat.taxonomy,
                          "taxonomy_version": cat.taxonomy_version, "reference_url": cat.reference_url},
                "model": {"model_id": model_id, "description": v["description"], "training_population": v["population"],
                          "training_population_sha256": ds[v["population"]].sha256(), **victim.describe()},
                "target_system": MembershipService(victim, policy).describe(), "dataset": ds.describe(), "seed": seed,
                "output_policy": policy}

    @staticmethod
    def _candidate_arrays(ds, req) -> tuple[np.ndarray, np.ndarray, list[MembershipTruth], list[str]]:
        sel = req.candidates
        mem, non = select_candidates(req.seed, sel.n_members, sel.n_non_members)
        decoy = VICTIMS[req.model_id]["population"] != "victim_train"
        rows = [("member", "victim_train", i) for i in mem] + [("non_member", "evaluation_non_members", i) for i in non]
        order = np.random.default_rng([req.seed, 4106]).permutation(len(rows))  # position must not reveal the group
        rows = [rows[int(k)] for k in order]
        X = np.array([ds[p].X[i] for _, p, i in rows], dtype=np.float64).reshape(len(rows), -1)
        y = np.array([ds[p].y[i] for _, p, i in rows], dtype=np.int64)
        truths = [MembershipTruth(group=g, in_training=(g == "member" and not decoy), population=p, population_index=i)
                  for g, p, i in rows]
        return X, y, truths, [f"C-{k:04d}" for k in range(len(rows))]

    def _utility_rows(self, ds, model_id: str, victim, policy: dict[str, Any]) -> list[UtilityRow]:
        service = MembershipService(victim, policy)
        out = []
        for split, pop in (("train", VICTIMS[model_id]["population"]), ("test", "victim_test")):
            p = ds[pop]
            full = service.full_scores(p.X)
            rel = service.respond(p.X)
            out += [UtilityRow(split=split, index=i, y_true=int(p.y[i]), p_full=float(full[i]), released=float(rel[i]))
                    for i in range(len(p.y))]
        return out

    def _run_attack(self, ds, req: MembershipAttackRequest, rec_recipe: Recipe, policy: dict[str, Any], victim,
                    deadline: float, permute_truth: bool = False):
        cfg: MembershipAttackConfig = req.attack
        X, y, truths, ids = self._candidate_arrays(ds, req)
        if permute_truth:  # evaluator-side only (attacker_independence): the attack must not notice
            perm = np.random.default_rng([req.seed, 4999]).permutation(len(truths))
            groups = [truths[int(k)].group for k in perm]
            if groups == [t.group for t in truths]:
                groups = groups[1:] + groups[:1]
            truths = [t.model_copy(update={"group": g}) for t, g in zip(truths, groups)]
        know = AttackerKnowledge(candidate_ids=tuple(ids), X=X, y=y if cfg.label_known else None, recipe=rec_recipe,
                                 policy=dict(policy), n_train=N_TRAIN, shadow=ds["shadow"],
                                 calibration=ds["calibration"], seed=req.seed)
        query, used = counting_oracle(MembershipService(victim, policy), cfg.query_budget)
        released = query_candidates(know, query)
        _check_deadline(deadline)
        cfgd = cfg.model_dump()
        ref = reference_data(know, cfg.n_reference_models, deadline)
        th_cal, th_dec = threshold_attack(know, released, ref, cfgd)
        sh_prov, sh_cal, sh_dec = shadow_attack(know, released, ref, cfgd, deadline)
        cands = [CandidateResult(candidate_id=ids[i], truth=truths[i],
                                 observation=AttackerObservation(features=X[i].tolist(),
                                                                 label=int(y[i]) if cfg.label_known else None,
                                                                 released_output=float(released[i]), query_index=i + 1),
                                 threshold=AttackDecision(**th_dec[i]), shadow=AttackDecision(**sh_dec[i]))
                 for i in range(len(ids))]
        calibration = {
            "population": "calibration (attacker auxiliary data, disjoint from every victim and evaluation population)",
            "population_sha256": ds["calibration"].sha256(), "n_reference_models": len(ref.models),
            "reference_models": ref.models,
            "provenance": "thresholds of BOTH attacks are chosen on these reference models' records, whose membership "
                          "the attacker controls; no evaluation candidate or victim membership flag is used",
            "attacks": {"threshold": AttackCalibration.model_validate(th_cal).model_dump(),
                        "shadow": AttackCalibration.model_validate(sh_cal).model_dump()}}
        return cands, calibration, sh_prov, used()

    @staticmethod
    def _capabilities(cfg: MembershipAttackConfig, policy: dict[str, Any], rec_recipe: Recipe,
                      adaptive: bool) -> dict[str, Any]:
        return {
            "knows": "each candidate's 8 features" + (" and true label" if cfg.label_known else "")
                     + "; the victim's training recipe (model family + hyperparameters, incl. any training-time "
                       "defense) and training-set size; two auxiliary populations from the same distribution "
                       "(shadow and calibration), disjoint from the victim's data",
            "does_not_know": "whether any candidate is a training member; the victim's training indices or "
                             "records; its fitted parameters; any evaluator evidence",
            "access": f"black-box prediction service, one query per candidate (≤ {cfg.query_budget} counted queries); "
                      "shadow and reference models are trained locally without querying the victim",
            "output_seen": describe_policy(policy)["releases"],
            "recipe_assumed": rec_recipe.describe(),
            "adapts_to_defense": adaptive,
            "methods": f"threshold on {cfg.threshold_signal} ({cfg.threshold_mode}); shadow models "
                       f"(n={cfg.n_shadow_models}) + gradient-boosting meta-classifier; operating thresholds from "
                       f"{cfg.n_reference_models} reference model(s)",
            "goal": "decide for each candidate whether it was in the victim's training set",
        }

    def _failure_body(self, req, mode: str) -> dict[str, Any]:
        r = req
        defense = None
        if isinstance(req, MembershipDefendRequest):
            r = MembershipAttackRequest.model_validate(self.store.load(req.parent_experiment_id).request)
            defense = req.defense.model_dump()
        ds = load_membership_dataset(r.seed)
        rec_recipe, _ = apply_defense(victim_recipe(r.model_id), defense)
        policy = policy_of(defense)
        victim = victim_model(r.model_id, r.seed, rec_recipe)
        return {**self._common(r.model_id, r.seed, victim, policy, ds),
                "evaluation": {"baseline": "clean", "attack": "membership_inference"}.get(
                    mode, "adaptive_membership_inference"),
                "intentionally_vulnerable": mode != "defend", "attacker_capabilities": None,
                "attack_config": getattr(r, "attack", None) and r.attack.model_dump(), "defense_config": defense,
                "candidate_selection": r.candidates.model_dump(),
                "parent_experiment_id": getattr(req, "parent_experiment_id", None)}

    def _load_parent(self, parent_id: str) -> MembershipRecord:
        from ..experiments import InvalidExperiment
        parent = self.store.load(parent_id)
        if not isinstance(parent, MembershipRecord):
            raise InvalidExperiment(f"parent {parent_id} is not an ML04 membership experiment")
        if parent.status != "completed":
            raise InvalidExperiment(f"parent {parent_id} did not complete (status {parent.status})")
        if parent.mode != "attack" or parent.evaluation != "membership_inference":
            raise InvalidExperiment(f"parent {parent_id} must be a membership attack against the undefended victim "
                                    f"(got mode={parent.mode}, evaluation={parent.evaluation})")
        self.engine.require_verified(parent)
        return parent

    def _metrics(self, evaluation: str, req, cands, utility, calibration, ds) -> dict[str, Any]:
        budget = req.attack.query_budget if hasattr(req, "attack") else None
        return derive_membership_metrics(evaluation, cands, utility, req.candidates.operational_prevalences,
                                         calibration, budget, ds.describe()["disjointness"]["disjoint"])

    def _baseline(self, req: MembershipBaselineRequest, deadline: float) -> dict[str, Any]:
        ds = load_membership_dataset(req.seed)
        victim = victim_model(req.model_id, req.seed, victim_recipe(req.model_id))
        _check_deadline(deadline)
        _, _, truths, ids = self._candidate_arrays(ds, req)
        cands = [CandidateResult(candidate_id=i, truth=t) for i, t in zip(ids, truths)]
        utility = self._utility_rows(ds, req.model_id, victim, FULL)
        return {**self._common(req.model_id, req.seed, victim, FULL, ds), "evaluation": "clean",
                "intentionally_vulnerable": True, "attacker_capabilities": None, "attack_config": None,
                "defense_config": None, "candidate_selection": req.candidates.model_dump(),
                "metrics": self._metrics("clean", req, cands, utility, None, ds), "candidates": cands,
                "utility": utility}

    def _attack(self, req: MembershipAttackRequest, deadline: float) -> dict[str, Any]:
        ds = load_membership_dataset(req.seed)
        rec_recipe = victim_recipe(req.model_id)
        victim = victim_model(req.model_id, req.seed, rec_recipe)
        _check_deadline(deadline)
        cands, cal, shadow, _ = self._run_attack(ds, req, rec_recipe, FULL, victim, deadline)
        utility = self._utility_rows(ds, req.model_id, victim, FULL)
        return {**self._common(req.model_id, req.seed, victim, FULL, ds), "evaluation": "membership_inference",
                "intentionally_vulnerable": True,
                "attacker_capabilities": self._capabilities(req.attack, FULL, rec_recipe, False),
                "attack_config": req.attack.model_dump(), "defense_config": None,
                "candidate_selection": req.candidates.model_dump(), "calibration": cal, "shadow": shadow,
                "metrics": self._metrics("membership_inference", req, cands, utility, cal, ds), "candidates": cands,
                "utility": utility}

    def _defend(self, req: MembershipDefendRequest, parent: MembershipRecord, deadline: float) -> dict[str, Any]:
        from ..experiments import InvalidExperiment
        preq = MembershipAttackRequest.model_validate(parent.request)
        defense = req.defense.model_dump()
        try:
            rec_recipe, applied = apply_defense(victim_recipe(preq.model_id), defense)
        except ValueError as exc:
            raise InvalidExperiment(str(exc)) from exc
        ds = load_membership_dataset(preq.seed)
        base = victim_model(preq.model_id, preq.seed, victim_recipe(preq.model_id))
        fp = {k: v for k, v in ds.fingerprint().items()}
        problems = [name for name, ok in (
            ("dataset population hashes", all(parent.dataset.get(k) == v for k, v in fp.items())),
            ("undefended victim artifact hash", parent.model.get("artifact_sha256") == base.artifact_sha256()))
            if not ok]
        if problems:
            raise InvalidExperiment(f"parent {parent.experiment_id} is incompatible: differs in " + ", ".join(problems))
        victim = victim_model(preq.model_id, preq.seed, rec_recipe)
        policy = policy_of(defense)
        _check_deadline(deadline)
        cands, cal, shadow, _ = self._run_attack(ds, preq, rec_recipe, policy, victim, deadline)
        if [(c.candidate_id, c.truth) for c in cands] != [(c.candidate_id, c.truth) for c in parent.candidates]:
            raise InvalidExperiment("parent candidates do not match the regenerated evaluation population")
        for c, p in zip(cands, parent.candidates):
            member = p.truth.group == "member"
            c.undefended_correct = {a: getattr(p, a).member_balanced == member for a in ATTACKS}
        utility = self._utility_rows(ds, preq.model_id, victim, policy)
        lineage = {
            "relation": "recalibrated_membership_attack_on", "parent_experiment_id": parent.experiment_id,
            "parent_evaluation": parent.evaluation, "parent_schema_version": parent.schema_version,
            "parent_config_sha256": parent.reproducibility.get("config_sha256"),
            "parent_results_sha256": parent.reproducibility.get("results_sha256"),
            "parent_record_sha256": parent.reproducibility.get("record_sha256"),
            "shared": {"scenario_id": SCENARIO_ID, "model_id": preq.model_id, "seed": preq.seed, **fp,
                       "undefended_victim_artifact_sha256": base.artifact_sha256(),
                       "candidate_ids_sha256": sha256_json([c.candidate_id for c in cands]),
                       "attack_config": preq.attack.model_dump()},
            "defense_applied": applied,
            "checks_passed": ["parent completed", "parent is a membership attack on the undefended victim",
                              "parent evidence integrity", "dataset population hashes",
                              "undefended victim artifact hash", "same candidates and ground truth",
                              "same attack configuration; reference and shadow models retrained with the defended "
                              "recipe and the defended output policy"],
        }
        return {**self._common(preq.model_id, preq.seed, victim, policy, ds),
                "evaluation": "adaptive_membership_inference", "intentionally_vulnerable": False,
                "attacker_capabilities": self._capabilities(preq.attack, policy, rec_recipe, True),
                "attack_config": preq.attack.model_dump(), "defense_config": defense,
                "candidate_selection": preq.candidates.model_dump(), "calibration": cal, "shadow": shadow,
                "parent_experiment_id": parent.experiment_id, "lineage": lineage,
                "metrics": self._metrics("adaptive_membership_inference", preq, cands, utility, cal, ds),
                "candidates": cands, "utility": utility}
