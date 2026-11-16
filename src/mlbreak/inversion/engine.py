"""ML03 experiment engine: victim utility + auxiliary baseline → inversion → adaptive inversion per output policy.

Evaluations:

* ``clean`` (mode ``baseline``) — the victim on the evaluation split under full probability (utility, and how
  relevant the private attribute is), and the AUXILIARY-ONLY attacker on the selected targets (no queries).
* ``inversion`` (mode ``attack``) — candidate-matching reconstruction against the undefended full-probability
  service, compared target-by-target with the auxiliary-only attacker.
* ``adaptive_inversion`` (mode ``defend``) — a NEW reconstruction against a restrictive output policy, with the
  parent attack's model, seed, targets and attack configuration; the observed target output and every candidate
  query go through that policy. Paired with the parent (same targets).

Ground-truth isolation: this engine is the only place that holds the private attribute. For each target it
computes the observed output of the REAL record through the service, builds an :class:`~.attacks.AttackerView`
(public features, observed output, the attacker's auxiliary prior) and hands the attack only that view and a
counting oracle. Correctness is computed after the attack returns.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ..attacks import ExperimentTimeout, QueryBudgetExceeded, _check_deadline
from ..hashing import sha256_json
from ..integrity import record_sha256
from ..owasp import get_category
from .attacks import AttackerView, auxiliary_guess, candidate_matching
from .data import load_inversion_dataset
from .defenses import PredictionService, apply_policy, counting_oracle, describe_policy
from .integrity import results_fingerprint
from .metrics import derive_inversion_metrics
from .models import auxiliary_model, public_only_reference, victim_model
from .schemas import (SCENARIO_ID, GroundTruth, InversionAttackConfig, InversionAttackRequest,
                      InversionBaselineRequest, InversionDefendRequest, InversionQueryRequest, InversionRecord,
                      InversionTargetsRequest, ServiceRow, TargetResult, TargetSelection)

if TYPE_CHECKING:  # pragma: no cover
    from ..experiments import ExperimentEngine

FULL = {"kind": "full_probability"}


def select_targets(sel: TargetSelection, seed: int, n_eval: int) -> list[int]:
    from ..experiments import InvalidExperiment
    if sel.indices is not None:
        if len(set(sel.indices)) != len(sel.indices):
            raise InvalidExperiment("target indices must be unique")
        bad = [i for i in sel.indices if i >= n_eval]
        if bad:
            raise InvalidExperiment(f"target indices must be < {n_eval} (evaluation split size)")
        return list(sel.indices)
    order = list(range(n_eval)) if sel.order == "first" else \
        [int(i) for i in np.random.default_rng([seed, 3103]).permutation(n_eval)]
    return order[:sel.max_targets]


class InversionLab:
    def __init__(self, engine: "ExperimentEngine"):
        self.engine = engine
        self.store = engine.store

    # ------------------------------------------------------------------ API
    def run_baseline(self, req: InversionBaselineRequest, replay_of: str | None = None) -> InversionRecord:
        return self._execute("baseline", req, self._baseline, replay_of)

    def run_attack(self, req: InversionAttackRequest, replay_of: str | None = None) -> InversionRecord:
        return self._execute("attack", req, self._attack, replay_of)

    def run_defend(self, req: InversionDefendRequest, replay_of: str | None = None) -> InversionRecord:
        parent = self._load_parent(req.parent_experiment_id)
        return self._execute("defend", req, lambda r, d: self._defend(r, parent, d), replay_of)

    def targets_view(self, req: InversionTargetsRequest) -> dict[str, Any]:
        """ATTACKER-FACING view of the selected targets: public features and the service's released output for
        the real record. Contains no private attribute, label, score before the policy or model parameter."""
        def work(deadline: float) -> dict[str, Any]:
            ds = load_inversion_dataset(req.seed)
            service = PredictionService(victim_model(req.model_id, req.seed), req.output_policy.model_dump())
            idx = select_targets(req.targets, req.seed, len(ds.s_eval))
            observed = service.respond(ds.X_eval[idx], ds.s_eval[idx])
            _check_deadline(deadline)
            return {"model_id": req.model_id, "seed": req.seed,
                    "output_policy": describe_policy(req.output_policy.model_dump()),
                    "public_features": list(ds.describe()["public_features"]),
                    "private_attribute": {"name": ds.describe()["private_attribute"], "domain": [0, 1],
                                          "value": "hidden"},
                    "targets": [{"target_index": i, "public_features": ds.X_eval[i].tolist(),
                                 "observed_output": float(o)} for i, o in zip(idx, observed)]}
        return self.engine.gated(work)

    def query(self, req: InversionQueryRequest) -> dict[str, Any]:
        """One black-box query: the service evaluates exactly the record supplied (nothing is looked up)."""
        def work(deadline: float) -> dict[str, Any]:
            service = PredictionService(victim_model(req.model_id, req.seed), req.output_policy.model_dump())
            out = float(service.respond(np.array([req.public_features]), np.array([req.private_attribute]))[0])
            return {"released": out, "output_policy": describe_policy(req.output_policy.model_dump())}
        return self.engine.gated(work)

    def replay(self, original: InversionRecord) -> dict[str, Any]:
        from ..experiments import InvalidExperiment, environment_differences, environment_note
        if original.status != "completed":
            raise InvalidExperiment("only completed experiments can be replayed")
        integrity = self.engine.require_verified(original)
        oid = original.experiment_id
        if original.mode == "baseline":
            new = self.run_baseline(InversionBaselineRequest.model_validate(original.request), replay_of=oid)
        elif original.mode == "attack":
            new = self.run_attack(InversionAttackRequest.model_validate(original.request), replay_of=oid)
        else:
            new = self.run_defend(InversionDefendRequest.model_validate(original.request), replay_of=oid)
        o, n = original.reproducibility, new.reproducibility
        pairs = [("status", original.status, new.status),
                 ("config_sha256", o.get("config_sha256"), n.get("config_sha256")),
                 ("train_split_sha256", original.dataset["train_sha256"], new.dataset.get("train_sha256")),
                 ("auxiliary_split_sha256", original.dataset["auxiliary_sha256"], new.dataset.get("auxiliary_sha256")),
                 ("test_split_sha256", original.dataset["test_sha256"], new.dataset.get("test_sha256")),
                 ("model_artifact_sha256", original.model["artifact_sha256"], new.model.get("artifact_sha256")),
                 ("auxiliary_model_artifact_sha256", (original.auxiliary_model or {}).get("artifact_sha256"),
                  (new.auxiliary_model or {}).get("artifact_sha256")),
                 ("inferred_values_sha256", sha256_json([t.trace and t.trace.inferred for t in original.targets]),
                  sha256_json([t.trace and t.trace.inferred for t in new.targets])),
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
                 replay_of: str | None) -> InversionRecord:
        from ..experiments import _new_id, _now, reproducibility_info

        def work(deadline: float) -> InversionRecord:
            started, exp_id = time.monotonic(), _new_id()
            try:
                body = fn(req, deadline)
                status, error = "completed", None
            except (ExperimentTimeout, QueryBudgetExceeded) as exc:
                body, status, error = self._failure_body(req, mode), "failed", f"{type(exc).__name__}: {exc}"
            body.setdefault("metrics", {})
            body.setdefault("targets", [])
            body.setdefault("service", [])
            done = status == "completed"
            repro = reproducibility_info()
            repro["config_sha256"] = sha256_json(req.model_dump(mode="json"))
            repro["metrics_sha256"] = sha256_json(body["metrics"]) if done else None
            repro["results_sha256"] = (results_fingerprint(body["targets"], body["service"], body["metrics"])
                                       if done else None)
            repro["replay_command"] = f"mlbreak replay {exp_id}"
            repro["verify_command"] = f"mlbreak check {exp_id}"
            rec = InversionRecord(experiment_id=exp_id, created_at=_now(), mode=mode, status=status, error=error,
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
        ds = load_inversion_dataset(seed)
        victim, aux = victim_model(model_id, seed), auxiliary_model(seed)
        _check_deadline(deadline)
        return ds, victim, aux, {
            "scenario_id": scen.scenario_id,
            "owasp": {"id": cat.id, "name": cat.name, "taxonomy": cat.taxonomy,
                      "taxonomy_version": cat.taxonomy_version, "reference_url": cat.reference_url},
            "model": victim.describe(), "auxiliary_model": aux.describe(), "dataset": ds.describe(), "seed": seed}

    def _failure_body(self, req, mode: str) -> dict[str, Any]:
        r = req
        if isinstance(req, InversionDefendRequest):
            r = InversionAttackRequest.model_validate(self.store.load(req.parent_experiment_id).request)
        _, _, _, common = self._context(r.model_id, r.seed, None)
        policy = req.output_policy.model_dump() if isinstance(req, InversionDefendRequest) else FULL
        return {**common, "evaluation": {"baseline": "clean", "attack": "inversion"}.get(mode, "adaptive_inversion"),
                "intentionally_vulnerable": mode != "defend", "attacker_capabilities": None,
                "target_system": {"model_artifact_sha256": None, "output_policy": describe_policy(policy)},
                "output_policy": policy, "attack_config": getattr(r, "attack", None) and r.attack.model_dump(),
                "defense_config": policy if mode == "defend" else None, "target_selection": r.targets.model_dump(),
                "parent_experiment_id": getattr(req, "parent_experiment_id", None)}

    def _load_parent(self, parent_id: str) -> InversionRecord:
        from ..experiments import InvalidExperiment
        parent = self.store.load(parent_id)
        if not isinstance(parent, InversionRecord):
            raise InvalidExperiment(f"parent {parent_id} is not an ML03 inversion experiment")
        if parent.status != "completed":
            raise InvalidExperiment(f"parent {parent_id} did not complete (status {parent.status})")
        if parent.mode != "attack" or parent.evaluation != "inversion":
            raise InvalidExperiment(f"parent {parent_id} must be an inversion attack against the full-probability "
                                    f"service (got mode={parent.mode}, evaluation={parent.evaluation})")
        self.engine.require_verified(parent)
        return parent

    @staticmethod
    def _capabilities(cfg: InversionAttackConfig, policy: dict[str, Any], adaptive: bool) -> dict[str, Any]:
        return {
            "knows": "the target's public features exactly; the service's released output for the target record; "
                     "the private attribute's possible values {0, 1}; an auxiliary population sample (public "
                     "features + private attribute, no labels, no victim outputs)",
            "does_not_know": "the target's private attribute, its label, the model's parameters, the private "
                             "training records or the exact training set",
            "access": f"black-box prediction service, ≤ {cfg.query_budget} crafted-record queries per target "
                      "(counted and enforced)",
            "output_seen": describe_policy(policy)["releases"],
            "adapts_to_policy": adaptive,
            "method": f"candidate matching (tolerance {cfg.tolerance:g}), ties → {cfg.tie_policy}",
            "assumptions": "deterministic service; the attacker's public features equal the values the service used",
            "goal": "reconstruct the binary private attribute of each held-out target record",
        }

    def _service_rows(self, ds, victim, policy: dict[str, Any], with_reference: bool) -> list[ServiceRow]:
        full = PredictionService(victim, FULL).full_scores(ds.X_eval, ds.s_eval)
        released = apply_policy(policy, full)
        pub = public_only_reference(ds.seed).proba(np.c_[ds.X_eval, ds.s_eval]) if with_reference else None
        return [ServiceRow(eval_index=i, y_true=int(ds.y_eval[i]), p_full=float(full[i]), released=float(released[i]),
                           decision=int(full[i] >= 0.5), consumer_decision=int(released[i] >= 0.5),
                           p_public_only=None if pub is None else float(pub[i])) for i in range(len(full))]

    @staticmethod
    def _baseline_rows(ds, aux, idx: list[int]) -> list[TargetResult]:
        p1 = aux.proba(ds.X_eval[idx]) if idx else np.empty(0)
        guess = auxiliary_guess(p1)
        return [TargetResult(target_index=i, auxiliary_p1=float(p), baseline_inferred=int(g),
                             baseline_correct=bool(g == ds.s_eval[i]),
                             ground_truth=GroundTruth(private_attribute=int(ds.s_eval[i]), target_label=int(ds.y_eval[i])))
                for i, p, g in zip(idx, p1, guess)]

    def _invert(self, ds, victim, rows: list[TargetResult], cfg: InversionAttackConfig, policy: dict[str, Any],
                deadline: float) -> None:
        service = PredictionService(victim, policy)
        for r in rows:
            _check_deadline(deadline)
            i = r.target_index
            # The ONLY use of the hidden attribute before scoring: the service answering for the real record.
            observed = float(service.respond(ds.X_eval[i:i + 1], ds.s_eval[i:i + 1])[0])
            view = AttackerView(public_features=tuple(float(v) for v in ds.X_eval[i]), observed_output=observed,
                                auxiliary_p1=r.auxiliary_p1)
            query, _ = counting_oracle(service, cfg.query_budget)
            r.trace = candidate_matching(view, query, cfg)
            r.inversion_correct = r.trace.inferred is not None and r.trace.inferred == r.ground_truth.private_attribute

    def _baseline(self, req: InversionBaselineRequest, deadline: float) -> dict[str, Any]:
        ds, victim, aux, common = self._context(req.model_id, req.seed, deadline)
        idx = select_targets(req.targets, req.seed, len(ds.s_eval))
        rows = self._baseline_rows(ds, aux, idx)
        service = self._service_rows(ds, victim, FULL, with_reference=True)
        _check_deadline(deadline)
        return {**common, "evaluation": "clean", "intentionally_vulnerable": True, "attacker_capabilities": None,
                "target_system": PredictionService(victim, FULL).describe(), "output_policy": FULL,
                "attack_config": None, "defense_config": None, "target_selection": req.targets.model_dump(),
                "metrics": derive_inversion_metrics("clean", rows, service), "targets": rows, "service": service}

    def _attack(self, req: InversionAttackRequest, deadline: float) -> dict[str, Any]:
        ds, victim, aux, common = self._context(req.model_id, req.seed, deadline)
        idx = select_targets(req.targets, req.seed, len(ds.s_eval))
        rows = self._baseline_rows(ds, aux, idx)
        self._invert(ds, victim, rows, req.attack, FULL, deadline)
        service = self._service_rows(ds, victim, FULL, with_reference=False)
        return {**common, "evaluation": "inversion", "intentionally_vulnerable": True,
                "attacker_capabilities": self._capabilities(req.attack, FULL, False),
                "target_system": PredictionService(victim, FULL).describe(), "output_policy": FULL,
                "attack_config": req.attack.model_dump(), "defense_config": None,
                "target_selection": req.targets.model_dump(),
                "metrics": derive_inversion_metrics("inversion", rows, service), "targets": rows, "service": service}

    def _defend(self, req: InversionDefendRequest, parent: InversionRecord, deadline: float) -> dict[str, Any]:
        from ..experiments import InvalidExperiment
        preq = InversionAttackRequest.model_validate(parent.request)
        ds, victim, aux, common = self._context(preq.model_id, preq.seed, deadline)
        fp = ds.fingerprint()
        problems = [name for name, ok in (
            ("dataset split hashes", all(parent.dataset.get(k) == v for k, v in fp.items())),
            ("victim artifact hash", parent.model.get("artifact_sha256") == victim.artifact_sha256()),
            ("auxiliary model artifact hash",
             (parent.auxiliary_model or {}).get("artifact_sha256") == aux.artifact_sha256())) if not ok]
        if problems:
            raise InvalidExperiment(f"parent {parent.experiment_id} is incompatible: differs in " + ", ".join(problems))
        idx = [t.target_index for t in parent.targets]
        rows = self._baseline_rows(ds, aux, idx)
        if [t.ground_truth for t in rows] != [t.ground_truth for t in parent.targets]:
            raise InvalidExperiment("parent targets do not match the regenerated evaluation split")
        policy = req.output_policy.model_dump()
        self._invert(ds, victim, rows, preq.attack, policy, deadline)
        for r, p in zip(rows, parent.targets):
            r.undefended_inversion_correct = p.inversion_correct
        service = self._service_rows(ds, victim, policy, with_reference=False)
        lineage = {
            "relation": "adaptive_inversion_of", "parent_experiment_id": parent.experiment_id,
            "parent_evaluation": parent.evaluation, "parent_schema_version": parent.schema_version,
            "parent_config_sha256": parent.reproducibility.get("config_sha256"),
            "parent_results_sha256": parent.reproducibility.get("results_sha256"),
            "parent_record_sha256": parent.reproducibility.get("record_sha256"),
            "shared": {"scenario_id": SCENARIO_ID, "model_id": preq.model_id, "seed": preq.seed, **fp,
                       "victim_artifact_sha256": victim.artifact_sha256(),
                       "auxiliary_artifact_sha256": aux.artifact_sha256(), "target_indices_sha256": sha256_json(idx),
                       "attack_config": preq.attack.model_dump()},
            "checks_passed": ["parent completed", "parent is an inversion attack on the full-probability service",
                              "parent evidence integrity", "dataset split hashes", "victim artifact hash",
                              "auxiliary model artifact hash", "same targets and ground truth",
                              "same attack configuration (budget, tie policy, tolerance)"],
        }
        return {**common, "evaluation": "adaptive_inversion", "intentionally_vulnerable": False,
                "attacker_capabilities": self._capabilities(preq.attack, policy, True),
                "target_system": PredictionService(victim, policy).describe(), "output_policy": policy,
                "attack_config": preq.attack.model_dump(), "defense_config": policy,
                "target_selection": preq.targets.model_dump(), "parent_experiment_id": parent.experiment_id,
                "lineage": lineage, "metrics": derive_inversion_metrics("adaptive_inversion", rows, service),
                "targets": rows, "service": service}
