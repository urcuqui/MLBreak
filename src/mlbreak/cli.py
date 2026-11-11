"""Command-line interface: ``mlbreak <command>``."""

from __future__ import annotations

import argparse
import json
import sys

from .config import cases_dir, data_dir
from .evidence import EvidenceStore
from .experiments import ExperimentEngine
from .regression import run_suite
from .schemas import (AdversarialTrainingDefense, AttackConfig, AttackRequest, BaselineRequest,
                      ConfidenceRejectionDefense, DefendRequest)

BANNER = ("MLBreak — intentionally vulnerable adversarial ML lab (OWASP ML01:2023). "
          "Educational and defensive research use only.")


def _engine() -> ExperimentEngine:
    return ExperimentEngine(EvidenceStore(data_dir()))


def _pct(r: dict | None) -> str:
    if not r:
        return "n/a"
    v = r["value"]
    return ("undefined" if v is None else f"{100 * v:5.1f}%") + f" ({r['numerator']}/{r['denominator']})"


def _print(rec) -> None:
    m = rec.metrics
    print(f"  {rec.experiment_id}  mode={rec.mode} evaluation={rec.evaluation} status={rec.status}")
    if rec.error:
        print(f"    error: {rec.error}")
    for key in ("clean_accuracy", "coverage", "adversarial_accuracy", "attack_success_rate"):
        if key in m:
            print(f"    {key:<22} {_pct(m[key])}")
    if "perturbation" in m and m["perturbation"]["attacked"]["n"]:
        p = m["perturbation"]
        print(f"    mean ‖δ‖ ({p['norm']}): L∞={p['attacked']['linf_mean']:.4f}  "
              f"max ‖δ‖/ε={p['max_budget_ratio']:.4f}")


def _defense(args):
    if args.defense == "adversarial_training":
        return AdversarialTrainingDefense(epsilon=args.train_epsilon, rounds=args.rounds)
    if args.defense == "confidence_rejection":
        return ConfidenceRejectionDefense(threshold=args.threshold)
    return None


def _attack_cfg(args) -> AttackConfig:
    return AttackConfig(attack=args.attack, norm=args.norm, epsilon=args.epsilon, steps=args.steps,
                        query_budget=args.queries)


def cmd_serve(args) -> int:
    import uvicorn

    from .api import create_app

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"WARNING: binding to {args.host} exposes an intentionally vulnerable lab beyond localhost.",
              file=sys.stderr)
    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


def cmd_baseline(args) -> int:
    rec = _engine().run_baseline(BaselineRequest(model_id=args.model, seed=args.seed, defense=_defense(args)))
    _print(rec)
    return 0 if rec.status == "completed" else 1


def cmd_attack(args) -> int:
    rec = _engine().run_attack(AttackRequest(model_id=args.model, seed=args.seed, attack=_attack_cfg(args),
                                             target_defense=_defense(args), parent_experiment_id=args.parent))
    _print(rec)
    return 0 if rec.status == "completed" else 1


def cmd_defend(args) -> int:
    rec = _engine().run_defend(DefendRequest(parent_experiment_id=args.parent, defense=_defense(args)))
    _print(rec)
    return 0 if rec.status == "completed" else 1


def cmd_replay(args) -> int:
    result = _engine().replay(args.experiment_id)
    print(json.dumps(result, indent=2))
    return 0 if result["identical"] else 1


def cmd_list(args) -> int:
    for e in EvidenceStore(data_dir()).list():
        print(f"{e['experiment_id']}  {e['created_at'][:19]}  {e['evaluation']:<16} {e['status']:<9} "
              f"attack={e['attack'] or '-':<13} defense={e['defense'] or '-':<22} ASR={e['attack_success'] or '-'}")
    return 0


def cmd_export(args) -> int:
    store = EvidenceStore(data_dir())
    print(store.markdown(args.experiment_id) if args.format == "md"
          else store.json_path(args.experiment_id).read_text())
    return 0


def cmd_verify(args) -> int:
    report = run_suite(_engine(), cases_dir(), data_dir() / "regression")
    for c in report["cases"]:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['id']} {c['title']}")
        for i in c["invariants"]:
            obs = "n/a" if i["observed"] is None else f"{i['observed']:.4f}"
            print(f"       {'ok ' if i['passed'] else 'BAD'} {i['kind']:<13} {i['posture']:<10} "
                  f"{i['metric']} {i['op']} {i['value']}  (observed {obs})")
    print(f"\n{report['passed']}/{report['total']} cases passed. "
          f"Report: {data_dir() / 'regression' / 'security_regression_report.md'}")
    return 0 if report["ok"] else 1


def cmd_demo(args) -> int:
    eng = _engine()
    print("1) Baseline — clean evaluation of the undefended model")
    base = eng.run_baseline(BaselineRequest())
    _print(base)
    print("\n2) Attack — white-box PGD L∞ ε=0.3 against the undefended model")
    atk = eng.run_attack(AttackRequest())
    _print(atk)
    print("\n3) Defend (replay) — same perturbations against an adversarially trained model")
    _print(eng.run_defend(DefendRequest(parent_experiment_id=atk.experiment_id,
                                        defense=AdversarialTrainingDefense())))
    print("\n4) Adaptive attack — NEW PGD attack against the adversarially trained model")
    _print(eng.run_attack(AttackRequest(target_defense=AdversarialTrainingDefense(),
                                        parent_experiment_id=atk.experiment_id)))
    print("\n5) Defend (replay) — same perturbations against confidence rejection (τ=0.8)")
    _print(eng.run_defend(DefendRequest(parent_experiment_id=atk.experiment_id,
                                        defense=ConfidenceRejectionDefense())))
    print("\n6) Adaptive attack — PGD that targets confidence above τ")
    _print(eng.run_attack(AttackRequest(target_defense=ConfidenceRejectionDefense(),
                                        parent_experiment_id=atk.experiment_id)))
    print("\n7) Black-box — decision-based random search (500 label queries/sample)")
    _print(eng.run_attack(AttackRequest(attack=AttackConfig(attack="random_search", query_budget=500))))
    print(f"\n8) Exact replay of {atk.experiment_id}")
    r = eng.replay(atk.experiment_id)
    print(f"  identical={r['identical']} replay={r['replay_experiment_id']}")
    print(f"\nEvidence written to {eng.store.dir}")
    print(f"  {eng.store.json_path(atk.experiment_id)}\n  {eng.store.md_path(atk.experiment_id)}")
    return 0 if r["identical"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mlbreak", description=BANNER)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="start the web UI + API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    def common(sp, attack=False, defense=True):
        sp.add_argument("--model", default="mlp-tanh-16x16", choices=["mlp-tanh-16x16", "logreg"])
        sp.add_argument("--seed", type=int, default=7)
        if attack:
            sp.add_argument("--attack", default="pgd", choices=["fgsm", "pgd", "random_search"])
            sp.add_argument("--norm", default="linf", choices=["linf", "l2"])
            sp.add_argument("--epsilon", type=float, default=0.3)
            sp.add_argument("--steps", type=int, default=40)
            sp.add_argument("--queries", type=int, default=500)
        if defense:
            sp.add_argument("--defense", choices=["adversarial_training", "confidence_rejection"])
            sp.add_argument("--train-epsilon", type=float, default=0.2)
            sp.add_argument("--rounds", type=int, default=3)
            sp.add_argument("--threshold", type=float, default=0.8)

    s = sub.add_parser("baseline", help="clean evaluation (optionally of a defended system)")
    common(s)
    s.set_defaults(fn=cmd_baseline)

    s = sub.add_parser("attack", help="attack the model; with --defense this is an ADAPTIVE attack")
    common(s, attack=True)
    s.add_argument("--parent")
    s.set_defaults(fn=cmd_attack)

    s = sub.add_parser("defend", help="replay an attack experiment's perturbations against a defense")
    s.add_argument("parent")
    s.add_argument("--defense", required=True, choices=["adversarial_training", "confidence_rejection"])
    s.add_argument("--train-epsilon", type=float, default=0.2)
    s.add_argument("--rounds", type=int, default=3)
    s.add_argument("--threshold", type=float, default=0.8)
    s.set_defaults(fn=cmd_defend)

    s = sub.add_parser("replay", help="re-execute an experiment and verify identical results")
    s.add_argument("experiment_id")
    s.set_defaults(fn=cmd_replay)

    s = sub.add_parser("list", help="list stored experiments")
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("export", help="print an experiment's evidence")
    s.add_argument("experiment_id")
    s.add_argument("--format", choices=["json", "md"], default="md")
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("verify", help="run the security regression suite (non-zero exit on failure)")
    s.set_defaults(fn=cmd_verify)

    s = sub.add_parser("demo", help="run the full Baseline → Attack → Defend → Replay walkthrough")
    s.set_defaults(fn=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (KeyError, ValueError) as exc:  # includes pydantic.ValidationError
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
