"""Command-line interface: ``mlbreak <command>``."""

from __future__ import annotations

import argparse
import json
import sys

from .config import cases_dir, data_dir
from .evidence import EvidenceStore
from .experiments import EvidenceIntegrityError, ExperimentEngine
from .regression import run_suite
from .poisoning.schemas import (KnnLabelSanitization, LabelFlipConfig, PoisonAttackRequest, PoisonBaselineRequest,
                                PoisonDefendRequest)
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
    for key in ("clean_accuracy", "coverage", "adversarial_accuracy", "robust_accuracy", "attack_success_rate",
                "abstention_rate_adversarial"):
        if key in m:
            print(f"    {key:<28} {_pct(m[key])}")
    if "outcomes" in m:
        counts = ", ".join(f"{k}={v}" for k, v in m["outcomes"]["counts"].items() if v)
        print(f"    outcomes (n={m['outcomes']['n_selected']}): {counts}")
    if rec.lineage:
        print(f"    lineage: {rec.lineage['relation']} {rec.lineage['parent_experiment_id']} (validated)")
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
    try:
        result = _engine().replay(args.experiment_id)
    except EvidenceIntegrityError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        _print_check(exc.report)
        return 3
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


def _print_check(report: dict) -> None:
    print(f"  {report['experiment_id']} ({report['schema_version']}): "
          f"{'INTEGRITY OK' if report['ok'] else 'INTEGRITY FAILED'}")
    for c in report["checks"]:
        print(f"    [{c['status']:^7}] {c['check']:<20} {c['detail']}")


def cmd_check(args) -> int:
    report = _engine().verify(args.experiment_id)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_check(report)
    return 0 if report["ok"] else 1


_POISON_KEYS = ("clean_test_accuracy", "undefended_test_accuracy", "test_accuracy", "train_accuracy",
                "clean_target_class_recall", "target_class_recall", "poison_rate", "attack_success_rate",
                "undefended_attack_success_rate", "detection_recall", "detection_precision", "false_positive_rate",
                "recovery_rate", "clean_data_test_accuracy")


def _print_poison(rec) -> None:
    print(f"  {rec.experiment_id}  ML02 mode={rec.mode} evaluation={rec.evaluation} status={rec.status}")
    if rec.error:
        print(f"    error: {rec.error}")
    for key in _POISON_KEYS:
        if key in rec.metrics:
            print(f"    {key:<30} {_pct(rec.metrics[key])}")


def cmd_poison(args) -> int:
    lab = _engine().poisoning
    if args.poison_cmd == "baseline":
        rec = lab.run_baseline(PoisonBaselineRequest(model_id=args.model, seed=args.seed))
    elif args.poison_cmd == "attack":
        cfg = LabelFlipConfig(strategy=args.strategy, fraction=args.fraction,
                              target_class=args.target_class if args.strategy == "targeted" else None,
                              selection=args.selection if args.strategy == "targeted" else "random")
        rec = lab.run_attack(PoisonAttackRequest(model_id=args.model, seed=args.seed, poison=cfg))
    else:
        rec = lab.run_defend(PoisonDefendRequest(parent_experiment_id=args.parent, defense=KnnLabelSanitization(
            k=args.k, agreement_threshold=args.threshold, action=args.action)))
    _print_poison(rec)
    return 0 if rec.status == "completed" else 1


def _demo_ml02(eng) -> tuple[list, bool]:
    lab, recs = eng.poisoning, []

    def show(rec):
        recs.append(rec)
        _print_poison(rec)
        return rec

    print("ML02-1) Clean training baseline")
    show(lab.run_baseline(PoisonBaselineRequest()))
    ok = True
    for i, (title, cfg) in enumerate([
        ("random label flipping, 20% of the training set", LabelFlipConfig(strategy="random", fraction=0.2)),
        ("targeted flipping of class 1, 35 samples nearest the clean boundary",
         LabelFlipConfig(strategy="targeted", fraction=0.1, target_class=1, selection="boundary")),
    ], start=2):
        print(f"\nML02-{i}a) Poison + retrain — {title}")
        atk = show(lab.run_attack(PoisonAttackRequest(poison=cfg)))
        print(f"ML02-{i}b) Defensive replay — kNN label sanitisation on the exact poisoned set")
        show(lab.run_defend(PoisonDefendRequest(parent_experiment_id=atk.experiment_id)))
        r = eng.replay(atk.experiment_id)
        print(f"ML02-{i}c) Exact replay of {atk.experiment_id}: identical={r['identical']}")
        ok &= r["identical"]
    return recs, ok


def cmd_demo(args) -> int:
    eng = _engine()
    if args.lab in ("ml02", "all"):
        recs, ok = _demo_ml02(eng)
        integrity = [eng.verify(r.experiment_id) for r in recs]
        print("\nML02 evidence integrity:", ", ".join(f"{i['experiment_id']}={'ok' if i['ok'] else 'FAILED'}"
                                                  for i in integrity))
        ok &= all(i["ok"] for i in integrity)
        if args.lab == "ml02":
            print(f"\nEvidence written to {eng.store.dir}")
            return 0 if ok else 1
        print("\n" + "=" * 72 + "\n")
        return min(1, (0 if ok else 1) + _demo_ml01(eng))
    return _demo_ml01(eng)


def _demo_ml01(eng) -> int:
    records = []

    def show(rec):
        records.append(rec)
        _print(rec)
        return rec

    print("1) Baseline — clean evaluation of the undefended model")
    base = show(eng.run_baseline(BaselineRequest()))
    print("\n2) Attack — white-box PGD L∞ ε=0.3 against the undefended model")
    atk = show(eng.run_attack(AttackRequest()))
    print("\n3) Defend (replay) — same perturbations against an adversarially trained model")
    show(eng.run_defend(DefendRequest(parent_experiment_id=atk.experiment_id,
                                        defense=AdversarialTrainingDefense())))
    print("\n4) Adaptive attack — NEW PGD attack against the adversarially trained model")
    show(eng.run_attack(AttackRequest(target_defense=AdversarialTrainingDefense(),
                                        parent_experiment_id=atk.experiment_id)))
    print("\n5) Defend (replay) — same perturbations against confidence rejection (τ=0.8)")
    show(eng.run_defend(DefendRequest(parent_experiment_id=atk.experiment_id,
                                        defense=ConfidenceRejectionDefense())))
    print("\n6) Adaptive attack — PGD that targets confidence above τ")
    show(eng.run_attack(AttackRequest(target_defense=ConfidenceRejectionDefense(),
                                        parent_experiment_id=atk.experiment_id)))
    print("\n7) Black-box — decision-based random search (500 label queries/sample)")
    show(eng.run_attack(AttackRequest(attack=AttackConfig(attack="random_search", query_budget=500))))
    print(f"\n8) Exact replay of {atk.experiment_id}")
    r = eng.replay(atk.experiment_id)
    print(f"  identical={r['identical']} replay={r['replay_experiment_id']}")
    for note in r["notes"]:
        print(f"  note: {note}")
    print("\n9) Evidence integrity — hashes and metrics re-derived from per-sample results")
    integrity = [eng.verify(rec.experiment_id) for rec in records]
    for rep_ in integrity:
        print(f"  {rep_['experiment_id']}: {'ok' if rep_['ok'] else 'FAILED'}")
    print(f"\nEvidence written to {eng.store.dir}")
    print(f"  {eng.store.json_path(atk.experiment_id)}\n  {eng.store.md_path(atk.experiment_id)}")
    return 0 if r["identical"] and all(i["ok"] for i in integrity) else 1


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

    s = sub.add_parser("check", help="verify an experiment's evidence integrity without re-running it")
    s.add_argument("experiment_id")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser("verify", help="run the security regression suite (non-zero exit on failure)")
    s.set_defaults(fn=cmd_verify)

    s = sub.add_parser("poison", help="ML02 data poisoning: baseline | attack | defend")
    psub = s.add_subparsers(dest="poison_cmd", required=True)
    for name, hlp in (("baseline", "train on clean labels"), ("attack", "flip training labels and retrain")):
        ps = psub.add_parser(name, help=hlp)
        ps.add_argument("--model", default="mlp-tanh-16x16", choices=["mlp-tanh-16x16", "logreg"])
        ps.add_argument("--seed", type=int, default=7)
        if name == "attack":
            ps.add_argument("--strategy", default="random", choices=["random", "targeted"])
            ps.add_argument("--fraction", type=float, default=0.2, help="fraction of the training set to flip")
            ps.add_argument("--target-class", type=int, default=1, choices=[0, 1])
            ps.add_argument("--selection", default="random", choices=["random", "boundary"])
        ps.set_defaults(fn=cmd_poison)
    ps = psub.add_parser("defend", help="kNN label sanitisation on a poisoning experiment's exact training set")
    ps.add_argument("parent")
    ps.add_argument("--k", type=int, default=10)
    ps.add_argument("--threshold", type=float, default=0.5)
    ps.add_argument("--action", default="relabel", choices=["relabel", "remove"])
    ps.set_defaults(fn=cmd_poison)

    s = sub.add_parser("demo", help="run the full Baseline → Attack → Defend → Replay walkthrough")
    s.add_argument("--lab", default="ml01", choices=["ml01", "ml02", "all"])
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
