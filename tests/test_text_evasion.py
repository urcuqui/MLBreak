"""ML01 spam-filter evasion: data, model, attack constraints, defenses, evidence, API and CLI.

Covers the milestone's required regression behaviours: a successful constrained evasion, an unsuccessful
attack, a skipped clean misclassification, query-budget enforcement, deterministic replay, the
text-normalisation defense, adaptive evaluation and evidence integrity.
"""

import json
import unicodedata

import numpy as np
import pytest
from pydantic import ValidationError

from mlbreak.attacks import QueryBudgetExceeded
from mlbreak.evidence import parse_record
from mlbreak.experiments import EvidenceIntegrityError, InvalidExperiment
from mlbreak.text import data as tdata
from mlbreak.text import engine as tengine
from mlbreak.text.attacks import (STOPWORDS, AttackResult, Oracle, all_edits, apply_edits, deepwordbug_attack,
                                  levenshtein, segment, validate_constraints)
from mlbreak.text.data import CorpusIntegrityError, load_corpus, load_text_dataset
from mlbreak.text.defenses import base_text_model
from mlbreak.text.integrity import verify_text_record
from mlbreak.text.models import train_text_model
from mlbreak.text.normalize import CONFUSABLES, INVISIBLE, Normalizer
from mlbreak.text.schemas import (TextAdversarialTrainingDefense, TextAttackConfig, TextAttackRequest,
                                  TextBaselineRequest, TextClassifyRequest, TextDefendRequest, TextEdit,
                                  TextNormalizationDefense, TextSampleSelection)

DS = load_text_dataset(7)
MODEL = base_text_model("tfidf-word-logreg", 7)
CFG = TextAttackConfig()
SPAM_TEST = [i for i, y in enumerate(DS.test_labels) if y == 1]
SMALL = TextSampleSelection(max_samples=40)


def _attack(i, cfg=CFG, model=MODEL):
    return deepwordbug_attack(DS.test_texts[i], DS.test_labels[i], model.p_spam, cfg, np.random.default_rng([7, 1001, i]))


@pytest.fixture
def lab(engine):
    return engine.text


@pytest.fixture
def attack_rec(lab):
    return lab.run_attack(TextAttackRequest(samples=SMALL))


# ---------------------------------------------------------------------------
# Data: bundled, verified, de-duplicated, leakage-free split
# ---------------------------------------------------------------------------
def test_corpus_is_verified_and_deduplicated():
    texts, labels, n_raw = load_corpus()
    assert n_raw == 5574 and len(texts) == 5574 - 415
    keys = [" ".join(t.lower().split()) for t in texts]
    assert len(set(keys)) == len(keys)
    assert set(labels) == {0, 1}


def test_no_message_in_both_splits_and_split_is_deterministic():
    norm = lambda t: " ".join(t.lower().split())  # noqa: E731
    assert not {norm(t) for t in DS.train_texts} & {norm(t) for t in DS.test_texts}
    load_text_dataset.cache_clear()
    again = load_text_dataset(7)
    assert again.fingerprint() == DS.fingerprint()
    assert load_text_dataset(8).fingerprint() != DS.fingerprint()
    d = DS.describe()
    assert (d["n_train"], d["n_test"]) == (4127, 1032) and d["license"] == "CC BY 4.0"


def test_modified_corpus_is_refused(monkeypatch):
    load_corpus.cache_clear()
    monkeypatch.setattr(tdata, "CORPUS_SHA256", "0" * 64)
    with pytest.raises(CorpusIntegrityError):
        load_corpus()
    monkeypatch.undo()
    load_corpus.cache_clear()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_id", ["tfidf-word-logreg", "tfidf-word-nb"])
def test_models_are_reproducible_and_accurate(model_id):
    a = train_text_model(model_id, DS.train_texts, DS.train_labels, 7)
    b = train_text_model(model_id, DS.train_texts, DS.train_labels, 7)
    assert a.artifact_sha256() == b.artifact_sha256()
    pred = a.predict(DS.test_texts)
    assert (pred == DS.y_test).mean() >= 0.95
    assert np.array_equal(pred, (a.p_spam(DS.test_texts) >= 0.5).astype(int))
    d = a.describe()
    assert d["tfidf"]["ngram_range"] == [1, 2] and d["tfidf"]["analyzer"] == "word" and d["n_train"] == 4127


def test_normalizer_uses_training_vocabulary_only():
    norm = Normalizer(TextNormalizationDefense(), DS.train_texts)
    train_words = {w for t in DS.train_texts for w in norm._pre(t).split()}
    assert set(norm.vocab) <= {w for t in DS.train_texts for w in __import__("re").findall(r"[a-z']+", norm._pre(t))}
    assert train_words  # sanity
    assert norm("fr​ee") == norm("free")                       # invisible characters removed
    assert norm("frее") == "free"                                    # Cyrillic е → e
    assert norm("fr3e") == "free"                                    # leetspeak between letters
    assert norm("frwe") == "free" and norm("pirze") == "prize"       # spell snap to a frequent training word
    assert norm("08712400200") == "08712400200"                      # digits untouched


# ---------------------------------------------------------------------------
# Attack: constraints, outcomes, budget, determinism
# ---------------------------------------------------------------------------
def test_successful_constrained_evasion():
    results = [(i, _attack(i)) for i in SPAM_TEST[:60]]
    evaded = [(i, r) for i, r in results if r.outcome == "evaded"]
    assert evaded, "the attack must find at least one constrained evasion"
    for i, r in evaded:
        text = DS.test_texts[i]
        checks, stats = validate_constraints(text, r.text, CFG, r.edits)
        assert all(checks.values()), checks
        assert r.p_spam < 0.5 and MODEL.predict([r.text])[0] == 0          # label really changed
        assert stats["edit_distance"] <= CFG.max_edit_distance
        assert stats["edit_distance"] <= CFG.max_char_ratio * len(text)
        assert r.queries <= CFG.query_budget
        assert apply_edits(text, CFG, r.edits) == r.text


def test_unsuccessful_attack_is_reported_as_failed():
    i = max(SPAM_TEST, key=lambda k: MODEL.p_spam([DS.test_texts[k]])[0])  # most confidently spam
    cfg = TextAttackConfig(max_edit_distance=1)
    r = _attack(i, cfg)
    assert r.outcome == "failed" and r.p_spam >= 0.5 and MODEL.predict([r.text])[0] == 1
    assert levenshtein(DS.test_texts[i], r.text) <= 1


def test_clean_misclassification_is_skipped():
    missed = [i for i in SPAM_TEST if MODEL.predict([DS.test_texts[i]])[0] == 0]
    assert missed
    r = _attack(missed[0])
    assert r.outcome == "skipped_misclassified" and r.queries == 1 and r.text == DS.test_texts[missed[0]] and not r.edits


def test_query_budget_is_enforced():
    oracle = Oracle(MODEL.p_spam, budget=3)
    oracle(["a", "b"])
    with pytest.raises(QueryBudgetExceeded):
        oracle(["c", "d"])
    assert oracle.used == 2
    tiny = TextAttackConfig(query_budget=5)
    outs = [_attack(i, tiny) for i in SPAM_TEST[:30]]
    assert all(r.queries <= 5 for r in outs)
    assert any(r.outcome == "budget_exhausted" for r in outs)


def test_protected_tokens_and_stopwords_are_never_modified():
    text = "Call 09061701461 now to claim your FREE prize at www.prizes.co.uk or win@prizes.com"
    seg = segment(text, CFG)
    modifiable = {seg.segments[i] for i in seg.modifiable}
    assert {"Call", "claim", "FREE", "prize"} <= modifiable
    assert not modifiable & {"09061701461", "www", "prizes", "co", "uk", "win", "com", "to", "your", "at", "or"}
    assert all(seg.segments[i].lower() not in STOPWORDS for i in seg.modifiable)


@pytest.mark.parametrize("ops,charsets", [(["swap"], ["ascii_letters"]), (["insert"], ["invisible"]),
                                          (["substitute"], ["confusables"]), (["delete", "insert"], ["ascii_letters"])])
def test_only_permitted_operations_and_characters(ops, charsets):
    cfg = TextAttackConfig(operations=ops, charsets=charsets)
    for i in SPAM_TEST[:25]:
        r = _attack(i, cfg)
        for e in r.edits:
            assert e.op in ops
            if e.char_after is not None and e.op != "swap":
                if charsets == ["invisible"]:
                    assert e.char_after in INVISIBLE
                elif charsets == ["confusables"]:
                    assert e.char_after in CONFUSABLES
                else:
                    assert e.char_after.isascii() and e.char_after.isalpha()
        if r.outcome != "skipped_misclassified":
            checks, _ = validate_constraints(DS.test_texts[i], r.text, cfg, r.edits)
            assert all(checks.values())
            r.text.encode("utf-8")
            assert not any(unicodedata.category(c) in ("Cc", "Cs", "Co", "Cn") for c in r.text)


def test_validation_rejects_constraint_violations():
    text = DS.test_texts[SPAM_TEST[0]]
    seg = segment(text, CFG)
    w = seg.modifiable[0]
    bad_edit = TextEdit(word_index=w, op="substitute", position=0, char_before=seg.segments[w][0], char_after="\x07",
                        word_before=seg.segments[w], word_after="\x07" + seg.segments[w][1:])
    adv = apply_edits(text, CFG, [bad_edit])
    checks, _ = validate_constraints(text, adv, CFG, [bad_edit])
    assert not checks["charset_allowed"] and not checks["valid_unicode"]
    checks, _ = validate_constraints(text, text + "x" * 40, CFG, [])
    assert not checks["edit_distance"] and not checks["edit_script_reproduces_text"]


def test_attack_is_deterministic():
    a = [_attack(i).text for i in SPAM_TEST[:20]]
    b = [_attack(i).text for i in SPAM_TEST[:20]]
    assert a == b


def test_all_edits_are_single_edits():
    for w in ("free", "prize", "aa"):
        for op, pos, before, after, new in all_edits(w, CFG):
            assert levenshtein(w, new) == (2 if op == "swap" else 1) and new != w


@pytest.mark.parametrize("kwargs", [
    {"operations": []}, {"operations": ["swap", "swap"]}, {"charsets": ["invisible"], "operations": ["swap"]},
    {"query_budget": 0}, {"query_budget": 10**6}, {"max_edit_distance": 0}, {"max_char_ratio": 0.9},
    {"goal": "make_it_spam"}, {"min_target_confidence": 0.4}, {"exec": "x"},
])
def test_attack_config_validation(kwargs):
    with pytest.raises(ValidationError):
        TextAttackConfig(**kwargs)


# ---------------------------------------------------------------------------
# Experiments: evidence, defenses, replay, adaptive
# ---------------------------------------------------------------------------
def test_attack_record_evidence(attack_rec, store):
    data = json.loads(store.json_path(attack_rec.experiment_id).read_text())
    for key in ("scenario_id", "owasp", "model", "target_system", "dataset", "seed", "attack_config", "samples",
                "metrics", "reproducibility", "attacker_capabilities", "test_predictions"):
        assert key in data, key
    assert data["owasp"]["id"] == "ML01:2023" and data["scenario_id"] == "ml01-spam-filter-evasion"
    assert data["model"]["tfidf"]["ngram_range"] == [1, 2] and data["model"]["artifact_sha256"]
    assert data["dataset"]["train_sha256"] and data["dataset"]["test_sha256"]
    m = attack_rec.metrics
    assert sum(m["outcomes"]["counts"].values()) == len(attack_rec.samples) == 40
    asr = m["attack_success_rate"]
    assert asr["denominator"] == m["outcomes"]["attacked"] and asr["numerator"] == m["outcomes"]["counts"]["evaded"]
    assert m["constraint_violation_rate"]["numerator"] == 0
    for s in attack_rec.samples:
        if s.outcome in ("evaded", "failed", "budget_exhausted"):
            assert s.text_adversarial is not None and s.edit_distance is not None and s.queries <= 200
            assert all(s.constraints.values())
        else:
            assert s.text_adversarial is None
    assert verify_text_record(attack_rec)["ok"]


def test_constraint_violating_evasion_fails_the_experiment(lab, monkeypatch):
    """A label change alone is not success: an 'evaded' result that breaks a constraint is refused."""
    def cheat(text, y, p_spam, cfg, rng, deadline=None):
        return AttackResult("evaded", text + " zz" * 30, 0.1, 3, [], 1, 0)
    monkeypatch.setattr(tengine, "deepwordbug_attack", cheat)
    rec = lab.run_attack(TextAttackRequest(samples=TextSampleSelection(indices=[SPAM_TEST[0]])))
    assert rec.status == "failed" and "TextConstraintViolation" in rec.error and rec.samples == []


def test_replay_is_deterministic(engine, attack_rec):
    r = engine.replay(attack_rec.experiment_id)
    assert r["identical"], r["checks"]
    assert {"adversarial_texts_sha256", "results_sha256", "model_artifact_sha256"} <= {c["check"] for c in r["checks"]}


def test_normalisation_replay_uses_exact_parent_messages(lab, attack_rec):
    d = lab.run_defend(TextDefendRequest(parent_experiment_id=attack_rec.experiment_id, defense=TextNormalizationDefense()))
    assert d.evaluation == "replay" and d.lineage["relation"] == "replay_against_defense_of"
    parent = {s.test_index: s.text_adversarial for s in attack_rec.samples}
    for s in d.samples:
        if s.text_adversarial is not None:
            assert s.text_adversarial == parent[s.test_index] and s.queries == 0
    cmp = d.metrics["comparison"]["attack_success_rate"]
    assert cmp["defended"]["value"] < cmp["undefended"]["value"]
    assert verify_text_record(d)["ok"]


def test_unicode_normalisation_neutralises_invisible_characters(lab):
    cfg = TextAttackConfig(operations=["insert"], charsets=["invisible"])
    a = lab.run_attack(TextAttackRequest(attack=cfg, samples=SMALL))
    assert a.metrics["attack_success_rate"]["numerator"] > 0
    ev = next(s for s in a.samples if s.outcome == "evaded")
    assert "".join(c for c in ev.text_adversarial if c not in INVISIBLE) == ev.text_original  # visually identical
    unicode_only = TextNormalizationDefense(leetspeak=False, spell_snap=False)
    rp = lab.run_defend(TextDefendRequest(parent_experiment_id=a.experiment_id, defense=unicode_only))
    ad = lab.run_attack(TextAttackRequest(attack=cfg, samples=SMALL, target_defense=unicode_only,
                                          parent_experiment_id=a.experiment_id))
    assert rp.metrics["attack_success_rate"]["numerator"] == 0
    assert ad.metrics["attack_success_rate"]["numerator"] == 0


def test_adaptive_attack_is_a_separate_constrained_experiment(lab, attack_rec):
    ad = lab.run_attack(TextAttackRequest(samples=SMALL, target_defense=TextNormalizationDefense(),
                                          parent_experiment_id=attack_rec.experiment_id))
    assert ad.evaluation == "adaptive_attack" and ad.attacker_capabilities["knows_defense"]
    assert [s.test_index for s in ad.samples] == [s.test_index for s in attack_rec.samples]
    assert ad.target_system["model_artifact_sha256"] != attack_rec.model["artifact_sha256"]
    assert any(s.queries and s.queries > 1 for s in ad.samples)                       # it searched again
    assert ad.metrics["constraint_violation_rate"]["numerator"] == 0


def test_adaptive_attack_must_keep_parent_constraints(lab, attack_rec):
    looser = TextAttackConfig(max_edit_distance=30)
    with pytest.raises(InvalidExperiment, match="max_edit_distance"):
        lab.run_attack(TextAttackRequest(attack=looser, samples=SMALL, target_defense=TextNormalizationDefense(),
                                         parent_experiment_id=attack_rec.experiment_id))
    # Effort knobs may change: a stronger adaptive attacker is allowed.
    stronger = TextAttackConfig(max_edits_per_word=2, query_budget=400)
    ok = lab.run_attack(TextAttackRequest(attack=stronger, samples=SMALL, target_defense=TextNormalizationDefense(),
                                          parent_experiment_id=attack_rec.experiment_id))
    assert ok.status == "completed"


def test_adversarial_augmentation_uses_training_data_only(lab, monkeypatch):
    from mlbreak.text import defenses as tdef
    seen = []
    real = tdef.deepwordbug_attack

    def spy(text, *a, **k):
        seen.append(text)
        return real(text, *a, **k)
    monkeypatch.setattr(tdef, "deepwordbug_attack", spy)
    tdef.clear_text_caches()
    cfg = TextAdversarialTrainingDefense(max_train_samples=30, query_budget=60)
    rec = lab.run_baseline(TextBaselineRequest(defense=cfg))
    assert len(seen) == 30 and set(seen) <= set(DS.train_texts) and not set(seen) & set(DS.test_texts)
    aug = rec.target_system["augmentation"]
    assert aug["attacked"] == 30 and aug["source"].startswith("training split")
    tdef.clear_text_caches()


def test_defended_systems_are_evaluated_on_the_same_held_out_split(lab):
    recs = [lab.run_baseline(TextBaselineRequest(defense=d)) for d in
            (None, TextNormalizationDefense(), TextAdversarialTrainingDefense(max_train_samples=30, query_budget=60))]
    assert len({r.dataset["test_sha256"] for r in recs}) == 1
    assert all(len(r.test_predictions) == 1032 for r in recs)


# ---------------------------------------------------------------------------
# Evidence integrity
# ---------------------------------------------------------------------------
def _tamper(store, exp_id, fn):
    path = store.json_path(exp_id)
    d = json.loads(path.read_text())
    fn(d)
    path.write_text(json.dumps(d))
    return parse_record(d)


def _first(d, outcome):
    return next(s for s in d["samples"] if s["outcome"] == outcome)


@pytest.mark.parametrize("edit,check", [
    (lambda d: _first(d, "evaded").update(text_adversarial=_first(d, "evaded")["text_original"] + " FREE"),
     "constraints_revalidated"),
    (lambda d: _first(d, "failed").update(outcome="evaded"), "sample_consistency"),
    (lambda d: d["metrics"]["attack_success_rate"].update(numerator=0), "metrics_derivation"),
    (lambda d: d["request"]["attack"].update(max_edit_distance=40), "config_sha256"),
])
def test_tampering_is_detected_and_blocks_replay(engine, store, attack_rec, edit, check):
    rec = _tamper(store, attack_rec.experiment_id, edit)
    st = {c["check"]: c["status"] for c in verify_text_record(rec)["checks"]}
    assert st[check] == "fail" and st["record_sha256"] == "fail"
    with pytest.raises(EvidenceIntegrityError):
        engine.replay(attack_rec.experiment_id)


def test_tampered_parent_cannot_be_replayed_against_a_defense(lab, store, attack_rec):
    _tamper(store, attack_rec.experiment_id, lambda d: d.update(created_at="2000-01-01T00:00:00+00:00"))
    with pytest.raises(EvidenceIntegrityError):
        lab.run_defend(TextDefendRequest(parent_experiment_id=attack_rec.experiment_id, defense=TextNormalizationDefense()))


# ---------------------------------------------------------------------------
# API and CLI
# ---------------------------------------------------------------------------
def test_api_flow(client):
    msgs = client.get("/api/text/messages?label=1&limit=5").json()["messages"]
    assert len(msgs) == 5 and all(m["y_true"] == 1 for m in msgs)
    c = client.post("/api/text/classify", json={"text": "WINNER! Claim your free prize now"}).json()
    assert c["label"] in ("spam", "ham") and abs(c["p_spam"] + c["p_ham"] - 1) < 1e-9
    assert client.post("/api/experiments/text/baseline", json={}).json()["family"] == "text_evasion"
    a = client.post("/api/experiments/text/attack", json={"samples": {"max_samples": 15}})
    assert a.status_code == 200, a.text
    aid = a.json()["experiment_id"]
    d = client.post("/api/experiments/text/defend", json={"parent_experiment_id": aid,
                                                          "defense": {"kind": "text_normalization"}})
    assert d.status_code == 200 and d.json()["evaluation"] == "replay"
    ad = client.post("/api/experiments/text/attack", json={"samples": {"max_samples": 15}, "parent_experiment_id": aid,
                                                           "target_defense": {"kind": "text_normalization"}})
    assert ad.status_code == 200 and ad.json()["evaluation"] == "adaptive_attack"
    assert client.get(f"/api/experiments/{aid}/verify").json()["ok"]
    assert client.post(f"/api/experiments/{aid}/replay").json()["identical"]
    md = client.get(f"/api/experiments/{aid}/report.md").text
    assert "ML01:2023" in md and "TF-IDF configuration" in md and "constraints" in md
    csv = client.get(f"/api/experiments/{aid}/samples.csv").text.splitlines()
    assert csv[0].startswith("test_index") and len(csv) == 16
    listed = {e["experiment_id"]: e for e in client.get("/api/experiments").json()}
    assert listed[aid]["family"] == "text_evasion" and listed[aid]["attack"] == "deepwordbug:spam_to_ham"


@pytest.mark.parametrize("path,body", [
    ("experiments/text/attack", {"attack": {"query_budget": 100000}}),
    ("experiments/text/attack", {"attack": {"charsets": ["emoji"]}}),
    ("experiments/text/attack", {"samples": {"indices": [10**6]}}),
    ("experiments/text/attack", {"model_id": "bert-from-the-internet"}),
    ("experiments/text/defend", {"parent_experiment_id": "MLB-EXP-xyz", "defense": {"kind": "text_normalization"}}),
    ("text/classify", {"text": ""}),
    ("text/classify", {"text": "x" * 5000}),
])
def test_api_rejects_invalid_text_requests(client, path, body):
    assert client.post(f"/api/{path}", json=body).status_code == 422


def test_api_rejects_indices_of_the_wrong_class(client):
    ham = next(i for i, y in enumerate(DS.test_labels) if y == 0)
    r = client.post("/api/experiments/text/attack", json={"samples": {"indices": [ham]}})
    assert r.status_code == 422 and "source class" in r.json()["detail"]


def test_cli_text_commands(tmp_path, monkeypatch, capsys):
    from mlbreak import cli
    monkeypatch.setenv("MLBREAK_DATA_DIR", str(tmp_path / "var"))
    assert cli.main(["text", "classify", "Free entry! Txt WIN to 80086 to claim"]) == 0
    assert cli.main(["text", "attack", "--samples", "10"]) == 0
    aid = [w for w in capsys.readouterr().out.split() if w.startswith("MLB-EXP-")][-1]
    assert cli.main(["text", "defend", aid, "--defense", "text_normalization"]) == 0
    assert cli.main(["check", aid]) == 0 and cli.main(["replay", aid]) == 0
    assert cli.main(["text", "attack", "--queries", "0"]) == 2
