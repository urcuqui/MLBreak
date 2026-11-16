"""Scientific-validity checks for ML01 spam-filter evasion (consolidation review).

Each test pins one claim made by the scenario's documentation:

* every successful evasion satisfies its declared constraints (over a seeded sweep of configurations);
* success rates have explicit, correct denominators and undefined ≠ 0;
* perturbations are reproducible and independent of which other messages are attacked;
* query limits hold for every budget;
* protected tokens (digits, URLs, e-mails, function words) are unchanged in every adversarial message;
* normalisation and augmentation use training data only;
* adaptive attacks query the DEFENDED pipeline;
* constraint-valid evasion is never presented as semantic preservation.
"""

import re

import numpy as np
import pytest

from mlbreak.text import normalize as tnorm
from mlbreak.text.attacks import PROTECTED_SPAN_RE, STOPWORDS, deepwordbug_attack, segment, validate_constraints
from mlbreak.text.data import load_text_dataset
from mlbreak.text.defenses import base_text_model, clear_text_caches
from mlbreak.text.metrics import derive_text_metrics
from mlbreak.text.normalize import Normalizer
from mlbreak.text.schemas import (TextAttackConfig, TextAttackRequest, TextNormalizationDefense, TextSampleSelection)

DS = load_text_dataset(7)
MODEL = base_text_model("tfidf-word-logreg", 7)
SPAM = [i for i, y in enumerate(DS.test_labels) if y == 1]


def _attack(i, cfg):
    return deepwordbug_attack(DS.test_texts[i], DS.test_labels[i], MODEL.p_spam, cfg, np.random.default_rng([7, 1001, i]))


def _random_configs(n=8, seed=2026):
    rng = np.random.default_rng(seed)
    ops_all, cs_all = ["swap", "substitute", "delete", "insert"], ["ascii_letters", "confusables", "invisible"]
    out = []
    while len(out) < n:
        ops = [o for o in ops_all if rng.random() < 0.7] or ["substitute"]
        cs = [c for c in cs_all if rng.random() < 0.5] or ["ascii_letters"]
        try:
            out.append(TextAttackConfig(operations=ops, charsets=cs, max_edit_distance=int(rng.integers(2, 16)),
                                        max_char_ratio=float(rng.uniform(0.05, 0.3)),
                                        max_modified_word_ratio=float(rng.uniform(0.2, 1.0)),
                                        max_edits_per_word=int(rng.integers(1, 3)), query_budget=int(rng.integers(20, 300)),
                                        candidates_per_word=int(rng.integers(2, 12))))
        except ValueError:  # e.g. invisible charset without insert: rejected by the schema, as intended
            continue
    return out


@pytest.mark.parametrize("cfg", _random_configs(), ids=lambda c: f"{'+'.join(c.operations)}|{'+'.join(c.charsets)}|ed{c.max_edit_distance}")
def test_every_evasion_satisfies_its_declared_constraints(cfg):
    for i in SPAM[:30]:
        r = _attack(i, cfg)
        assert r.queries <= cfg.query_budget
        if r.outcome == "skipped_misclassified":
            continue
        checks, stats = validate_constraints(DS.test_texts[i], r.text, cfg, r.edits)
        assert all(checks.values()), (i, checks)              # also holds for failed (partially edited) messages
        assert stats["edit_distance"] <= min(cfg.max_edit_distance, cfg.max_char_ratio * len(DS.test_texts[i]))
        if r.outcome == "evaded":
            assert (1 - r.p_spam) >= cfg.min_target_confidence and MODEL.predict([r.text])[0] == 0


def _protected_tokens(text: str, cfg: TextAttackConfig) -> list[tuple[int, str]]:
    seg = segment(text, cfg)
    return [(i, s) for i in seg.word_indices if i not in set(seg.modifiable) for s in [seg.segments[i]]]


@pytest.mark.parametrize("charsets,ops", [(["ascii_letters"], ["swap", "substitute", "delete", "insert"]),
                                          (["invisible"], ["insert"]), (["confusables"], ["substitute"])])
def test_protected_tokens_are_unchanged_in_every_adversarial_message(engine, charsets, ops):
    cfg = TextAttackConfig(operations=ops, charsets=charsets)
    rec = engine.text.run_attack(TextAttackRequest(attack=cfg, samples=TextSampleSelection(max_samples=60)))
    checked = 0
    for s in rec.samples:
        if s.text_adversarial is None:
            continue
        orig, adv = segment(s.text_original, cfg).segments, None
        # Rebuild the adversarial segment list from the edit script and compare every non-modifiable token.
        adv = list(orig)
        for e in s.edits or []:
            adv[e.word_index] = e.word_after
        assert "".join(adv) == s.text_adversarial
        for idx, tok in _protected_tokens(s.text_original, cfg):
            assert adv[idx] == tok, (s.test_index, tok)
            checked += 1
        # Independent of segmentation: every digit run, URL/e-mail and function word survives verbatim.
        for d in re.findall(r"\d+", s.text_original):
            assert d in s.text_adversarial
        for m in PROTECTED_SPAN_RE.finditer(s.text_original):
            assert m.group(0) in s.text_adversarial
        words_adv = re.findall(r"[^\W_]+(?:'[^\W_]+)*", s.text_adversarial)
        for w in re.findall(r"[^\W_]+(?:'[^\W_]+)*", s.text_original):
            if w.lower() in STOPWORDS and charsets != ["invisible"]:
                assert w in words_adv
    assert checked > 100


def test_perturbation_is_independent_of_the_other_selected_messages(engine):
    """Per-message RNG streams: attacking message i alone or inside a batch yields the same perturbation."""
    batch = engine.text.run_attack(TextAttackRequest(samples=TextSampleSelection(max_samples=40)))
    for s in [x for x in batch.samples if x.outcome == "evaded"][:3] + [x for x in batch.samples if x.outcome == "failed"][:2]:
        alone = engine.text.run_attack(TextAttackRequest(samples=TextSampleSelection(indices=[s.test_index]))).samples[0]
        assert (alone.text_adversarial, alone.queries, alone.outcome) == (s.text_adversarial, s.queries, s.outcome)


@pytest.mark.parametrize("budget", [1, 2, 3, 5, 8, 13, 21])
def test_query_budget_holds_for_every_budget(budget):
    cfg = TextAttackConfig(query_budget=budget)
    for i in SPAM[:25]:
        r = _attack(i, cfg)
        assert r.queries <= budget
        if budget == 1 and r.outcome != "skipped_misclassified":
            assert r.outcome == "budget_exhausted" and r.text == DS.test_texts[i]   # only the clean query fit


def test_denominators_are_explicit_and_undefined_is_not_zero(engine):
    rec = engine.text.run_attack(TextAttackRequest(samples=TextSampleSelection(max_samples=50)))
    m = rec.metrics
    for k, v in m.items():
        if isinstance(v, dict) and "numerator" in v:
            assert v["population"] and v["definition"], k
    counts = m["outcomes"]["counts"]
    assert m["attack_success_rate"]["denominator"] == counts["evaded"] + counts["failed"] + counts["budget_exhausted"]
    missed = [s.test_index for s in rec.samples if s.outcome == "skipped_misclassified"]
    assert missed
    only_missed = engine.text.run_attack(TextAttackRequest(samples=TextSampleSelection(indices=missed)))
    asr = only_missed.metrics["attack_success_rate"]
    assert asr["denominator"] == 0 and asr["value"] is None                       # undefined, never 0
    assert derive_text_metrics([], [])["clean_accuracy"]["value"] is None


def test_spell_vocabulary_contains_no_test_only_words():
    norm = Normalizer(TextNormalizationDefense(), DS.train_texts)
    pre = norm._pre
    train_words = {w for t in DS.train_texts for w in re.findall(r"[a-z']+", pre(t))}
    test_only = {w for t in DS.test_texts for w in re.findall(r"[a-z']+", pre(t))} - train_words
    assert test_only and not test_only & set(norm.vocab)
    assert set(norm.vocab) == train_words


def test_adaptive_attack_queries_the_defended_pipeline(engine, monkeypatch):
    """The adaptive attacker's queries pass through the defended system's normaliser; replay sends none."""
    clear_text_caches()
    calls = {"n": 0}
    real = tnorm.Normalizer.__call__

    def counting(self, text):
        calls["n"] += 1
        return real(self, text)
    monkeypatch.setattr(tnorm.Normalizer, "__call__", counting)
    parent = engine.text.run_attack(TextAttackRequest(samples=TextSampleSelection(max_samples=20)))
    assert calls["n"] == 0                                                 # undefended attack: no normaliser
    before = calls["n"]
    ad = engine.text.run_attack(TextAttackRequest(samples=TextSampleSelection(max_samples=20),
                                                  target_defense=TextNormalizationDefense(),
                                                  parent_experiment_id=parent.experiment_id))
    attack_queries = sum(s.queries or 0 for s in ad.samples)
    # every query, the clean batch and the full-test-split evaluation are normalised
    assert calls["n"] - before >= attack_queries + len(ad.samples)
    clear_text_caches()


def test_constraint_validity_is_never_presented_as_semantic_preservation(engine):
    from mlbreak.reporting import SEMANTIC_DISCLAIMER
    rec = engine.text.run_attack(TextAttackRequest(samples=TextSampleSelection(max_samples=10)))
    assert SEMANTIC_DISCLAIMER in engine.report(rec.experiment_id)
    from mlbreak.config import PROJECT_ROOT
    doc = (PROJECT_ROOT / "docs" / "ml01-spam-filter-evasion.md").read_text()
    assert "not proof" in doc.lower() or "NOT proof" in doc
    ui = (PROJECT_ROOT / "frontend" / "src" / "pages" / "SpamWorkspace.tsx").read_text()
    assert "not proof that its meaning is preserved" in ui
