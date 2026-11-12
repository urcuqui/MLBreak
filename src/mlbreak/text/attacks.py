"""DeepWordBug-style character-level evasion of a text classifier (OWASP ML01:2023).

Reference: Gao, Lanchantin, Soffa & Qi (2018), *Black-box Generation of Adversarial Text Sequences to Evade
Deep Learning Classifiers* (DeepWordBug), as packaged by TextAttack's ``DeepWordBugGao2018`` recipe.

What is reproduced:

* black-box, score-based: only the deployed system's class probabilities are used (``Oracle``);
* word importance by the change in the source-class score when a word is removed — Gao et al.'s
  "Replace-1" score with deletion, which for a bag-of-words TF-IDF model is equivalent to an unknown token
  (TextAttack's ``wir_method="unk"``);
* the four character transformations: adjacent swap, substitution, deletion, insertion;
* greedy search in importance order, one edit per word by default (TextAttack's RepeatModification) and no
  stopword edits (StopwordModification), stopping at the first success;
* a Levenshtein constraint against the ORIGINAL text.

Differences (this is "DeepWordBug-style", not the original algorithm):

* the original paper scores with several functions (Replace-1, Temporal Head/Tail, Combined) computed on
  RNN/CNN models; only Replace-1 is implemented;
* candidate edits per word are a seeded random sample of ``candidates_per_word`` from ALL one-edit variants,
  and the best is kept only if it raises the target-class probability (TextAttack keeps the best of its
  transformations; the original applies one random edit to the top-m words);
* constraints are much tighter than the TextAttack recipe's ``max_edit_distance=30`` and add a character
  ratio, a modified-word ratio and protected tokens (digits, URLs, e-mails), because a 30-edit budget on a
  7-word SMS (≈ 40% of words in the SpamShieldAI notebook) no longer preserves the message;
* ``invisible`` / ``confusables`` charsets are opt-in extensions used to study the normalisation defense.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..attacks import ExperimentTimeout, QueryBudgetExceeded, _check_deadline
from .normalize import ASCII_LETTERS, ASCII_TO_CONFUSABLE, CONFUSABLES, INVISIBLE
from .schemas import TextAttackConfig, TextEdit

WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*")
PROTECTED_SPAN_RE = re.compile(
    r"(?i)\bhttps?://\S+|\bwww\.\S+|\S+@\S+\.\S+|\b[\w-]+\.(?:com|co\.uk|net|org|biz|info|uk|tv|mobi)\b\S*")
# Function words that are never modified (TextAttack's StopwordModification). An explicit list close to NLTK's
# English stopwords, deliberately WITHOUT content words: scikit-learn's ENGLISH_STOP_WORDS contains "call",
# "now", "get", "find", "bill", "amount" — the very words of a spam call-to-action — so it is not used.
STOPWORDS = frozenset("""
a an the and or but if then else so as of at by for with about against between into through during before after
above below to from up down in out on off over under again further once here there when where why how all any
both each few more most other some such no nor not only own same than too very can will just should would could
shall may might must i me my myself we our ours ourselves you your yours yourself yourselves he him his himself
she her hers herself it its itself they them their theirs themselves what which who whom whose this that these
those am is are was were be been being have has had having do does did doing s t ll re ve d m o y don isn aren
wasn weren hasn haven hadn doesn didn won wouldn shouldn couldn mustn needn shan
""".split())

__all__ = ["Oracle", "Segmented", "segment", "levenshtein", "deepwordbug_attack", "validate_constraints",
           "apply_edits", "TextConstraintViolation", "AttackResult", "ExperimentTimeout", "QueryBudgetExceeded"]


class TextConstraintViolation(RuntimeError):
    """A produced adversarial text violates its declared constraints (never silently accepted)."""


class Oracle:
    """Counts model queries and enforces the per-sample budget. Every model call of the attack goes through
    here; a call that would exceed the budget raises :class:`QueryBudgetExceeded` instead of running."""

    def __init__(self, p_spam: Callable[[Sequence[str]], np.ndarray], budget: int):
        self._p_spam, self.budget, self.used = p_spam, budget, 0

    @property
    def remaining(self) -> int:
        return self.budget - self.used

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        if self.used + len(texts) > self.budget:
            raise QueryBudgetExceeded(f"{self.used} + {len(texts)} queries exceed the budget of {self.budget}")
        self.used += len(texts)
        return self._p_spam(texts)


def levenshtein(a: str, b: str) -> int:
    """Code-point Levenshtein distance (insert / delete / substitute = 1; an adjacent swap counts 2)."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass
class Segmented:
    segments: list[str]  # words and separators, "".join(segments) == text
    word_indices: list[int]  # segment index of every word
    modifiable: list[int]  # segment indices the attacker may modify


def segment(text: str, cfg: TextAttackConfig) -> Segmented:
    protected_spans = [m.span() for m in PROTECTED_SPAN_RE.finditer(text)] if cfg.protect_numbers else []
    segs, words, modifiable, pos = [], [], [], 0
    for m in WORD_RE.finditer(text):
        if m.start() > pos:
            segs.append(text[pos:m.start()])
        w = m.group(0)
        segs.append(w)
        idx = len(segs) - 1
        words.append(idx)
        in_span = any(s <= m.start() < e for s, e in protected_spans)
        ok = (len(w) >= cfg.min_word_length and not in_span
              and not (cfg.protect_numbers and any(ch.isdigit() for ch in w))
              and not (cfg.protect_stopwords and w.lower() in STOPWORDS))
        if ok:
            modifiable.append(idx)
        pos = m.end()
    if pos < len(text):
        segs.append(text[pos:])
    return Segmented(segs, words, modifiable)


# --------------------------------------------------------------------------
# Candidate edits
# --------------------------------------------------------------------------
def _sub_chars(ch: str, charsets: set[str]) -> list[str]:
    out = []
    if "ascii_letters" in charsets and ch.isalpha():
        letters = ASCII_LETTERS.upper() if ch.isupper() else ASCII_LETTERS
        out += [c for c in letters if c != ch]
    if "confusables" in charsets:
        out += ASCII_TO_CONFUSABLE.get(ch, [])
    return out


def _ins_chars(prev: str, charsets: set[str]) -> list[str]:
    out = []
    if "ascii_letters" in charsets:
        out += list(ASCII_LETTERS.upper() if prev.isupper() else ASCII_LETTERS)
    if "invisible" in charsets:
        out += list(INVISIBLE)
    return out


def all_edits(word: str, cfg: TextAttackConfig) -> list[tuple[str, int, str | None, str | None, str]]:
    """Every single-edit variant: (op, position, char_before, char_after, new_word). Deterministic order."""
    ops, charsets = set(cfg.operations), set(cfg.charsets)
    out, n = [], len(word)
    if "swap" in ops:
        for i in range(n - 1):
            if word[i] != word[i + 1]:
                out.append(("swap", i, word[i:i + 2], word[i + 1] + word[i], word[:i] + word[i + 1] + word[i] + word[i + 2:]))
    if "substitute" in ops:
        for i, ch in enumerate(word):
            for c in _sub_chars(ch, charsets):
                out.append(("substitute", i, ch, c, word[:i] + c + word[i + 1:]))
    if "delete" in ops and n > 1:
        for i in range(n):
            out.append(("delete", i, word[i], None, word[:i] + word[i + 1:]))
    if "insert" in ops:
        for i in range(n + 1):
            for c in _ins_chars(word[max(i - 1, 0)] if word else "a", charsets):
                out.append(("insert", i, None, c, word[:i] + c + word[i:]))
    seen, uniq = set(), []
    for e in out:  # several edits can yield the same word (e.g. deleting either of two equal letters)
        if e[4] != word and e[4] not in seen:
            seen.add(e[4])
            uniq.append(e)
    return uniq


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------
@dataclass
class AttackResult:
    outcome: str  # evaded | failed | budget_exhausted | skipped_misclassified
    text: str
    p_spam: float
    queries: int
    edits: list[TextEdit] = field(default_factory=list)
    words_modifiable: int = 0
    words_modified: int = 0


def _target_prob(p_spam: float, target: int) -> float:
    return p_spam if target == 1 else 1.0 - p_spam


def deepwordbug_attack(text: str, y_true: int, p_spam_fn: Callable[[Sequence[str]], np.ndarray],
                       cfg: TextAttackConfig, rng: np.random.Generator,
                       deadline: float | None = None) -> AttackResult:
    oracle = Oracle(p_spam_fn, cfg.query_budget)
    tgt = cfg.target_class
    p0 = float(oracle([text])[0])
    if int(p0 >= 0.5) != y_true:  # clean input already misclassified → nothing to evade (TextAttack: SKIPPED)
        return AttackResult("skipped_misclassified", text, p0, oracle.used)
    seg = segment(text, cfg)
    cur = list(seg.segments)
    n_mod = len(seg.modifiable)
    max_words = int(np.floor(cfg.max_modified_word_ratio * n_mod + 1e-9))
    max_ed = min(cfg.max_edit_distance, int(np.floor(cfg.max_char_ratio * len(text) + 1e-9)))
    score = _target_prob(p0, tgt)
    result = AttackResult("failed", text, p0, 0, [], n_mod, 0)
    if not seg.modifiable or max_ed < 1 or max_words < 1:
        result.queries = oracle.used
        return result

    # 1) word importance: drop in source-class score when the word is removed
    probes = ["".join(cur[:i] + cur[i + 1:]) for i in seg.modifiable][:oracle.remaining]
    if not probes:
        result.outcome, result.queries = "budget_exhausted", oracle.used
        return result
    p_probe = oracle(probes)
    importance = np.full(n_mod, -np.inf)
    importance[:len(probes)] = _target_prob(p_probe, tgt) - score  # larger = removing it helps the attacker more
    order = [seg.modifiable[k] for k in np.argsort(-importance, kind="stable")]

    # 2) greedy edits in importance order
    edits_per_word: dict[int, int] = {}
    word_ed: dict[int, int] = {}  # per-word Levenshtein to the original word; their sum bounds the text distance
    for _pass in range(cfg.max_edits_per_word):
        for i in order:
            _check_deadline(deadline)
            if edits_per_word.get(i, 0) >= cfg.max_edits_per_word:
                continue
            if i not in edits_per_word and len(edits_per_word) >= max_words:
                continue
            variants = all_edits(cur[i], cfg)
            if not variants:
                continue
            k = min(cfg.candidates_per_word, len(variants))
            sample = [variants[j] for j in sorted(rng.choice(len(variants), size=k, replace=False))]
            base_total = sum(word_ed.values()) - word_ed.get(i, 0)
            sample = [v for v in sample if base_total + levenshtein(seg.segments[i], v[4]) <= max_ed]
            if not sample:
                continue
            if oracle.remaining <= 0:
                result.outcome = "budget_exhausted"
                break
            sample = sample[:oracle.remaining]
            texts = ["".join(cur[:i] + [v[4]] + cur[i + 1:]) for v in sample]
            p = oracle(texts)
            t_scores = _target_prob(p, tgt)
            j = int(np.argmax(t_scores))
            if t_scores[j] > score:
                op, pos, before, after, new_word = sample[j]
                result.edits.append(TextEdit(word_index=i, op=op, position=pos, char_before=before,
                                             char_after=after, word_before=cur[i], word_after=new_word))
                cur[i] = new_word
                score = float(t_scores[j])
                result.p_spam = float(p[j])
                edits_per_word[i] = edits_per_word.get(i, 0) + 1
                word_ed[i] = levenshtein(seg.segments[i], new_word)
                if score >= cfg.min_target_confidence:
                    result.outcome = "evaded"
                    break
            if oracle.remaining <= 0 and result.outcome != "evaded":
                result.outcome = "budget_exhausted"
                break
        if result.outcome in ("evaded", "budget_exhausted"):
            break
    result.text = "".join(cur)
    result.queries = oracle.used
    result.words_modified = len(edits_per_word)
    return result


# --------------------------------------------------------------------------
# Constraint validation (re-run on the final text, and by the integrity check)
# --------------------------------------------------------------------------
def apply_edits(text: str, cfg: TextAttackConfig, edits: Sequence[TextEdit]) -> str:
    """Rebuild the adversarial text from the original and the recorded edit script."""
    segs = segment(text, cfg).segments
    for e in edits:
        if segs[e.word_index] != e.word_before:
            raise TextConstraintViolation(f"edit script does not apply at word {e.word_index}")
        segs[e.word_index] = e.word_after
    return "".join(segs)


def _allowed_char(op: str, ch: str | None, cfg: TextAttackConfig) -> bool:
    if ch is None:
        return True
    cs = set(cfg.charsets)
    if "ascii_letters" in cs and ch.isascii() and ch.isalpha():
        return True
    if op == "substitute" and "confusables" in cs and ch in CONFUSABLES:
        return True
    if op == "insert" and "invisible" in cs and ch in INVISIBLE:
        return True
    return op in ("swap",)  # swaps only move existing characters


def validate_constraints(original: str, adversarial: str, cfg: TextAttackConfig,
                         edits: Sequence[TextEdit]) -> tuple[dict[str, bool], dict[str, float | int]]:
    seg = segment(original, cfg)
    ed = levenshtein(original, adversarial)
    per_word: dict[int, int] = {}
    for e in edits:
        per_word[e.word_index] = per_word.get(e.word_index, 0) + 1
    n_mod = len(seg.modifiable)
    try:
        rebuilt = apply_edits(original, cfg, edits)
    except TextConstraintViolation:
        rebuilt = None
    introduced = set(adversarial) - set(original)
    checks = {
        "edit_distance": ed <= cfg.max_edit_distance,
        "char_ratio": ed <= cfg.max_char_ratio * len(original) + 1e-9,
        "modified_word_ratio": len(per_word) <= cfg.max_modified_word_ratio * n_mod + 1e-9,
        "edits_per_word": all(v <= cfg.max_edits_per_word for v in per_word.values()),
        "operations_allowed": all(e.op in cfg.operations for e in edits),
        "charset_allowed": all(_allowed_char(e.op, e.char_after, cfg) for e in edits),
        "only_modifiable_words": all(e.word_index in seg.modifiable for e in edits),
        "edit_script_reproduces_text": rebuilt == adversarial,
        "valid_unicode": _valid_unicode(adversarial, introduced, cfg),
    }
    stats = {"edit_distance": ed, "char_ratio": ed / max(len(original), 1),
             "words_modifiable": n_mod, "words_modified": len(per_word)}
    return checks, stats


def _valid_unicode(text: str, introduced: set[str], cfg: TextAttackConfig) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:  # lone surrogates
        return False
    for ch in introduced:
        cat = unicodedata.category(ch)
        if cat in ("Cc", "Cs", "Co", "Cn"):
            return False
        if cat == "Cf" and not ("invisible" in cfg.charsets and ch in INVISIBLE):
            return False
    return True
