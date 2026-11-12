# ML01:2023 — Spam-filter evasion (character-level adversarial text)

Second scenario of the OWASP **ML01:2023 Input Manipulation** lab, next to the 2-D numerical evasion lab.
Scenario `ml01-spam-filter-evasion`. Code: `src/mlbreak/text/`. Regression cases: `MLB-REG-009` … `013`.
Evidence schema: `mlbreak.text-evasion/v1`. UI: **Attack Lab → ML01 scenario → Spam Filter Evasion**
(`/lab/ml01/spam/…`).

> **Intentionally vulnerable.** The undefended spam filter is deliberately exposed to evasion, on bundled public
> data, for education and defensive research. Run it locally.

## 1. Reference analysis: SpamShieldAI `red.ipynb`

The scenario was designed from SpamShieldAI's red-team notebook
([`app/notebooks/red.ipynb` @ `bc9c4e5`](https://github.com/urcuqui/SpamShieldAI/blob/bc9c4e56ebc10f6046b9be03bbfe0a70c247cbc2/app/notebooks/red.ipynb)).
It was read as data: no code was executed and no artifact was copied.

| Aspect | Reference notebook |
|---|---|
| Model and input | Pretrained BERT `Goodmotion/spam-mail-classifier`, **downloaded from Hugging Face at runtime**; tokenizer → logits → softmax; class 1 = spam |
| Attack | TextAttack `DeepWordBugGao2018`: `GreedyWordSwapWIR` (word-importance ranking with `wir_method="unk"`), untargeted classification goal, black-box (scores) |
| Transformations | one random character per word: neighbouring swap, substitution, deletion, insertion |
| Constraints | Levenshtein ≤ **30** against the original, `RepeatModification` (a word is modified once), `StopwordModification` |
| Data and labels | **10 hand-written messages**, no train/test split; labels assigned by hand (some debatable, e.g. "Important security update for your account" = spam); one French message, out of distribution |
| Recorded outcome | 3 succeeded / 4 failed / 3 skipped (clean misclassifications); accuracy 70% → 40%; success 3/7 = 42.86%; **39.7% of words perturbed** on average; 19.4 queries on average |
| Example | "Congratulations! You've won a free gift card." → "Congratualtions! Yu've won a rfee gqift acrd." (5 of 7 words modified) |

Limitations of the reference that this scenario addresses:

* **Not reproducible offline.** The model is fetched from the Hub at runtime and nothing pins its revision.
  MLBreak trains a bundled traditional model in-process.
* **Ten examples are not a measurement.** The notebook's 42.86% has a 95% confidence interval of roughly
  10–82% (exact Clopper–Pearson: 9.9–81.6%). MLBreak attacks up to 300 held-out messages and reports every rate with its numerator and denominator.
  **The notebook's numbers are illustrative only and are never used as a threshold.**
* **Loose constraints.** 30 edits on a 7-word message allow rewriting most of it. MLBreak adds a character
  ratio, a modified-word ratio and protected tokens, and defaults to ≤ 10 edits.
* **A measurement error in the notebook.** Cell 7 probes `"[[Congratualtions]]! [[Yu've]] …"`, which includes
  TextAttack's `[[ ]]` *display markup*, and reports 74% ham for a string the attack never produced (the attack
  log says 98%). MLBreak stores the exact adversarial string and its edit script and re-validates both.

## 2. Threat model

| | |
|---|---|
| **Asset** | Integrity of the spam filter's decision on incoming messages |
| **Trust boundary** | Message sender → classifier (the message text is attacker-controlled) |
| **Attacker goal** | `spam_to_ham` (default): get a spam message delivered as ham. `ham_to_spam`: make a legitimate message flagged (false positive). The target-class probability must reach `min_target_confidence` (default 0.5) |
| **Knowledge** | Black-box and score-based: the attacker observes P(spam) of the **deployed system**, nothing else (no parameters, gradients, vocabulary or training data) |
| **Access** | ≤ `query_budget` queries per message (default 200), counted by an oracle that refuses any query beyond the budget |
| **Can modify** | Characters of non-protected words of the message at inference time, with the permitted operations and characters |
| **Cannot** | Modify training data, the model, the normaliser or the decision threshold; edit digits, phone numbers, short codes, URLs or e-mail addresses (`protect_numbers`); edit function words (`protect_stopwords`) |
| **Adaptive attacker** | Knows the defense and queries the *defended* pipeline, so normalisation is inside its search loop. It may spend more effort (edits per word, query budget) but never relaxes the text-preservation constraints of its parent experiment |

**Attack surface.** Any character the sender controls. The scenario covers the four DeepWordBug edits with
ASCII letters, plus two opt-in charsets that attackers use against real filters: **confusable homoglyphs**
(Cyrillic `е` for Latin `e`, …) and **invisible zero-width characters** (U+200B/U+200C/U+200D/U+2060). The
latter leave the *rendered* message pixel-identical while splitting the tokens the filter sees.

## 3. Data and classifier

* **Dataset:** SMS Spam Collection v.1 (Almeida & Gómez Hidalgo, 2011; UCI, DOI 10.24432/C5CC84; **CC BY 4.0**),
  bundled byte-for-byte under `src/mlbreak/text/data/sms_spam_collection/` with a NOTICE. The loader verifies its
  SHA-256 and refuses a modified file. Nothing is downloaded.
* **De-duplication before splitting:** 415 of the 5,574 messages are duplicates (case- and whitespace-
  insensitive). They are removed **before** the split, so no test message also appears in training.
* **Split:** stratified 80/20 with `random_state = seed`: 4,127 training and 1,032 test messages (128 test spam
  at seed 7).
* **Representation:** `TfidfVectorizer(analyzer="word", lowercase=True, ngram_range=(1, 2), min_df=2,
  sublinear_tf=True, token_pattern=r"(?u)\b\w\w+\b", norm="l2")`.
* **Classifiers (approved configurations, trained in-process, never unpickled):**
  `tfidf-word-logreg` (default; `LogisticRegression(C=10, class_weight="balanced", solver="lbfgs")`) and
  `tfidf-word-nb` (`MultinomialNB(alpha=0.1)`) for comparison.
* **Artifact hash:** SHA-256 over the vocabulary, IDF weights, classifier parameters and normaliser.

Clean performance at seed 7: logistic regression 98.6% accuracy, spam recall 119/128, ham false-positive rate
5/904. Naive Bayes is slightly less accurate on spam.

## 4. The attack (DeepWordBug-style)

Reference: Gao, Lanchantin, Soffa & Qi (2018), *Black-box Generation of Adversarial Text Sequences to Evade
Deep Learning Classifiers*, as packaged by TextAttack's `DeepWordBugGao2018`.

1. Query the clean message. If it is already misclassified, it is **skipped** (nothing to evade).
2. **Word importance:** for every modifiable word, the change in target-class probability when that word is
   removed (Gao et al.'s *Replace-1* score; for a bag-of-words model, removal is equivalent to an unknown token).
3. **Greedy search:** visit words in importance order. Draw `candidates_per_word` single-character edits
   (swap, substitute, delete, insert) from **all** permitted variants of the word with a seeded RNG, query them,
   and keep the best if it raises the target-class probability. Stop at the first success.
4. Re-validate the final message against every constraint.

**Not the original algorithm.** Only the Replace-1 score is implemented (no Temporal Head/Tail or Combined
scores); candidates are sampled from all one-edit variants; constraints are much tighter than the TextAttack
recipe; and the invisible/confusable charsets are extensions. Results therefore are not comparable
number-for-number with the paper or the notebook.

### Constraints (enforced during the search and re-validated on the final text)

| Constraint | Default | Meaning |
|---|---|---|
| `max_edit_distance` | 10 | Levenshtein distance (code points) between original and adversarial message |
| `max_char_ratio` | 0.15 | edit distance / original length |
| `max_modified_word_ratio` | 0.5 | modified words / modifiable words |
| `max_edits_per_word` | 1 | DeepWordBug's one edit per word (an adaptive attacker may use 2–3) |
| `operations`, `charsets` | all four; ASCII letters | the only permitted edits and characters |
| `protect_numbers` | on | digits, phone numbers, short codes, URLs, e-mails are never touched |
| `protect_stopwords` | on | an explicit list of English function words (scikit-learn's `ENGLISH_STOP_WORDS` is **not** used: it contains "call", "now", "get", "bill", the words of a spam call-to-action) |
| valid Unicode | always | no control, surrogate, private-use or unassigned characters; format characters only through the `invisible` charset |

**A label change alone is not an evasion.** An outcome is `evaded` only if the target class is reached **and**
every constraint holds. If the search ever returned a success that violates a constraint, the experiment is
recorded as **failed** (`TextConstraintViolation`), never as an evasion. `mlbreak check` re-validates every
stored adversarial message from its original text and edit script.

During the search, the edit-distance constraint uses the sum of per-word Levenshtein distances. That sum is a
sound upper bound on the full-text distance, because words are disjoint and each is edited a bounded number of
times. The exact distance is recorded in the evidence.

## 5. Defenses and evaluation

| Defense | What it does | Leakage control |
|---|---|---|
| **Text normalisation** | `unicode`: NFKC, strip format characters, map confusables; `leetspeak`: `fr3e`→`free`; `spell_snap`: replace a token absent from the training vocabulary with the most frequent training word at edit distance 1 | Applied to training and inference alike; the spell vocabulary comes from the **training split only** |
| **Adversarial data augmentation** | Attack the first `max_train_samples` **training** spam messages with the undefended model; add every successful adversarial message (label spam); retrain | Test messages are never attacked or trained on (tested) |

Each defense is evaluated two ways, on the **same held-out messages**:

* **Replay** (non-adaptive): the exact adversarial messages of the parent attack, re-submitted unchanged.
* **Adaptive attack:** a **new** experiment that searches against the defended pipeline, with the same messages,
  goal and text-preservation constraints (verified by lineage), optionally with more effort.

Defense claims are tested with **paired statistics** on the same messages (`src/mlbreak/stats.py`):
absolute reduction and an exact one-sided McNemar test.

### Measured results (seed 7, 100 held-out spam messages, 93 attackable; Linux and macOS identical)

| Evaluation | Logistic regression | Naive Bayes |
|---|---|---|
| Attack, default constraints | **25/93 = 26.9%** | 17/86 = 19.8% |
| … with number protection **off** (the phone number may be corrupted) | 53/93 = 57.0% | 34/86 = 39.5% |
| … with ≤ 5 edits / ≤ 10% of characters | 15/93 = 16.1% | 10/86 = 11.6% |
| Normalisation: replay → adaptive (default effort) | 4/93 → **9/93** | 3/86 → **14/86** (p = 0.25, not significant) |
| Normalisation: adaptive with 2 edits/word, 400 queries (undefended 27/93) | **16/93** (p = 0.004) | 17/86 (p = 0.36) |
| Augmentation: replay → adaptive (2 edits/word, 400 queries) | 5/91 → **14/91** (p = 0.002) | 14/86 → 18/87 |
| Invisible-character insertion: undefended → Unicode normalisation (replay and adaptive) | **41/93 → 0/93** | 28/86 → 0/86 |
| Confusable homoglyphs: undefended → Unicode normalisation | 17/93 → 0/93 | 14/86 → 0/86 |

Lessons the scenario is built to show:

1. **Constraints decide the headline number.** Allowing the attack to corrupt phone numbers doubles its
   "success", but such a message no longer works as spam.
2. **Replay overstates defenses.** Normalisation replay leaves 4/93; a new attack through the normalising
   pipeline gets 16/93, using edits that land on *other* real words (`msg`→`msn`, `top`→`lop`), which spell-snap
   cannot undo.
3. **A defense does not transfer between models.** The same normalisation significantly reduces adaptive success
   for logistic regression but not for naive Bayes (MLB-REG-013).
4. **Narrow defenses can be complete within their scope.** Unicode normalisation neutralises invisible and
   homoglyph attacks entirely, because those charsets only produce characters it removes, and says nothing about
   ASCII typos.

## 6. Evidence

Each record (`mlbreak.text-evasion/v1`) stores the scenario, OWASP category and version, model configuration and
artifact hash, TF-IDF configuration, dataset provenance and split hashes, attack and defense configuration,
attacker capabilities, and per message: original and adversarial text, the exact edit script, predictions and
P(spam) before and after, queries, edit distance, character and word ratios, per-constraint validation and the
outcome. It also stores the clean predictions on the full test split, metrics with explicit denominators,
lineage (parent hashes) and reproducibility metadata (exact dependency versions, BLAS backend and environment
fingerprint). JSON, Markdown and per-message CSV exports are available. Integrity checks re-derive every metric
and re-validate every constraint; replays require them to pass.

## 7. Limitations

* **Traditional ML only.** Results for TF-IDF models do not carry over to transformers such as the reference
  BERT model, which tokenizes into subwords and reacts very differently to character edits.
* **Proxy for meaning preservation.** Edit distance, ratios and protected tokens are proxies. Nothing checks that
  a human reads the adversarial message the same way, or that it still persuades. Some successful edits are
  awkward (`Vree`).
* **Greedy, sampled search.** The attack is a lower bound on what a stronger attacker achieves: no beam search,
  no character-embedding guidance, and `candidates_per_word` samples of the one-edit neighbourhood.
* **Adaptive attacker scope.** It knows the defense and queries the defended pipeline, but does not exploit
  specific weaknesses beyond more effort (for example, deliberately choosing edits that spell-snap maps to
  benign words).
* **Old, small, English, SMS-only data** (2011, UK and Singapore). Modern spam, other languages and e-mail differ.
  De-duplication is exact-match only; near-duplicates (templated spam) can still straddle the split.
* **Single seed and split in the regression cases.** The effects are paired and significant where claimed, but
  the margins are a few messages; they were identical on macOS/Accelerate and Linux/OpenBLAS.

## 8. Reproduce

```bash
mlbreak demo --lab ml01-spam                         # classifier → attack → replay/adaptive per defense → invisible chars → replay
mlbreak text classify "WINNER! Claim your prize, call 09061701461"
mlbreak text attack --samples 100                    # default constraints
mlbreak text defend <ATTACK_ID> --defense text_normalization
mlbreak text attack --defense text_normalization --parent <ATTACK_ID> --edits-per-word 2 --queries 400   # adaptive
mlbreak text attack --operations insert --charsets invisible
mlbreak check <ID>        # integrity + constraint re-validation
mlbreak replay <ID>       # refused if integrity fails; identical within the same numerical environment
mlbreak verify            # includes MLB-REG-009 … 013
```
