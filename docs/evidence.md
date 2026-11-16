# Evidence across laboratories

All three scenarios produce evidence the same way, while keeping their own metric definitions.

| Scenario | Record schema | Per-sample evidence | Specialised metrics |
|---|---|---|---|
| ML01 · 2D Classifier Evasion | `mlbreak.experiment/v2` | perturbed inputs, predictions, ‖δ‖, outcome partition | attack success, adversarial / robust accuracy, abstention |
| ML01 · Spam Filter Evasion | `mlbreak.text-evasion/v1` | original/adversarial text, edit script, queries, constraint checks | attack success within constraints, accuracy under attack, query effort |
| ML02 · Label-Flipping Poisoning | `mlbreak.poisoning/v1` | training rows (true/observed/sanitised labels, flags), test rows | attack success, target-class recall, detection precision/recall, recovery |

Every ratio carries numerator, denominator, population and definition; an empty denominator is `null`, never 0.

## One report structure

Every Markdown report (`GET /api/experiments/{id}/report.md`, `mlbreak export <id> --format md`, the **MD** link in
each lab's *Evidence & replay* step) has the same seven sections:

1. **Summary** — identity, status, OWASP category and scenario
2. **Threat model** — the scenario's asset, trust boundary, attacker goal and what is out of scope, plus the
   attacker capabilities of this experiment
3. **Experiment configuration** — model and artifact hashes, data and split hashes, attack, defense, lineage
4. **Observed outcomes** — the family's metrics and their definitions, per-sample results
5. **Defense limitations** — what the result does *not* show (replay is not robustness, adaptive success is a
   lower bound, the known bypasses pinned by regression cases, and for text: constraint-valid ≠ meaning-preserving)
6. **Reproducibility** — fingerprints, numerical environment, replay command
7. **Integrity verification** — the result of the integrity + lineage check **at the moment the report is
   rendered**

Reports are rendered from the stored JSON when requested, so a record edited after the fact is reported as
**FAIL** rather than shown with a stale clean report. (A write-time snapshot is still saved next to the JSON.)

## Verify and replay

| Action | UI | CLI / API | Meaning |
|---|---|---|---|
| Verify | *Verify* in each lab's Evidence step and on every experiment page | `mlbreak check <id>`, `GET …/verify` | fingerprints recomputed, metrics re-derived from per-sample evidence, family-specific checks (e.g. text constraints re-validated, poisoning label vectors), parent lineage |
| Replay | *Replay* in the same places | `mlbreak replay <id>`, `POST …/replay` | re-executes the experiment; refused (HTTP 409) if verification fails; compares every fingerprint |

Integrity is **not authenticity**: see [reproducibility.md](reproducibility.md#integrity-is-not-authenticity).
