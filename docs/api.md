# HTTP API

Interactive OpenAPI documentation is served at **http://127.0.0.1:8000/docs** (ReDoc: `/redoc`) when the backend
runs; routes are grouped by tag. This page summarises the contract and gives examples.

## Conventions

* Every request body is a strict schema: **unknown fields are rejected** (422). Numeric limits are enforced by the
  schema (`GET /api/limits`).
* Experiment endpoints return the complete **evidence record** that was stored (`schema_version` tells the
  family: `mlbreak.experiment/v2`, `mlbreak.text-evasion/v1`, `mlbreak.poisoning/v1`).
* Experiment ids match `MLB-EXP-<12 hex>`; anything else is a 404 (this also prevents path traversal).

| Status | Meaning |
|---|---|
| 404 | unknown experiment id |
| 409 | the stored evidence **failed its integrity check**; nothing was executed (body: `integrity` report) |
| 422 | invalid request, or a valid request that cannot run (e.g. an incompatible parent experiment) |
| 429 | concurrency gate full; nothing was executed — retry (the web UI queues and retries automatically) |
| 504 | wall-clock limit exceeded for non-experiment work; experiments record `status: failed` instead |

## Endpoints by tag

| Tag | Endpoints |
|---|---|
| `catalogue` | `GET /api/health`, `/api/owasp`, `/api/scenarios`, `/api/limits` |
| `ml01-2d` | `POST /api/experiments/baseline`, `/attack`, `/defend`; `POST /api/boundary`; `GET /api/scenarios/ml01-moons-evasion/dataset` |
| `ml01-spam` | `POST /api/experiments/text/baseline`, `/text/attack`, `/text/defend`; `POST /api/text/classify`; `GET /api/text/messages` |
| `ml02` | `POST /api/experiments/poisoning/baseline`, `/poisoning/attack`, `/poisoning/defend`; `GET /api/experiments/{id}/boundary` |
| `evidence` | `GET /api/experiments`, `/api/experiments/{id}`, `/{id}/verify`, `/{id}/evidence.json`, `/{id}/report.md`, `/{id}/samples.csv`; `POST /api/experiments/{id}/replay` |
| `regression` | `GET /api/regression/cases`, `/api/regression/latest`; `POST /api/regression/run` |
| `ui` | `GET /api/dashboard` |

Adaptive attacks use the `attack` endpoint of their lab with `target_defense` and `parent_experiment_id`;
non-adaptive replay uses the lab's `defend` endpoint. ML02 has no adaptive evaluation.

## Examples

```bash
B=http://127.0.0.1:8000
# ML01 spam: attack 50 held-out spam messages, replay them against normalisation, then an adaptive attack
A=$(curl -s -X POST $B/api/experiments/text/attack -H 'Content-Type: application/json' \
      -d '{"samples": {"max_samples": 50}}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["experiment_id"])')
curl -s -X POST $B/api/experiments/text/defend -H 'Content-Type: application/json' \
     -d "{\"parent_experiment_id\": \"$A\", \"defense\": {\"kind\": \"text_normalization\"}}" > /dev/null
curl -s -X POST $B/api/experiments/text/attack -H 'Content-Type: application/json' \
     -d "{\"samples\": {\"max_samples\": 50}, \"parent_experiment_id\": \"$A\",
          \"target_defense\": {\"kind\": \"text_normalization\"}, \"attack\": {\"max_edits_per_word\": 2, \"query_budget\": 400}}" > /dev/null

curl -s $B/api/experiments/$A/verify            # integrity (+ lineage) without re-running
curl -s -X POST $B/api/experiments/$A/replay    # 409 if integrity fails; otherwise compares fingerprints
curl -s $B/api/experiments/$A/report.md         # seven-section report, rendered with a fresh integrity check

# ML01 2-D: PGD against the undefended model
curl -s -X POST $B/api/experiments/attack -H 'Content-Type: application/json' -d '{"attack": {"attack": "pgd", "epsilon": 0.3}}'
# ML02: targeted label flipping
curl -s -X POST $B/api/experiments/poisoning/attack -H 'Content-Type: application/json' \
     -d '{"poison": {"strategy": "targeted", "fraction": 0.1, "target_class": 1, "selection": "boundary"}}'
```
