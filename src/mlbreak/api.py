"""FastAPI application: JSON API under ``/api`` and the built React UI at ``/``.

Security posture: local-first (binds 127.0.0.1 by default), no authentication
because there are no credentials or secrets, no upload endpoints, no endpoint
that accepts code, file paths or serialised models. Every request body is a
strict, bounded Pydantic model.
"""

from __future__ import annotations

import hmac
import json
import os
import re
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import LIMITS, PROJECT_ROOT, cases_dir, data_dir
from .attacks import ExperimentTimeout
from .evidence import EvidenceStore, ExperimentNotFound
from .experiments import SCENARIOS, EngineBusy, EvidenceIntegrityError, ExperimentEngine, InvalidExperiment
from .models import MODELS
from .owasp import CATEGORIES, TAXONOMY_NAME, TAXONOMY_URL, TAXONOMY_VERSION
from .regression import load_cases, run_suite
from .inversion.schemas import (InversionAttackRequest, InversionBaselineRequest, InversionDefendRequest,
                                InversionTargetsRequest)
from .inversion.sensitivity import SensitivityReport, SensitivityRequest
from .inversion.sessions import (AttackerQueryResponse, AttackerSessionView, AttackerSubmitReceipt, SessionCreateRequest,
                                 SessionNotFound, SessionNotSubmitted, SessionQueryRequest, SessionRefused, SessionScore,
                                 SessionStore, SessionSubmitRequest, TargetsPreview)
from .membership.schemas import MembershipAttackRequest, MembershipBaselineRequest, MembershipDefendRequest
from .membership.study import StudyReport, StudyRequest
from .poisoning.schemas import PoisonAttackRequest, PoisonBaselineRequest, PoisonDefendRequest
from .text.schemas import (ModelId as TextModelId, TextAttackRequest, TextBaselineRequest, TextClassifyRequest,
                           TextDefendRequest)
from .schemas import AttackRequest, BaselineRequest, BoundaryRequest, DefendRequest

FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

_CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; "
        "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")


API_DESCRIPTION = """Adversarial ML security lab (OWASP Machine Learning Security Top 10, 2023). **Intentionally
vulnerable — run locally.**

Every request body is a strict schema (unknown fields are rejected). Error contract:

| Status | Meaning |
|---|---|
| 401 | evaluator-token mode only: an evaluator endpoint was called without `Authorization: Bearer <token>` |
| 403 | ML03 attack session refused: budget exhausted, target out of scope, session closed or session limit (body: `error` code) |
| 404 | unknown experiment id (ids must match `MLB-EXP-<12 hex>`) or ML03 session id (`MLB-SES-<32 hex>`) |
| 409 | the stored evidence failed its integrity check; nothing was executed (body: `integrity` report); or an ML03 session was scored before submission |
| 422 | invalid request, or a valid request that cannot run (e.g. an incompatible parent experiment) |
| 429 | the concurrency gate is full; nothing was executed — retry |
| 504 | the wall-clock limit was exceeded for non-experiment work (experiments record a failed status instead) |

Experiment endpoints return the full evidence record (evaluator-facing; ML04 records include the evaluator-only
membership ground truth, so ML04 has no attacker-facing endpoint). The ML03 **attacker-facing** surface is the
`ml03-attacker` tag only: a session view, budgeted candidate queries and one submission, with response models that
contain no ground truth, labels, model parameters, training records or evaluation metrics. With
`MLBREAK_EVALUATOR_TOKEN` set, every other `/api/` endpoint requires `Authorization: Bearer <token>`. Reports (`/report.md`) are rendered from the stored JSON
with a fresh integrity check. See `docs/api.md` for examples."""

OPENAPI_TAGS = [
    {"name": "catalogue", "description": "OWASP registry, scenarios, limits, health."},
    {"name": "ml01-2d", "description": "ML01 Input Manipulation — 2-D classifier evasion (baseline, attack, defend)."},
    {"name": "ml01-spam", "description": "ML01 Input Manipulation — spam-filter evasion with character-level text."},
    {"name": "ml02", "description": "ML02 Data Poisoning — label flipping and kNN label sanitisation."},
    {"name": "ml03", "description": "ML03 Model Inversion — EVALUATOR-facing: experiments, target preview, sessions "
                                    "(create/score), sensitivity studies."},
    {"name": "ml03-attacker", "description": "ML03 Model Inversion — ATTACKER-facing: an evaluator-issued session's view, "
                                             "budgeted candidate queries, one submission. Open in evaluator-token mode."},
    {"name": "ml04", "description": "ML04 Membership Inference — EVALUATOR-facing: experiments (records contain the "
                                    "evaluator-only membership ground truth), multi-seed studies, attacker-independence "
                                    "check. There is no attacker-facing ML04 endpoint."},
    {"name": "evidence", "description": "Stored experiments of every lab: list, read, verify, replay, export."},
    {"name": "regression", "description": "Security regression cases and gate."},
    {"name": "ui", "description": "Dashboard data for the web UI."},
]

_TAG_RULES = (  # first matching prefix wins
    ("/api/experiments/inversion/", "ml03"), ("/api/inversion/", "ml03"),
    ("/api/experiments/membership/", "ml04"), ("/api/membership/", "ml04"),
    ("/api/experiments/poisoning/", "ml02"), ("/api/experiments/text/", "ml01-spam"), ("/api/text/", "ml01-spam"),
    ("/api/experiments/{experiment_id}/boundary", "ml02"),
    ("/api/experiments/baseline", "ml01-2d"), ("/api/experiments/attack", "ml01-2d"),
    ("/api/experiments/defend", "ml01-2d"), ("/api/boundary", "ml01-2d"), ("/api/scenarios/{scenario_id}/dataset", "ml01-2d"),
    ("/api/experiments", "evidence"), ("/api/regression", "regression"), ("/api/dashboard", "ui"), ("/api/", "catalogue"),
)


# Attacker-facing routes: open even in evaluator-token mode. Everything else under /api/ is evaluator-facing.
ATTACKER_ROUTES = (
    ("GET", re.compile(r"^/api/(health|owasp|scenarios|limits)$")),
    ("GET", re.compile(r"^/api/inversion/sessions/[^/]+$")),
    ("POST", re.compile(r"^/api/inversion/sessions/[^/]+/(query|submit)$")),
)


def is_attacker_route(method: str, path: str) -> bool:
    return any(method == m and rx.fullmatch(path) for m, rx in ATTACKER_ROUTES)


def _tag_routes(app: FastAPI) -> None:
    from fastapi.routing import APIRoute
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path.startswith("/api/") and not route.tags:
            route.tags = [next(tag for prefix, tag in _TAG_RULES if route.path.startswith(prefix))]


def _summary(report: dict | None) -> dict | None:
    if report is None:
        return None
    return {k: report[k] for k in ("generated_at", "total", "passed", "failed", "ok")}


def create_app(root: Path | None = None, cases: Path | None = None,
               frontend_dist: Path | None = FRONTEND_DIST, evaluator_token: str | None = None) -> FastAPI:
    """``evaluator_token`` (default: ``MLBREAK_EVALUATOR_TOKEN``) switches on evaluator-token mode: evaluator-facing
    endpoints then require ``Authorization: Bearer <token>``; attacker-facing ML03 session endpoints stay open. Without
    a token (the default local research workflow) one local user holds both roles."""
    token = evaluator_token if evaluator_token is not None else os.environ.get("MLBREAK_EVALUATOR_TOKEN") or None
    store = EvidenceStore(root or data_dir())
    sessions = SessionStore(store.root)
    engine = ExperimentEngine(store)
    cases_path = cases or cases_dir()
    app = FastAPI(title="MLBreak", version=__version__, description=API_DESCRIPTION, openapi_tags=OPENAPI_TAGS)
    app.state.store, app.state.engine = store, engine

    @app.middleware("http")
    async def evaluator_gate(request: Request, call_next):
        path = request.url.path
        if token and path.startswith("/api/") and not is_attacker_route(request.method, path):
            given = request.headers.get("authorization", "")
            if not hmac.compare_digest(given.encode(), f"Bearer {token}".encode()):
                return JSONResponse({"detail": "evaluator-facing endpoint: Authorization: Bearer <evaluator token> "
                                               "required", "error": "evaluator_token_required"}, 401)
        return await call_next(request)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        resp = await call_next(request)
        if not request.url.path.startswith(("/docs", "/redoc")):  # Swagger UI loads from a CDN
            resp.headers["Content-Security-Policy"] = _CSP
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["X-Frame-Options"] = "DENY"
        return resp

    @app.exception_handler(ExperimentNotFound)
    async def _nf(_: Request, exc: ExperimentNotFound):
        return JSONResponse({"detail": f"experiment not found: {exc.args[0] if exc.args else ''}"}, 404)

    @app.exception_handler(SessionNotFound)
    async def _snf(_: Request, exc: SessionNotFound):
        return JSONResponse({"detail": "attack session not found", "error": "session_not_found"}, 404)

    @app.exception_handler(SessionRefused)
    async def _sref(_: Request, exc: SessionRefused):
        body = {"detail": exc.message, "error": exc.code}
        if exc.budget is not None:
            body["budget"] = exc.budget
        return JSONResponse(body, 403)

    @app.exception_handler(SessionNotSubmitted)
    async def _snsub(_: Request, exc: SessionNotSubmitted):
        return JSONResponse({"detail": str(exc), "error": "not_submitted"}, 409)

    @app.exception_handler(InvalidExperiment)
    async def _inv(_: Request, exc: InvalidExperiment):
        return JSONResponse({"detail": str(exc)}, 422)

    @app.exception_handler(EvidenceIntegrityError)
    async def _integrity(_: Request, exc: EvidenceIntegrityError):
        # 409: the stored evidence conflicts with its own fingerprints; nothing was executed.
        return JSONResponse({"detail": str(exc), "integrity": exc.report}, 409)

    @app.exception_handler(EngineBusy)
    async def _busy(_: Request, exc: EngineBusy):
        return JSONResponse({"detail": str(exc)}, 429)

    @app.exception_handler(ExperimentTimeout)
    async def _timeout(_: Request, exc: ExperimentTimeout):
        # Experiments record timeouts as failed evidence; this covers non-experiment work (boundary, dataset).
        return JSONResponse({"detail": f"wall-clock limit of {LIMITS.max_experiment_seconds:g}s exceeded: {exc}"}, 504)

    # ------------------------------------------------------------ catalogue
    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/api/owasp")
    def owasp():
        return {"taxonomy": TAXONOMY_NAME, "version": TAXONOMY_VERSION, "url": TAXONOMY_URL,
                "categories": [c.model_dump() for c in CATEGORIES]}

    @app.get("/api/scenarios")
    def scenarios():
        return [{**s.__dict__, "models": list(MODELS)} for s in SCENARIOS.values()]

    @app.get("/api/limits")
    def limits():
        return LIMITS.__dict__

    @app.get("/api/scenarios/{scenario_id}/dataset")
    def dataset(scenario_id: str, model_id: str = "mlp-tanh-16x16", seed: int = 7):
        scen = SCENARIOS.get(scenario_id)
        if scen is None or model_id not in MODELS or not 0 <= seed < 2**31:
            raise HTTPException(404, "unknown scenario, model or invalid seed")
        # Gated: an uncached seed trains a model, so it shares the experiment concurrency/time limits.
        return engine.dataset_view(scenario_id, model_id, seed)

    @app.post("/api/boundary")
    def boundary(req: BoundaryRequest):
        return engine.boundary(req)

    # ---------------------------------------------------------- experiments
    @app.post("/api/experiments/baseline")
    def run_baseline(req: BaselineRequest):
        return engine.run_baseline(req).model_dump(mode="json")

    @app.post("/api/experiments/attack")
    def run_attack(req: AttackRequest):
        return engine.run_attack(req).model_dump(mode="json")

    @app.post("/api/experiments/defend")
    def run_defend(req: DefendRequest):
        return engine.run_defend(req).model_dump(mode="json")

    # ------------------------------------------------- ML02 data poisoning
    @app.post("/api/experiments/poisoning/baseline")
    def poisoning_baseline(req: PoisonBaselineRequest):
        return engine.poisoning.run_baseline(req).model_dump(mode="json")

    @app.post("/api/experiments/poisoning/attack")
    def poisoning_attack(req: PoisonAttackRequest):
        return engine.poisoning.run_attack(req).model_dump(mode="json")

    @app.post("/api/experiments/poisoning/defend")
    def poisoning_defend(req: PoisonDefendRequest):
        return engine.poisoning.run_defend(req).model_dump(mode="json")

    # ------------------------------------------- ML01 spam-filter evasion
    @app.post("/api/experiments/text/baseline")
    def text_baseline(req: TextBaselineRequest):
        return engine.text.run_baseline(req).model_dump(mode="json")

    @app.post("/api/experiments/text/attack")
    def text_attack(req: TextAttackRequest):
        return engine.text.run_attack(req).model_dump(mode="json")

    @app.post("/api/experiments/text/defend")
    def text_defend(req: TextDefendRequest):
        return engine.text.run_defend(req).model_dump(mode="json")

    @app.post("/api/text/classify")
    def text_classify(req: TextClassifyRequest):
        """Classify one message with the bundled (optionally defended) spam filter. Not stored as evidence."""
        return engine.text.classify(req)

    @app.get("/api/text/messages")
    def text_messages(model_id: TextModelId = "tfidf-word-logreg", seed: int = Query(7, ge=0, lt=2**31),
                      label: int | None = Query(None, ge=0, le=1, description="0 = ham, 1 = spam"),
                      limit: int = Query(100, ge=1, le=LIMITS.max_text_samples)):
        """Held-out test messages with the undefended filter's predictions (to choose what to attack)."""
        return engine.text.test_messages(model_id, seed, label, limit)

    # ---------------------------------------------------- ML03 model inversion
    @app.post("/api/experiments/inversion/baseline")
    def inversion_baseline(req: InversionBaselineRequest):
        return engine.inversion.run_baseline(req).model_dump(mode="json")

    @app.post("/api/experiments/inversion/attack")
    def inversion_attack(req: InversionAttackRequest):
        return engine.inversion.run_attack(req).model_dump(mode="json")

    @app.post("/api/experiments/inversion/defend")
    def inversion_defend(req: InversionDefendRequest):
        return engine.inversion.run_defend(req).model_dump(mode="json")

    @app.post("/api/inversion/targets", response_model=TargetsPreview)
    def inversion_targets(req: InversionTargetsRequest):
        """EVALUATOR-facing preview: public features + released output of the selected targets (no attribute)."""
        return engine.inversion.targets_view(req)

    @app.post("/api/inversion/sensitivity", response_model=SensitivityReport)
    def inversion_sensitivity(req: SensitivityRequest):
        """EVALUATOR-facing: controlled sensitivity study, every setting compared with the auxiliary-only attacker."""
        return engine.inversion.sensitivity(req)

    @app.get("/api/inversion/sensitivity/{report_id}", response_model=SensitivityReport)
    def inversion_sensitivity_get(report_id: str):
        return engine.inversion.load_sensitivity(report_id)

    @app.get("/api/inversion/sensitivity/{report_id}/verify")
    def inversion_sensitivity_verify(report_id: str):
        return engine.inversion.verify_sensitivity(report_id)

    # ML03 interactive attack sessions -------------------------------------------------------------
    @app.post("/api/inversion/sessions", response_model=AttackerSessionView)
    def session_create(req: SessionCreateRequest):
        """EVALUATOR-facing: issue a session (fixed model, seed, policy, targets and budgets). Returns the attacker
        view, including the session id to hand to the attacker."""
        return sessions.create(req)

    @app.get("/api/inversion/sessions/{session_id}", response_model=AttackerSessionView, tags=["ml03-attacker"])
    def session_view(session_id: str):
        """ATTACKER-facing: public features + observed outputs of the in-scope targets and the budget state."""
        return sessions.view(session_id)

    @app.post("/api/inversion/sessions/{session_id}/query", response_model=AttackerQueryResponse,
              tags=["ml03-attacker"])
    def session_query(session_id: str, req: SessionQueryRequest):
        """ATTACKER-facing: one counted candidate query for an in-scope target (403 when the budget is exhausted)."""
        return sessions.query(session_id, req)

    @app.post("/api/inversion/sessions/{session_id}/submit", response_model=AttackerSubmitReceipt,
              tags=["ml03-attacker"])
    def session_submit(session_id: str, req: SessionSubmitRequest):
        """ATTACKER-facing: submit one guess per target, once. Closes the session; no score is returned."""
        return sessions.submit(session_id, req)

    @app.get("/api/inversion/sessions/{session_id}/score", response_model=SessionScore)
    def session_score(session_id: str):
        """EVALUATOR-facing: score a SUBMITTED session against ground truth and the auxiliary-only attacker."""
        return sessions.score(session_id)

    # ---------------------------------------------- ML04 membership inference (evaluator-facing only)
    @app.post("/api/experiments/membership/baseline")
    def membership_baseline(req: MembershipBaselineRequest):
        return engine.membership.run_baseline(req).model_dump(mode="json")

    @app.post("/api/experiments/membership/attack")
    def membership_attack(req: MembershipAttackRequest):
        return engine.membership.run_attack(req).model_dump(mode="json")

    @app.post("/api/experiments/membership/defend")
    def membership_defend(req: MembershipDefendRequest):
        return engine.membership.run_defend(req).model_dump(mode="json")

    @app.get("/api/membership/independence/{experiment_id}")
    def membership_independence(experiment_id: str):
        """Re-run a stored attack with the evaluation ground truth permuted (attacker outputs must not change) and
        check that the attacker's populations are disjoint from the victim's data."""
        rec = store.load(experiment_id)
        if getattr(rec, "family", None) != "membership":
            raise InvalidExperiment(f"{experiment_id} is not an ML04 membership experiment")
        return engine.membership.attacker_independence(rec)

    @app.post("/api/membership/study", response_model=StudyReport)
    def membership_study(req: StudyRequest):
        """Multi-seed study of every configuration (one gated call per seed; up to 20 seeds ≈ 2 minutes)."""
        return engine.membership.study(req)

    @app.get("/api/membership/study/{report_id}", response_model=StudyReport)
    def membership_study_get(report_id: str):
        return engine.membership.load_study(report_id)

    @app.get("/api/membership/study/{report_id}/verify")
    def membership_study_verify(report_id: str):
        return engine.membership.verify_study(report_id)

    @app.get("/api/experiments/{experiment_id}/boundary")
    def experiment_boundary(experiment_id: str,
                            model: Literal["clean", "poisoned", "defended", "clean_sanitized"] = "clean",
                            resolution: int = Query(80, ge=20, le=LIMITS.max_grid_resolution)):
        """Decision boundary of a model of a stored ML02 experiment, retrained from the record's own labels
        and checked against the recorded artifact hash."""
        return engine.poisoning.boundary(experiment_id, model, resolution)

    @app.get("/api/experiments")
    def list_experiments():
        return store.list()

    @app.get("/api/experiments/{experiment_id}")
    def get_experiment(experiment_id: str):
        return store.load(experiment_id).model_dump(mode="json")

    @app.post("/api/experiments/{experiment_id}/replay")
    def replay(experiment_id: str):
        return engine.replay(experiment_id)

    @app.get("/api/experiments/{experiment_id}/verify")
    def verify(experiment_id: str):
        """Evidence integrity check (hashes + metrics re-derived from per-sample results). No re-execution."""
        return engine.verify(experiment_id)

    @app.get("/api/experiments/{experiment_id}/evidence.json")
    def evidence_json(experiment_id: str):
        path = store.json_path(experiment_id)
        if not path.is_file():
            raise ExperimentNotFound(experiment_id)
        return FileResponse(path, media_type="application/json", filename=f"{experiment_id}.json")

    @app.get("/api/experiments/{experiment_id}/report.md")
    def evidence_md(experiment_id: str):
        # Rendered from the JSON record with a fresh integrity check (not the write-time snapshot).
        return PlainTextResponse(engine.report(experiment_id), media_type="text/markdown",
                                 headers={"Content-Disposition": f'attachment; filename="{experiment_id}.md"'})

    @app.get("/api/experiments/{experiment_id}/samples.csv")
    def evidence_csv(experiment_id: str):
        return Response(store.samples_csv(experiment_id), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{experiment_id}-samples.csv"'})

    # ----------------------------------------------------------- regression
    @app.get("/api/regression/cases")
    def regression_cases():
        return [c.model_dump(mode="json") for c in load_cases(cases_path)]

    @app.post("/api/regression/run")
    def regression_run():
        return run_suite(engine, cases_path, store.root / "regression")

    def _latest_report():
        path = store.root / "regression" / "security_regression_report.json"
        return json.loads(path.read_text()) if path.is_file() else None

    @app.get("/api/regression/latest")
    def regression_latest():
        return _latest_report()

    # ------------------------------------------------------------- dashboard
    @app.get("/api/dashboard")
    def dashboard():
        exps = store.list()
        by_mode: dict[str, int] = {}
        for e in exps:
            by_mode[e["evaluation"]] = by_mode.get(e["evaluation"], 0) + 1
        return {
            "experiments_total": len(exps),
            "experiments_by_evaluation": by_mode,
            "experiments_failed": sum(e["status"] == "failed" for e in exps),
            "recent": exps[:8],
            "owasp_implemented": sum(c.lab_status == "implemented" for c in CATEGORIES),
            "owasp_total": len(CATEGORIES),
            "regression_cases": len(load_cases(cases_path)),
            "regression_latest": _summary(_latest_report()),
        }

    _tag_routes(app)

    # -------------------------------------------------------------- frontend
    if frontend_dist is not None and (frontend_dist / "index.html").is_file():
        app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")
        index = frontend_dist / "index.html"

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404)
            candidate = (frontend_dist / path).resolve()
            if path and candidate.is_file() and candidate.is_relative_to(frontend_dist.resolve()):
                return FileResponse(candidate)
            return FileResponse(index)
    else:
        @app.get("/", include_in_schema=False)
        def no_ui():
            return PlainTextResponse("MLBreak API is running. Frontend not built: run `make ui` "
                                     "(or `cd frontend && npm ci && npm run build`). API docs: /docs")

    return app
