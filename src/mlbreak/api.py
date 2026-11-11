"""FastAPI application: JSON API under ``/api`` and the built React UI at ``/``.

Security posture: local-first (binds 127.0.0.1 by default), no authentication
because there are no credentials or secrets, no upload endpoints, no endpoint
that accepts code, file paths or serialised models. Every request body is a
strict, bounded Pydantic model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import LIMITS, PROJECT_ROOT, cases_dir, data_dir
from .data import load_dataset
from .defenses import base_model
from .evidence import EvidenceStore, ExperimentNotFound
from .experiments import SCENARIOS, EngineBusy, ExperimentEngine, InvalidExperiment
from .models import MODELS
from .owasp import CATEGORIES, TAXONOMY_NAME, TAXONOMY_URL, TAXONOMY_VERSION
from .regression import load_cases, run_suite
from .schemas import AttackRequest, BaselineRequest, BoundaryRequest, DefendRequest

FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

_CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; "
        "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")


def _summary(report: dict | None) -> dict | None:
    if report is None:
        return None
    return {k: report[k] for k in ("generated_at", "total", "passed", "failed", "ok")}


def create_app(root: Path | None = None, cases: Path | None = None,
               frontend_dist: Path | None = FRONTEND_DIST) -> FastAPI:
    store = EvidenceStore(root or data_dir())
    engine = ExperimentEngine(store)
    cases_path = cases or cases_dir()
    app = FastAPI(title="MLBreak", version=__version__,
                  description="Adversarial ML security lab (OWASP ML Top 10, 2023). Intentionally vulnerable.")
    app.state.store, app.state.engine = store, engine

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

    @app.exception_handler(InvalidExperiment)
    async def _inv(_: Request, exc: InvalidExperiment):
        return JSONResponse({"detail": str(exc)}, 422)

    @app.exception_handler(EngineBusy)
    async def _busy(_: Request, exc: EngineBusy):
        return JSONResponse({"detail": str(exc)}, 429)

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
        ds = load_dataset(scen.dataset_id, seed)
        model = base_model(model_id, scen.dataset_id, seed)
        p1 = model.proba(ds.X_test)
        return {
            "scenario": scen.__dict__, "dataset": ds.describe(), "model": model.describe(),
            "train": {"X": ds.X_train.round(6).tolist(), "y": ds.y_train.tolist()},
            "test": {"X": ds.X_test.round(6).tolist(), "y": ds.y_test.tolist(),
                     "pred": model.predict(ds.X_test).tolist(), "p1": np.round(p1, 6).tolist()},
        }

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

    @app.get("/api/experiments")
    def list_experiments():
        return store.list()

    @app.get("/api/experiments/{experiment_id}")
    def get_experiment(experiment_id: str):
        return store.load(experiment_id).model_dump(mode="json")

    @app.post("/api/experiments/{experiment_id}/replay")
    def replay(experiment_id: str):
        return engine.replay(experiment_id)

    @app.get("/api/experiments/{experiment_id}/evidence.json")
    def evidence_json(experiment_id: str):
        path = store.json_path(experiment_id)
        if not path.is_file():
            raise ExperimentNotFound(experiment_id)
        return FileResponse(path, media_type="application/json", filename=f"{experiment_id}.json")

    @app.get("/api/experiments/{experiment_id}/report.md")
    def evidence_md(experiment_id: str):
        return PlainTextResponse(store.markdown(experiment_id), media_type="text/markdown",
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
