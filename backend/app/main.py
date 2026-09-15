from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd
from fastapi import APIRouter, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import joining, pipeline, profiling, sources
from .config import get_settings
from .jobs import JobManager
from .store import DEFAULT_CHUNK_BYTES, SessionStore, StoreError, TooLarge

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

_settings = get_settings()

app = FastAPI(title=_settings.app_name, version="0.5.0", root_path=_settings.root_path)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from .ai.factory import get_ai_client  # noqa: E402
_ai_client = get_ai_client(_settings)

_data_root = Path(_settings.data_root)
_store = SessionStore(_data_root / "sessions", max_file_bytes=int(_settings.max_upload_gb * 1024**3))
_jobs = JobManager(_data_root / "jobs", max_workers=_settings.max_concurrent_jobs)

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TrainRequest(BaseModel):
    n_rows: int = Field(5000, ge=500, le=200000)
    n_features: int = Field(32, ge=20, le=200)
    n_categorical: int = Field(10, ge=2, le=50)
    seed: int = 42
    decision_labels: Tuple[str, str] = ("default", "no_default")
    default_rate: float = Field(0.18, ge=0.01, le=0.6)
    tier: str = Field("balanced", pattern="^(quick|balanced|thorough)$")


class TrainResponse(BaseModel):
    dataset: dict
    leaderboard: List[dict]
    results: List[dict]


class TrainCSVRequest(BaseModel):
    session_id: str
    target_col: str
    positive_class: Optional[str] = None
    threshold: Optional[float] = None
    seed: int = 42
    tier: str = Field("balanced", pattern="^(quick|balanced|thorough)$")


class UploadCSVResponse(BaseModel):
    session_id: str
    row_count: int
    column_count: int
    columns: List[str]
    ai_analysis: Dict[str, Any]
    size_bytes: int = 0


class JobCreated(BaseModel):
    job_id: str
    status: str


class FileInit(BaseModel):
    filename: str
    size_bytes: int = Field(ge=0)
    chunk_bytes: int = Field(DEFAULT_CHUNK_BYTES, ge=1024 * 1024, le=256 * 1024 * 1024)


class JoinKey(BaseModel):
    file_id: str
    primary_col: str
    file_col: str


class JoinRequest(BaseModel):
    primary_file_id: str
    joins: List[JoinKey] = []


class TargetRequest(BaseModel):
    target_col: str
    positive_class: Optional[str] = None
    threshold: Optional[float] = None


class SessionTrainRequest(BaseModel):
    target_col: str
    positive_class: Optional[str] = None
    threshold: Optional[float] = None
    exclude_cols: List[str] = []
    time_col: Optional[str] = None
    seed: int = 42
    tier: str = Field("balanced", pattern="^(quick|balanced|thorough)$")


class ScoreRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., min_length=1, max_length=500)


class ExplainRequest(BaseModel):
    leaderboard: List[dict]
    results: List[dict]
    dataset_summary: Optional[dict] = None


class ExplainResponse(BaseModel):
    executive_summary: str
    best_model_analysis: str
    feature_insights: List[str]
    risk_flags: List[str]
    recommendations: List[str]


class ReportRequest(BaseModel):
    leaderboard: List[dict]
    results: List[dict]
    dataset_summary: Optional[dict] = None
    ai_insights: Optional[dict] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except TooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except StoreError as exc:
        status = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc


def _completed_files(meta: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {fid: f for fid, f in meta["files"].items() if f.get("status") == "ready"}


def _paths(session_id: str, files: Dict[str, Any]) -> Dict[str, Path]:
    return {fid: _store.parquet_path(session_id, fid) for fid in files}


def _dataset_path(session_id: str, meta: Dict[str, Any]) -> Path:
    """The modelling table: the joined file, else the primary source."""
    join = meta.get("join")
    if join and join.get("joined"):
        return _store.joined_path(session_id)
    if join and join.get("primary_file_id") in meta["files"]:
        return _store.parquet_path(session_id, join["primary_file_id"])
    files = _completed_files(meta)
    if len(files) == 1:
        return _store.parquet_path(session_id, next(iter(files)))
    raise HTTPException(status_code=400, detail="Confirm the join between your files first.")


def _dataset_profile(session_id: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    profile = meta.get("dataset_profile")
    if profile is None:
        profile = profiling.profile_parquet(_dataset_path(session_id, meta))
        _store.update(session_id, dataset_profile=profile)
    return profile


def _public_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Session metadata for the browser, with per-column profiles trimmed."""
    out = dict(meta)
    out["files"] = {
        fid: {k: v for k, v in f.items() if k != "profile"} | (
            {"profile": {
                "row_count": f["profile"]["row_count"],
                "column_count": f["profile"]["column_count"],
                "columns": [c["name"] for c in f["profile"]["columns"]],
                "date_columns": f["profile"]["date_columns"],
                "id_columns": f["profile"]["id_columns"],
            }} if f.get("profile") else {}
        )
        for fid, f in meta["files"].items()
    }
    return out


def _convert_and_profile(session_id: str, file_id: str) -> Dict[str, Any]:
    raw = _store.finish_upload(session_id, file_id)
    info = _store.read(session_id)["files"][file_id]
    out = _store.parquet_path(session_id, file_id)
    try:
        sources.convert_to_parquet(raw, info["kind"], out)
        profile = profiling.profile_parquet(out)
    except Exception as exc:
        _store.update_file(session_id, file_id, status="failed", error=f"{type(exc).__name__}: {exc}")
        raise HTTPException(status_code=400, detail=f"Could not read {info['filename']}: {exc}") from exc
    if profile["row_count"] < 100:
        _store.update_file(session_id, file_id, status="failed", error="fewer than 100 rows")
        raise HTTPException(status_code=400, detail=f"{info['filename']} must have at least 100 rows.")
    if profile["column_count"] < 2:
        _store.update_file(session_id, file_id, status="failed", error="fewer than 2 columns")
        raise HTTPException(status_code=400, detail=f"{info['filename']} must have at least 2 columns.")
    # A new file invalidates any previous join / target analysis.
    _store.update(session_id, join=None, dataset_profile=None, target=None)
    return _store.update_file(session_id, file_id, status="ready", profile=profile)


def _analyse_session(session_id: str) -> Dict[str, Any]:
    meta = _store.read(session_id)
    files = _completed_files(meta)
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one file first.")
    proposal = joining.propose(files, _paths(session_id, files), _ai_client)
    proposal["joined"] = False
    _store.update(session_id, join=proposal, dataset_profile=None, target=None)
    return proposal


def _apply_join(session_id: str, request: JoinRequest) -> Dict[str, Any]:
    meta = _store.read(session_id)
    files = _completed_files(meta)
    if request.primary_file_id not in files:
        raise HTTPException(status_code=400, detail="Primary file not found in this session.")
    proposal = meta.get("join") or {"roles": {}, "suggested_target_col": None, "time_col": None, "notes": ""}
    joins = []
    for key in request.joins:
        if key.file_id not in files or key.file_id == request.primary_file_id:
            raise HTTPException(status_code=400, detail=f"Unknown file in join: {key.file_id}")
        role = proposal.get("roles", {}).get(key.file_id, "other")
        joins.append({"file_id": key.file_id, "filename": files[key.file_id]["filename"], "role": role,
                      "selected": {"primary_col": key.primary_col, "file_col": key.file_col}})
    if joins:
        stats = joining.build_joined(
            duckdb.connect(), _store.parquet_path(session_id, request.primary_file_id),
            joins, _paths(session_id, files), files, _store.joined_path(session_id),
        )
    else:
        stats = None
    proposal.update(primary_file_id=request.primary_file_id, joins=joins, joined=bool(joins), stats=stats)
    _store.update(session_id, join=proposal, dataset_profile=None, target=None)
    return {"join": proposal, "dataset_profile": _dataset_profile(session_id, _store.read(session_id))}


def _analyse_target(session_id: str, request: TargetRequest) -> Dict[str, Any]:
    meta = _store.read(session_id)
    path = _dataset_path(session_id, meta)
    profile = _dataset_profile(session_id, meta)
    columns = {c["name"] for c in profile["columns"]}
    if request.target_col not in columns:
        raise HTTPException(status_code=400, detail=f"Column '{request.target_col}' not found.")

    sample = profiling.read_sample(path, profiling.IV_SAMPLE_ROWS)
    try:
        binarized = pipeline.binarize_target(sample, request.target_col, request.positive_class, request.threshold)
    except pipeline.TargetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    target = binarized.pop(pipeline.TARGET_BINARY_COL)

    join = meta.get("join") or {}
    time_col = join.get("time_col") if join.get("time_col") in columns else None
    if time_col is None and profile["date_columns"]:
        time_col = profile["date_columns"][0]

    feature_cols = [c for c in binarized.columns if c not in profile["date_columns"]]
    iv_table = profiling.compute_iv(binarized[feature_cols], target)
    flags = (profiling.pii_check(profile, sample) + profiling.leakage_check(iv_table, profile)
             + profiling.id_check(profile))
    flags = [f for f in flags if f["column"] != request.target_col]
    excluded = sorted({f["column"] for f in flags if f["kind"] in ("pii", "leakage", "identifier")})

    result = {
        "target_col": request.target_col,
        "binarization": pipeline.describe_binarization(sample[request.target_col], request.positive_class, request.threshold),
        "positive_rate": round(float(target.mean()), 4),
        "sample_rows": int(len(sample)),
        "iv": iv_table[:60],
        "flags": flags,
        "excluded_by_default": excluded,
        "time_col": time_col,
        "date_columns": profile["date_columns"],
        "split_strategy": "time" if time_col else "random",
    }
    _store.update(session_id, target=result)
    return result


def _submit_session_train(session_id: str, request: SessionTrainRequest) -> str:
    meta = _store.read(session_id)
    path = _dataset_path(session_id, meta)
    profile = _dataset_profile(session_id, meta)
    columns = {c["name"] for c in profile["columns"]}
    if request.target_col not in columns:
        raise HTTPException(status_code=400, detail=f"Column '{request.target_col}' not found.")
    time_col = request.time_col if request.time_col in columns else None
    return _jobs.submit("session", {
        "session_dir": str(_store.session_dir(session_id)),
        "data_path": str(path),
        "target_col": request.target_col,
        "positive_class": request.positive_class,
        "threshold": request.threshold,
        "exclude_cols": request.exclude_cols,
        "date_cols": profile["date_columns"],
        "time_col": time_col,
        "seed": request.seed,
        "tier": request.tier,
    })


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "app": _settings.app_name}


# ---------------------------------------------------------------------------
# Sessions: multi-file chunked upload → analyse → join → target → train
# ---------------------------------------------------------------------------


@router.post("/sessions", status_code=201)
async def create_session() -> dict:
    return {"session_id": _store.create_session(), "chunk_bytes": DEFAULT_CHUNK_BYTES,
            "max_file_bytes": _store.max_file_bytes}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    return _public_meta(_store_call(_store.read, session_id))


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str) -> Response:
    _store.delete_session(session_id)
    return Response(status_code=204)


@router.post("/sessions/{session_id}/files")
async def init_file(session_id: str, request: FileInit) -> dict:
    return _store_call(_store.init_file, session_id, request.filename, request.size_bytes, request.chunk_bytes)


@router.put("/sessions/{session_id}/files/{file_id}/chunks/{index}")
async def put_chunk(session_id: str, file_id: str, index: int, request: Request) -> dict:
    try:
        return await _store.append_chunk(session_id, file_id, index, request.stream())
    except TooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except StoreError as exc:
        raise HTTPException(status_code=409 if "Expected chunk" in str(exc) else 400, detail=str(exc)) from exc


@router.get("/sessions/{session_id}/files/{file_id}")
async def get_file(session_id: str, file_id: str) -> dict:
    meta = _store_call(_store.read, session_id)
    info = meta["files"].get(file_id)
    if info is None:
        raise HTTPException(status_code=404, detail="File not found in this session.")
    return {k: v for k, v in info.items() if k != "profile"}


@router.post("/sessions/{session_id}/files/{file_id}/complete")
async def complete_file(session_id: str, file_id: str) -> dict:
    info = await run_in_threadpool(_store_call, _convert_and_profile, session_id, file_id)
    return {k: v for k, v in info.items() if k != "profile"} | {"profile": info["profile"]}


@router.delete("/sessions/{session_id}/files/{file_id}", status_code=204)
async def delete_file(session_id: str, file_id: str) -> Response:
    _store_call(_store.delete_file, session_id, file_id)
    return Response(status_code=204)


@router.post("/sessions/{session_id}/analyse")
async def analyse_session(session_id: str) -> dict:
    return await run_in_threadpool(_store_call, _analyse_session, session_id)


@router.post("/sessions/{session_id}/join")
async def join_session(session_id: str, request: JoinRequest) -> dict:
    return await run_in_threadpool(_store_call, _apply_join, session_id, request)


@router.get("/sessions/{session_id}/dataset")
async def dataset_profile(session_id: str) -> dict:
    meta = _store_call(_store.read, session_id)
    return await run_in_threadpool(_store_call, _dataset_profile, session_id, meta)


@router.post("/sessions/{session_id}/target")
async def analyse_target(session_id: str, request: TargetRequest) -> dict:
    return await run_in_threadpool(_store_call, _analyse_target, session_id, request)


@router.post("/sessions/{session_id}/train", response_model=JobCreated, status_code=202)
async def train_session(session_id: str, request: SessionTrainRequest) -> JobCreated:
    job_id = _store_call(_submit_session_train, session_id, request)
    return JobCreated(job_id=job_id, status="queued")


# ---------------------------------------------------------------------------
# Single-request upload (mobile app, scripts): one CSV = one session
# ---------------------------------------------------------------------------


def _check_csv_name(filename: str) -> None:
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")


async def _single_file_session(filename: str, chunks) -> UploadCSVResponse:
    session_id = _store.create_session()
    try:
        file_id = await _store.save_whole(session_id, filename, chunks)
    except TooLarge as exc:
        _store.delete_session(session_id)
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    try:
        info = await run_in_threadpool(_convert_and_profile, session_id, file_id)
        sample = await run_in_threadpool(profiling.read_sample, _store.parquet_path(session_id, file_id), 200_000)
        ai_analysis = await run_in_threadpool(_ai_client.analyze_csv_metadata, sample)
    except HTTPException:
        _store.delete_session(session_id)
        raise
    profile = info["profile"]
    return UploadCSVResponse(
        session_id=session_id,
        row_count=profile["row_count"],
        column_count=profile["column_count"],
        columns=[c["name"] for c in profile["columns"]],
        ai_analysis=ai_analysis,
        size_bytes=info["received_bytes"],
    )


@router.post("/uploads", response_model=UploadCSVResponse)
async def upload_stream(request: Request, filename: str) -> UploadCSVResponse:
    """Raw-body upload (the file is the request body) streamed straight to disk."""
    _check_csv_name(filename)
    return await _single_file_session(filename, request.stream())


@router.post("/upload-csv", response_model=UploadCSVResponse)
async def upload_csv(file: UploadFile = File(...)) -> UploadCSVResponse:
    """Multipart upload, kept for the mobile app."""
    _check_csv_name(file.filename or "")

    async def chunks():
        while chunk := await file.read(1024 * 1024):
            yield chunk

    return await _single_file_session(file.filename or "upload.csv", chunks())


def _single_file_train_params(request: TrainCSVRequest) -> SessionTrainRequest:
    """Legacy train: exclude identifiers/PII automatically, use any date column for time splits."""
    analysis = _store_call(_analyse_target, request.session_id,
                           TargetRequest(target_col=request.target_col, positive_class=request.positive_class,
                                         threshold=request.threshold))
    return SessionTrainRequest(
        target_col=request.target_col, positive_class=request.positive_class, threshold=request.threshold,
        exclude_cols=analysis["excluded_by_default"], time_col=analysis["time_col"], seed=request.seed,
        tier=request.tier,
    )


# ---------------------------------------------------------------------------
# Background training jobs
# ---------------------------------------------------------------------------


@router.post("/jobs/train", response_model=JobCreated, status_code=202)
async def submit_train(request: TrainRequest) -> JobCreated:
    job_id = _jobs.submit("synthetic", request.model_dump())
    return JobCreated(job_id=job_id, status="queued")


@router.post("/jobs/train-csv", response_model=JobCreated, status_code=202)
async def submit_train_csv(request: TrainCSVRequest) -> JobCreated:
    params = await run_in_threadpool(_single_file_train_params, request)
    job_id = _store_call(_submit_session_train, request.session_id, params)
    return JobCreated(job_id=job_id, status="queued")


@router.get("/jobs/{job_id}")
async def job_status(job_id: str) -> dict:
    status = _jobs.status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return status


@router.get("/jobs/{job_id}/result")
async def job_result(job_id: str) -> FileResponse:
    status = _jobs.status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if status["status"] != "succeeded":
        raise HTTPException(status_code=409, detail=f"Job is {status['status']}.")
    return FileResponse(_jobs.result_path(job_id), media_type="application/json")


# ---------------------------------------------------------------------------
# Synchronous training (kept for the mobile app; the web UI uses /jobs)
# ---------------------------------------------------------------------------


@router.post("/train", response_model=TrainResponse)
def train(request: TrainRequest) -> dict:
    return pipeline.run_synthetic(**request.model_dump())


@router.post("/train-csv", response_model=TrainResponse)
def train_csv(request: TrainCSVRequest) -> dict:
    params = _single_file_train_params(request)
    meta = _store_call(_store.read, request.session_id)
    profile = _dataset_profile(request.session_id, meta)
    try:
        result = pipeline.run_session(
            str(_dataset_path(request.session_id, meta)),
            params.target_col, positive_class=params.positive_class, threshold=params.threshold,
            exclude_cols=params.exclude_cols, date_cols=profile["date_columns"],
            time_col=params.time_col, seed=params.seed, tier=params.tier,
        )
    except pipeline.TargetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _store.delete_session(request.session_id)
    return result


# ---------------------------------------------------------------------------
# Scoring new applications with a trained model bundle
# ---------------------------------------------------------------------------

_bundle_cache: Dict[str, Any] = {}


def _bundle(model_id: str):
    from .engine.artifact import ModelBundle

    if not model_id.isalnum():
        raise HTTPException(status_code=404, detail="Model not found.")
    if model_id not in _bundle_cache:
        model_dir = _jobs.result_path(model_id).parent / "model"
        if not (model_dir / "bundle.joblib").exists():
            raise HTTPException(status_code=404, detail="Model not found. Train a model first.")
        _bundle_cache[model_id] = ModelBundle.load(model_dir)
        if len(_bundle_cache) > 8:
            _bundle_cache.pop(next(iter(_bundle_cache)))
    return _bundle_cache[model_id]


@router.get("/models/{model_id}/schema")
async def model_schema(model_id: str) -> dict:
    return _bundle(model_id).form_schema()


@router.post("/models/{model_id}/score")
async def score_records(model_id: str, request: ScoreRequest) -> dict:
    bundle = _bundle(model_id)
    frame = pd.DataFrame(request.records)
    try:
        results = await run_in_threadpool(bundle.score, frame, True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not score these records: {exc}") from exc
    return {"results": results, "model": bundle.meta.get("champion")}


def _score_file(model_id: str, filename: str, raw: bytes) -> Dict[str, Any]:
    from io import BytesIO

    from .engine import harmonise

    bundle = _bundle(model_id)
    try:
        frame = pd.read_csv(BytesIO(raw), low_memory=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc
    if frame.empty:
        raise HTTPException(status_code=400, detail="The file has no rows.")
    mapping = harmonise.map_columns(bundle.schema, [str(c) for c in frame.columns], _ai_client)
    mapped = harmonise.apply_mapping(frame, bundle.schema, mapping)
    scored = bundle.score(mapped, with_reasons=len(mapped) <= 2000)
    drift_report = bundle.drift(mapped, [s["nu_score"] for s in scored])
    out = frame.copy()
    out["nu_score"] = [s["nu_score"] for s in scored]
    out["nu_band"] = [s["band"] for s in scored]
    out["probability"] = [s["probability"] for s in scored]
    out["base_score"] = [s["base_score"] for s in scored]
    for k in range(4):
        out[f"reason_{k + 1}"] = [
            (s["reasons"][k]["feature"] if len(s["reasons"]) > k else "") for s in scored
        ]
    file_id = _jobs.submit_file(model_id, out.to_csv(index=False).encode(), filename)
    return {
        "file_id": file_id, "rows": int(len(out)),
        "mapping": mapping,
        "missing": [m["schema_col"] for m in mapping if m["method"] == "missing"],
        "band_counts": out["nu_band"].value_counts().to_dict(),
        "mean_score": float(out["nu_score"].mean()),
        "drift": drift_report,
    }


@router.post("/models/{model_id}/score-file")
async def score_file(model_id: str, request: Request, filename: str = "scoring.csv") -> dict:
    raw = await request.body()
    if len(raw) > 200 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Scoring files are limited to 200 MB.")
    return await run_in_threadpool(_score_file, model_id, filename, raw)


@router.get("/models/{model_id}/score-file/{file_id}/download")
async def download_scored(model_id: str, file_id: str) -> FileResponse:
    path = _jobs.file_path(model_id, file_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Scored file not found.")
    return FileResponse(path, media_type="text/csv",
                        headers={"Content-Disposition": f"attachment; filename=nu-score-{file_id[:8]}.csv"})


# ---------------------------------------------------------------------------
# AI explanation
# ---------------------------------------------------------------------------


@router.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest) -> ExplainResponse:
    result = _ai_client.explain_model_results(
        leaderboard=request.leaderboard,
        results=request.results,
        dataset_summary=request.dataset_summary,
    )
    return ExplainResponse(**result)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


@router.post("/report/html")
def report_html(request: ReportRequest) -> HTMLResponse:
    from .reporting import generate_html_report

    html = generate_html_report(
        leaderboard=request.leaderboard,
        results=request.results,
        dataset_summary=request.dataset_summary,
        ai_insights=request.ai_insights,
    )
    return HTMLResponse(
        content=html,
        headers={"Content-Disposition": "attachment; filename=ml-report.html"},
    )


@router.post("/report/pdf")
def report_pdf(request: ReportRequest) -> Response:
    from .reporting import generate_pdf_report

    pdf_bytes = generate_pdf_report(
        leaderboard=request.leaderboard,
        results=request.results,
        dataset_summary=request.dataset_summary,
        ai_insights=request.ai_insights,
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=ml-report.pdf"},
    )


# ---------------------------------------------------------------------------
# Mounts: API under /api (web UI) and at the root (mobile app), then the SPA.
# ---------------------------------------------------------------------------

app.include_router(router, prefix="/api")
app.include_router(router)

if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
