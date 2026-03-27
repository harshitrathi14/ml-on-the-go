from __future__ import annotations

from dataclasses import asdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import get_settings
from .csv_session import csv_sessions
from .data import DatasetSummary, generate_synthetic_dataset, split_dataset, summarize_dataset
from .modeling import TrainingOutput, train_models

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="ML on the Go", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI client — created once at startup
_settings = get_settings()

from .ai.factory import get_ai_client  # noqa: E402
_ai_client = get_ai_client(_settings)

# ---------------------------------------------------------------------------
# Pydantic models — synthetic training
# ---------------------------------------------------------------------------


class TrainRequest(BaseModel):
    n_rows: int = Field(5000, ge=500, le=50000)
    n_features: int = Field(100, ge=20, le=200)
    n_categorical: int = Field(10, ge=2, le=50)
    seed: int = 42
    decision_labels: Tuple[str, str] = ("default", "no_default")


class TrainResponse(BaseModel):
    dataset: dict
    leaderboard: List[dict]
    results: List[dict]


# ---------------------------------------------------------------------------
# Pydantic models — CSV upload + AI
# ---------------------------------------------------------------------------


class BinarizationStrategy(BaseModel):
    strategy: str
    positive_class: Optional[str] = None
    threshold: Optional[float] = None
    notes: str = ""


class DataQualityIssue(BaseModel):
    column: str
    issue_type: str
    severity: str
    description: str


class AIAnalysis(BaseModel):
    suggested_target_col: str
    confidence: str
    problem_type: str
    binarization_strategy: BinarizationStrategy
    data_quality_issues: List[DataQualityIssue] = []
    feature_notes: List[str] = []
    recommendation: str = ""


class UploadCSVResponse(BaseModel):
    session_id: str
    row_count: int
    column_count: int
    columns: List[str]
    ai_analysis: Dict[str, Any]


class TrainCSVRequest(BaseModel):
    session_id: str
    target_col: str
    positive_class: Optional[str] = None
    threshold: Optional[float] = None
    seed: int = 42


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


# ---------------------------------------------------------------------------
# Endpoints — original
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/train", response_model=TrainResponse)
async def train(request: TrainRequest) -> TrainResponse:
    df = generate_synthetic_dataset(
        n_rows=request.n_rows,
        n_features=request.n_features,
        n_categorical=request.n_categorical,
        seed=request.seed,
        decision_labels=request.decision_labels,
    )
    splits = split_dataset(df, seed=request.seed)
    summary = summarize_dataset(splits, target_col="decision")
    output = train_models(
        splits.train,
        splits.test,
        splits.oot,
        splits.etrc,
        target_col="decision_binary",
        seed=request.seed,
    )
    return TrainResponse(
        dataset=asdict(summary),
        leaderboard=output.leaderboard,
        results=[asdict(r) for r in output.results],
    )


# ---------------------------------------------------------------------------
# Endpoints — CSV upload flow
# ---------------------------------------------------------------------------


@app.post("/upload-csv", response_model=UploadCSVResponse)
async def upload_csv(file: UploadFile = File(...)) -> UploadCSVResponse:
    # Guards
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    raw = await file.read()
    if len(raw) > 50 * 1024 * 1024:  # 50 MB
        raise HTTPException(status_code=400, detail="File exceeds 50 MB limit.")

    try:
        df = pd.read_csv(BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    if len(df) < 100:
        raise HTTPException(status_code=400, detail="CSV must have at least 100 rows.")
    if len(df.columns) < 2:
        raise HTTPException(status_code=400, detail="CSV must have at least 2 columns.")

    ai_analysis = _ai_client.analyze_csv_metadata(df)
    session_id = csv_sessions.create(df)

    return UploadCSVResponse(
        session_id=session_id,
        row_count=len(df),
        column_count=len(df.columns),
        columns=list(df.columns),
        ai_analysis=ai_analysis,
    )


@app.post("/train-csv", response_model=TrainResponse)
async def train_csv(request: TrainCSVRequest) -> TrainResponse:
    session = csv_sessions.get(request.session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired. Please re-upload the CSV.",
        )

    df = session.df.copy()
    target_col = request.target_col

    if target_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{target_col}' not found in uploaded CSV.",
        )

    # --- Binarize target ---
    series = df[target_col]
    unique_vals = series.dropna().unique()

    try:
        if set(unique_vals).issubset({0, 1, True, False, "0", "1"}):
            # Already binary
            binary = series.map({True: 1, False: 0, 1: 1, 0: 0, "1": 1, "0": 0}).astype(int)
        elif request.positive_class is not None:
            # label_encode / top_k_classes: 1 if value == positive_class
            binary = (series.astype(str) == str(request.positive_class)).astype(int)
        elif request.threshold is not None:
            binary = (pd.to_numeric(series, errors="coerce") > request.threshold).astype(int)
        else:
            # Default: median threshold for numeric, else use most-frequent class as negative
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().sum() > len(series) * 0.8:
                median = float(numeric.median())
                binary = (numeric > median).astype(int)
            else:
                # Categorical with no guidance: use most-frequent as 0, everything else as 1
                most_common = series.value_counts().index[0]
                binary = (series != most_common).astype(int)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not binarize target column '{target_col}': {exc}",
        ) from exc

    if binary.nunique() != 2:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_col}' could not be reduced to exactly 2 classes after binarization.",
        )

    df = df.drop(columns=[target_col])
    df["__target_binary__"] = binary

    splits = split_dataset(df, target_col="__target_binary__", seed=request.seed)
    summary = summarize_dataset(splits, target_col="__target_binary__")
    output = train_models(
        splits.train,
        splits.test,
        splits.oot,
        splits.etrc,
        target_col="__target_binary__",
        seed=request.seed,
    )

    return TrainResponse(
        dataset=asdict(summary),
        leaderboard=output.leaderboard,
        results=[asdict(r) for r in output.results],
    )


# ---------------------------------------------------------------------------
# Endpoint — AI explanation
# ---------------------------------------------------------------------------


@app.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest) -> ExplainResponse:
    result = _ai_client.explain_model_results(
        leaderboard=request.leaderboard,
        results=request.results,
        dataset_summary=request.dataset_summary,
    )
    return ExplainResponse(**result)
