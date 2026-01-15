from __future__ import annotations

from dataclasses import asdict
from typing import List, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .data import DatasetSummary, generate_synthetic_dataset, split_dataset, summarize_dataset
from .modeling import TrainingOutput, train_models

app = FastAPI(title="ML on the Go", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    payload = TrainResponse(
        dataset=asdict(summary),
        leaderboard=output.leaderboard,
        results=[asdict(result) for result in output.results],
    )

    return payload
