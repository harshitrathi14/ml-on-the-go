# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Nu Score On the Go** ("Build ML models with your data") is an ML platform that lets non-specialists
train and compare models on their own data: loan default, sales forecasting, churn, pattern
detection. It generates synthetic datasets, trains multiple ML models, picks a champion and
provides a dashboard, AI explanations and HTML/PDF reports. Branding is Northern Arc / AltiFi
(cyan `#4FC6E0`, navy `#0D1C31`, Host Grotesk; assets in `frontend/brand/`).

Product promises shown in the UI (keep code consistent with them): uploaded data is deleted as
soon as training results are produced (`backend/app/jobs.py`); no PII is needed; free for an
introductory period, then ₹10 per case.

## Production deployment (10.91.16.86)

- URL: http://10.91.16.86/nuscore_claudecode/ (internal networks only, no login)
- systemd: `nuscore-otg.service` → uvicorn on 127.0.0.1:8096 (`deploy/systemd/`)
- Env: `/etc/nuscore-otg.env` (template: `deploy/nuscore-otg.env.example`); local Ollama LLM only
- nginx: `/etc/nginx/snippets/nuscore-otg.conf` (source: `deploy/nginx/`), included by the
  port-80 (`conf.d/nbfc-toolbox.conf`) and port-443 (`conf.d/newspaper.conf`) server blocks
- Data/logs: `/mnt/harshitrathi/mlpipeline-data/{uploads,jobs,logs}`
- After code changes: `sudo systemctl restart nuscore-otg && deploy/smoke_test.sh http://10.91.16.86/nuscore_claudecode`

## Commands

### Development Setup
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run locally (backend serves the frontend too)
```bash
AI_PROVIDER=none DATA_ROOT=./mlpipeline-data uvicorn backend.app.main:app --reload --port 8096
```

Then open http://localhost:8096 (API under `/api/*`, docs at `/docs`).
Smoke test: `deploy/smoke_test.sh http://127.0.0.1:8096`.

Note: port 8000 on the production host is used by another service; this app uses 8096.

### Docker Deployment
```bash
cd docker && docker-compose up -d
```
- Backend API: http://localhost:8000
- Frontend: http://localhost:5173

## Architecture

```
backend/app/
├── main.py              # FastAPI entry point: /api/* routes + static frontend
├── config.py            # Pydantic settings (env vars, paths, defaults)
├── pipeline.py          # Loads a session's table / synthetic data and calls the engine
├── jobs.py              # Disk-backed background training jobs (process pool, progress)
├── store.py             # Upload sessions: chunked resumable uploads, per-file Parquet
├── sources.py           # CSV / bureau-JSON / Parquet -> Parquet conversion (streaming)
├── profiling.py         # DuckDB column profiling, date/ID detection, PII/leakage checks, IV
├── joining.py           # Common-identifier matching across files (LLM roles + value overlap)
├── data.py              # Synthetic loan book; random and time-aware splits
├── reporting.py         # HTML / PDF report generation
├── ai/                  # LLM clients (Claude, OpenAI-compatible incl. Ollama, NoOp)
├── engine/              # The modelling engine
│   ├── train.py         #   orchestrator: splits -> preprocessing -> tuned zoo -> champion -> Nu Score -> SHAP -> bundle
│   ├── preprocess.py    #   imputation + indicators, one-hot, OOF target encoding, InputSchema
│   ├── zoo.py           #   model specs + Optuna search spaces/tuning (tiers quick/balanced/thorough)
│   ├── transformer.py   #   FT-Transformer (PyTorch) as a sklearn classifier
│   ├── evaluate.py      #   AUC/Gini/KS/PR-AUC/Brier/ECE, lift, PSI, composite champion score
│   ├── nuscore.py       #   Platt calibration, 0-1000 scaling, data-driven bands, cut-off curve
│   ├── explain.py       #   SHAP (tree/linear/permutation), beeswarm data, reason codes
│   ├── artifact.py      #   ModelBundle: save/load, score new records, form schema, batch drift
│   ├── harmonise.py     #   map a customer's differently named/formatted columns onto the schema
│   ├── normalise.py     #   deterministic input clean-up (₹, %, lakh grouping, yes/no, NA) with a change log
│   ├── drift.py         #   PSI drift (train-defined bins) for features and scores
│   ├── diagnostics.py   #   bootstrap AUC CI, duplicate/constant/correlation checks, warning engine
│   └── glossary.py      #   plain-English "high means / low means / why" per variable (credit + retail)
├── modeling.py          # Legacy simple trainer (used by input/run_and_report.py)
└── core/                # Older library code (WOE encoders, PSI/CSI, tuning); mostly superseded by engine/
    ├── data/generator.py           # Synthetic data with 1000+ features
    ├── features/encoders.py        # WOE, target, and smart encoding
    ├── models/
    │   ├── definitions.py          # Model registry (Logistic, RF, XGBoost, LightGBM, GB)
    │   ├── tuning.py               # Optuna hyperparameter optimization
    │   └── trainer.py              # Training with CV and artifact management
    ├── evaluation/metrics.py       # AUC, KS statistic, Gini coefficient
    └── explainability/             # SHAP, PSI drift, CSI feature stability
```

## API

All routes are mounted twice: under `/api` (web UI) and at the root (mobile app).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/uploads?filename=` | POST | Raw-body CSV upload streamed to disk; returns `session_id` + AI analysis |
| `/upload-csv` | POST | Multipart variant of the above (mobile app) |
| `/jobs/train`, `/jobs/train-csv` | POST | Submit a background training job (202 + `job_id`) |
| `/jobs/{id}`, `/jobs/{id}/result` | GET | Poll status; fetch result (same shape as `/train`) |
| `/train`, `/train-csv` | POST | Synchronous training (mobile app) |
| `/explain` | POST | LLM explanation of results |
| `/report/html`, `/report/pdf` | POST | Report downloads |
| `/sessions` … | POST/GET/PUT/DELETE | Multi-file flow: create session, init file, PUT chunks, complete, analyse (roles + keys), join, target (IV + checks), train |
| `/models/{id}/schema`, `/models/{id}/score`, `/models/{id}/score-file` | GET/POST | Score new applications with a trained bundle (form or CSV; columns harmonised) |

Training jobs take `tier`: `quick` / `balanced` / `thorough` (Optuna trial budget). Job status carries a live `message` and `progress.stage`.

Train request payload:
```json
{
  "n_rows": 5000,
  "n_features": 100,
  "n_categorical": 10,
  "seed": 42,
  "decision_labels": ["default", "no_default"]
}
```

## Key Design Decisions

### Cohorts (`data.split_cohorts`)
- **Train (55%)**: fitting and Optuna tuning
- **Validation (12%)**: model selection, thresholds — champion composite uses only train/validation/calibration
- **Calibration (10%)**: Platt calibration and Nu Score band cut-offs
- **Test (10%)**: untouched final estimate (bootstrap CI reported)
- **OOT (13%)**: latest observation dates when a time column exists (rows sharing a date are never split);
  otherwise a random hold-out explicitly labelled "not a true out-of-time"

Rows with a blank outcome are scored with the final model ("scoring only") and never enter a cohort.

Tests: `.venv/bin/python -m pytest tests -q` (engine invariants), `deploy/smoke_test.sh`, `deploy/session_test.py` (live API).

### Financial Risk Metrics
- **KS statistic**: Maximum separation between cumulative distributions
- **Gini coefficient**: 2*AUC - 1
- **WOE/IV methodology**: For regulatory compliance (SR 11-7)
- **PSI/CSI monitoring**: Production model governance

IV thresholds: <0.02 (useless), 0.02-0.1 (weak), 0.1-0.3 (medium), 0.3-0.5 (strong), >0.5 (suspicious)

PSI thresholds: <0.1 (stable), 0.1-0.25 (moderate drift), >0.25 (significant drift)

### Hyperparameter Tuning
- Optuna with TPE (Tree-structured Parzen Estimator) sampling
- Default 50 trials with 5-fold stratified CV
- Early stopping after 10 rounds without improvement

## Environment Variables

```bash
APP_ENV=development        # development, staging, production
LOG_LEVEL=INFO             # DEBUG, INFO, WARNING, ERROR
```

Settings configured via `backend/app/config.py` with `.env` file support.

## Related Documentation

- **AGENTS.md**: Agent-based architecture design for ML pipeline orchestration
- **skills.md**: Skills matrix for ML Engineer, Data Scientist, Risk Analytics roles
