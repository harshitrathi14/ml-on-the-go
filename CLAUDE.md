# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML on the Go is a production-grade ML platform for financial risk analytics, specializing in loan default prediction. It generates synthetic datasets with 100+ features, trains multiple ML models with hyperparameter tuning, and provides a dashboard for visualization.

## Commands

### Development Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Start the Backend (FastAPI)
```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Start the Frontend (Static File Server)
```bash
python -m http.server 5173 --directory frontend
```

Then open http://localhost:5173

### Docker Deployment
```bash
cd docker && docker-compose up -d
```
- Backend API: http://localhost:8000
- Frontend: http://localhost:5173

## Architecture

```
backend/app/
├── main.py              # FastAPI entry point with /health and /train endpoints
├── config.py            # Pydantic settings (env vars, paths, defaults)
├── data.py              # Dataset generation and splitting
├── modeling.py          # Model training orchestration
└── core/
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

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/train` | POST | Generate data, split, train models, return leaderboard |

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

### Data Splits
- **Train (60%)**: Model fitting
- **Test (20%)**: Hyperparameter validation
- **OOT (10%)**: Out-of-Time simulation for temporal stability
- **ETRC (10%)**: Economic Through-the-Cycle stress testing

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
