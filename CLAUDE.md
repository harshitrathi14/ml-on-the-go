# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML on the Go is a production-grade ML platform for financial risk analytics, specializing in loan default prediction, investment decisioning, and credit risk scoring. It features:

- **1000+ feature synthetic data generation** with realistic financial characteristics
- **Advanced feature engineering** (WOE encoding, target encoding, IV calculation)
- **Multi-model training** with Optuna hyperparameter optimization
- **Comprehensive evaluation** (AUC, KS statistic, Gini coefficient, stability metrics)
- **Model explainability** (SHAP, PSI drift, CSI feature stability)
- **Docker containerization** for deployment

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
cd docker
docker-compose up -d
```

Services:
- Backend API: http://localhost:8000
- Frontend: http://localhost:5173

### Run Tests
```bash
pytest backend/tests/ -v
```

## Architecture

```
ML on the Go/
├── backend/app/
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Pydantic settings
│   ├── api/v1/                 # API endpoints (datasets, training, inference)
│   └── core/
│       ├── data/               # Data generation & validation
│       ├── features/           # Feature engineering (WOE, target encoding)
│       ├── models/             # Model training & tuning
│       ├── evaluation/         # Metrics (KS, Gini, stability)
│       └── explainability/     # SHAP, PSI, CSI
├── frontend/                   # Static dashboard
├── docker/                     # Containerization
├── configs/                    # YAML configurations
└── artifacts/                  # Model artifacts & logs
```

### Backend Core Modules

#### Data Generation (`backend/app/core/data/generator.py`)
- Generates 100k+ rows with 1000+ features
- Feature categories: demographic (50), financial (150), behavioral (200), bureau (150), time (100), derived (150), interaction (100), lag (100), noise (100)
- Configurable class imbalance (~15% default rate)
- Realistic missing value patterns (MCAR, MAR)
- Correlated features and non-linear target effects

#### Feature Engineering (`backend/app/core/features/encoders.py`)
- **WOEEncoder**: Weight of Evidence encoding with Information Value (IV) calculation
- **TargetEncoder**: Bayesian target encoding with cross-validation
- **SmartEncoder**: Automatic encoding strategy selection based on cardinality
- IV thresholds: <0.02 (useless), 0.02-0.1 (weak), 0.1-0.3 (medium), 0.3-0.5 (strong), >0.5 (suspicious)

#### Model Training (`backend/app/core/models/`)
- **definitions.py**: Model registry (Logistic, RandomForest, XGBoost, LightGBM, GradientBoosting)
- **tuning.py**: Optuna Bayesian optimization with TPE sampler and median pruning
- **trainer.py**: Training orchestration with CV, artifact management, leaderboard generation

Supported models:
| Model | Use Case |
|-------|----------|
| LogisticRegression | Interpretable credit scoring |
| RandomForest | Robust ensemble |
| GradientBoosting | High performance |
| XGBoost | State-of-the-art gradient boosting |
| LightGBM | Fast, memory-efficient |

#### Evaluation Metrics (`backend/app/core/evaluation/metrics.py`)
- Standard: accuracy, precision, recall, F1, ROC-AUC
- Financial: **KS statistic** (max separation), **Gini coefficient** (2*AUC - 1)
- Stability: cross-split metric comparison, degradation detection

#### Explainability (`backend/app/core/explainability/`)
- **shap_engine.py**: Global and local SHAP explanations
- **psi.py**: Population Stability Index for score distribution drift
- **csi.py**: Characteristic Stability Index for feature-level drift

PSI thresholds:
- <0.1: No significant drift
- 0.1-0.25: Moderate drift (investigation recommended)
- >0.25: Significant drift (action required)

### Frontend (`frontend/`)
- **index.html**: Single-page dashboard with Plotly.js
- **app.js**: API integration, renders leaderboard, ROC curves, feature importance, confusion matrix

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/train` | POST | Train models with configuration |
| `/datasets/generate` | POST | Generate synthetic data |
| `/datasets/{id}/split` | POST | Split into train/test/OOT/ETRC |
| `/models` | GET | List registered models |
| `/explain/shap/{model_id}` | GET | SHAP explanations |
| `/monitor/psi/{model_id}` | GET | PSI drift analysis |

## Key Design Decisions

### Data Splits
- **Train (60%)**: Model fitting
- **Test (20%)**: Hyperparameter validation
- **OOT (10%)**: Out-of-Time simulation for temporal stability
- **ETRC (10%)**: Economic Through-the-Cycle stress testing

### Financial Risk Focus
- KS and Gini metrics aligned with credit risk standards
- WOE/IV methodology for regulatory compliance (SR 11-7)
- PSI/CSI monitoring for production model governance
- SHAP explanations for model interpretability requirements

### Hyperparameter Tuning
- Optuna with TPE (Tree-structured Parzen Estimator) sampling
- Default 50 trials with 5-fold stratified CV
- Early stopping after 10 rounds without improvement
- Parameter importance analysis

## Configuration

### Environment Variables
```bash
APP_ENV=development        # development, staging, production
LOG_LEVEL=DEBUG            # DEBUG, INFO, WARNING, ERROR
ARTIFACT_DIR=./artifacts   # Model storage path
```

### Training Configuration
```python
TrainingConfig(
    model_types=[ModelType.LIGHTGBM, ModelType.XGBOOST],
    cv_folds=5,
    enable_tuning=True,
    tuning_trials=50,
    scoring="roc_auc",
)
```

## Documentation

- **AGENTS.md**: Agent-based architecture for ML pipeline orchestration
- **skills.md**: Skills matrix for ML Engineer, Data Scientist, Risk Analytics roles
- **README.md**: Quick start guide

## Dependencies

Key packages:
- FastAPI, Uvicorn (web framework)
- pandas, numpy, pyarrow (data processing)
- scikit-learn, XGBoost, LightGBM (ML models)
- Optuna (hyperparameter tuning)
- SHAP (explainability)
- structlog (structured logging)
