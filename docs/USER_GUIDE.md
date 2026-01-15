# ML on the Go - Comprehensive User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [System Requirements](#system-requirements)
4. [Installation](#installation)
5. [Data Generation](#data-generation)
6. [Feature Engineering](#feature-engineering)
7. [Model Training](#model-training)
8. [Model Evaluation](#model-evaluation)
9. [Explainability & Monitoring](#explainability--monitoring)
10. [API Reference](#api-reference)
11. [Docker Deployment](#docker-deployment)
12. [Configuration](#configuration)
13. [Troubleshooting](#troubleshooting)
14. [Best Practices](#best-practices)

---

## Introduction

ML on the Go is a production-grade machine learning platform designed for financial risk analytics. It provides end-to-end capabilities for building, training, evaluating, and monitoring credit risk models.

### Key Features

| Feature | Description |
|---------|-------------|
| **Synthetic Data Generation** | Generate 100k+ rows with 1000+ realistic financial features |
| **Feature Engineering** | WOE encoding, target encoding, IV calculation |
| **Multi-Model Training** | LogisticRegression, RandomForest, XGBoost, LightGBM, GradientBoosting |
| **Hyperparameter Tuning** | Optuna-based Bayesian optimization |
| **Evaluation Metrics** | AUC, KS statistic, Gini coefficient, precision, recall, F1 |
| **Explainability** | SHAP global/local explanations |
| **Drift Monitoring** | PSI (Population Stability Index), CSI (Characteristic Stability Index) |
| **Docker Support** | Containerized deployment for production |

### Use Cases

- **Credit Scoring**: Build scorecards for loan approval decisions
- **Default Prediction**: Predict probability of default (PD)
- **Portfolio Analytics**: Analyze portfolio risk characteristics
- **Model Validation**: Validate existing models with stability metrics
- **Regulatory Compliance**: Generate documentation for SR 11-7 requirements

---

## Quick Start

### 5-Minute Setup

```bash
# 1. Clone and setup
cd "ML on the go"
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Start the backend
uvicorn backend.app.main:app --reload --port 8000

# 3. In another terminal, start the frontend
python -m http.server 5173 --directory frontend

# 4. Open browser
open http://localhost:5173
```

### First Training Run

```python
from backend.app.core.data.generator import FinancialDataGenerator
from backend.app.core.models.trainer import train_models

# Generate synthetic data
generator = FinancialDataGenerator()
data = generator.generate()

X = data.drop('target', axis=1)
y = data['target']

# Train models
result = train_models(X, y, model_types=['lightgbm', 'xgboost'])
print(result.leaderboard)
```

---

## How to Run This Application

This section provides detailed instructions for running ML on the Go in different environments.

### Option 1: Local Development (Recommended for Development)

#### Prerequisites

1. **Python 3.9+** installed (3.11+ recommended)
2. **Git** for cloning the repository
3. **8GB+ RAM** for handling large datasets

#### Step-by-Step Instructions

**Step 1: Clone the Repository**

```bash
git clone https://github.com/harshitrathi14/ml-on-the-go.git
cd ml-on-the-go
```

**Step 2: Create Virtual Environment**

```bash
# Create virtual environment
python -m venv .venv

# Activate on macOS/Linux
source .venv/bin/activate

# Activate on Windows (Command Prompt)
.venv\Scripts\activate.bat

# Activate on Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

**Step 3: Install Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Step 4: Start the Backend API Server**

```bash
# Development mode with auto-reload
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode with multiple workers
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Step 5: Start the Frontend (in a new terminal)**

```bash
# Activate virtual environment first
source .venv/bin/activate  # or Windows equivalent

# Start static file server
python -m http.server 5173 --directory frontend
```

**Step 6: Access the Application**

- **Frontend Dashboard**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Health Check**: http://localhost:8000/health

#### Verify Installation

```bash
# Test the health endpoint
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy"}
```

---

### Option 2: Docker Deployment (Recommended for Production)

#### Prerequisites

1. **Docker** installed and running
2. **Docker Compose** installed
3. **4GB+ disk space** for images

#### Step-by-Step Instructions

**Step 1: Navigate to Docker Directory**

```bash
cd ml-on-the-go/docker
```

**Step 2: Build and Start Services**

```bash
# Build images and start containers in detached mode
docker-compose up -d --build

# View logs
docker-compose logs -f

# Check running services
docker-compose ps
```

**Step 3: Access Services**

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | http://localhost:5173 | Dashboard UI |
| Backend API | http://localhost:8000 | REST API |
| API Docs | http://localhost:8000/docs | Swagger documentation |

**Step 4: Stop Services**

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

#### Docker Commands Reference

```bash
# Rebuild specific service
docker-compose build backend

# View backend logs only
docker-compose logs -f backend

# Restart a service
docker-compose restart backend

# Scale backend workers
docker-compose up -d --scale backend=3

# Execute command in running container
docker-compose exec backend python -c "print('Hello from container')"
```

---

### Option 3: Python Script / Jupyter Notebook

For programmatic usage without running the web server:

**Step 1: Setup Environment**

```bash
cd ml-on-the-go
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Step 2: Run Python Script**

```python
#!/usr/bin/env python3
"""Example: Complete ML Pipeline"""

from backend.app.core.data.generator import FinancialDataGenerator
from backend.app.core.models.trainer import train_models
from backend.app.core.models.definitions import ModelType
from backend.app.core.evaluation.metrics import compute_all_metrics

# 1. Generate synthetic data
print("Generating data...")
generator = FinancialDataGenerator()
data = generator.generate()
print(f"Data shape: {data.shape}")

# 2. Prepare features and target
X = data.drop('target', axis=1)
y = data['target']

# 3. Simple train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Train models
print("Training models...")
result = train_models(
    X_train, y_train,
    X_test, y_test,
    model_types=['lightgbm', 'xgboost', 'logistic']
)

# 5. View results
print("\n=== LEADERBOARD ===")
print(result.leaderboard.to_string())

# 6. Get best model predictions
best_model = result.models[ModelType.LIGHTGBM].model
y_proba = best_model.predict_proba(X_test)[:, 1]
metrics = compute_all_metrics(y_test, best_model.predict(X_test), y_proba)

print(f"\n=== BEST MODEL METRICS ===")
print(f"AUC-ROC:  {metrics.auc_roc:.4f}")
print(f"KS Stat:  {metrics.ks_statistic:.4f}")
print(f"Gini:     {metrics.gini:.4f}")
```

**Step 3: Run in Jupyter Notebook**

```bash
# Install Jupyter
pip install jupyter

# Start Jupyter
jupyter notebook
```

Then create a new notebook and import the modules as shown above.

---

### Running the Web Dashboard

Once both backend and frontend are running, the dashboard provides:

1. **Dataset Generation**: Click "Generate Data" to create synthetic datasets
2. **Model Training**: Configure and train multiple models
3. **Results Visualization**: View ROC curves, feature importance, confusion matrix
4. **Leaderboard**: Compare model performance

#### Dashboard Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                     ML on the Go Dashboard                       │
├─────────────────────────────────────────────────────────────────┤
│  1. Configure Dataset                                            │
│     └─ Set rows, features, default rate                         │
│                                                                  │
│  2. Generate & Train                                             │
│     └─ Click "Run Training" button                              │
│                                                                  │
│  3. View Results                                                 │
│     ├─ Leaderboard (model comparison)                           │
│     ├─ ROC Curves (per model, per split)                        │
│     ├─ Feature Importance (top 15)                              │
│     └─ Confusion Matrix                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

### Common Commands Reference

| Task | Command |
|------|---------|
| Start backend (dev) | `uvicorn backend.app.main:app --reload --port 8000` |
| Start backend (prod) | `uvicorn backend.app.main:app --port 8000 --workers 4` |
| Start frontend | `python -m http.server 5173 --directory frontend` |
| Run with Docker | `cd docker && docker-compose up -d` |
| Stop Docker | `cd docker && docker-compose down` |
| Check API health | `curl http://localhost:8000/health` |
| View API docs | Open http://localhost:8000/docs |
| Run tests | `pytest backend/tests/ -v` |

---

### Environment Variables

Create a `.env` file in the project root:

```bash
# Application environment
APP_ENV=development          # development, staging, production
LOG_LEVEL=DEBUG              # DEBUG, INFO, WARNING, ERROR

# Server configuration
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Model training defaults
DEFAULT_CV_FOLDS=5
DEFAULT_TUNING_TRIALS=50

# Storage paths
ARTIFACT_DIR=./artifacts
LOG_DIR=./logs

# Resource limits (for large datasets)
OMP_NUM_THREADS=4
OPENBLAS_NUM_THREADS=4
```

---

### Troubleshooting Running Issues

| Problem | Solution |
|---------|----------|
| Port 8000 already in use | `lsof -i :8000` then `kill -9 <PID>` or use `--port 8001` |
| Port 5173 already in use | Use different port: `python -m http.server 5174 --directory frontend` |
| Module not found errors | Ensure venv is activated: `source .venv/bin/activate` |
| Permission denied (Docker) | Run with sudo or add user to docker group |
| Slow startup | First run downloads models; subsequent runs are faster |
| Memory errors | Reduce `n_rows` in data generation config |

---

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.9+ |
| RAM | 8 GB |
| Disk Space | 2 GB |
| OS | macOS, Linux, Windows |

### Recommended Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.11+ |
| RAM | 16 GB+ |
| CPU | 8+ cores |
| Disk Space | 10 GB |

### Dependencies

Core packages:
- fastapi >= 0.110
- pandas >= 2.1
- numpy >= 1.26
- scikit-learn >= 1.3
- xgboost >= 2.0
- lightgbm >= 4.0
- optuna >= 3.5
- shap >= 0.44

---

## Installation

### Standard Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Development Installation

```bash
# Install with dev dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black isort mypy

# Run tests
pytest backend/tests/ -v
```

### Docker Installation

```bash
cd docker
docker-compose up -d

# Check services
docker-compose ps
```

---

## Data Generation

### Overview

The data generator creates realistic financial datasets with configurable characteristics:

- **1000+ features** across multiple categories
- **Class imbalance** (configurable default rate)
- **Missing values** (MCAR and MAR patterns)
- **Correlated features** mimicking real credit bureau data
- **Non-linear target relationships**

### Feature Categories

| Category | Count | Examples |
|----------|-------|----------|
| Demographic | 50 | age, income, employment_tenure, region |
| Financial | 150 | credit_limit, utilization_ratio, debt_to_income |
| Behavioral | 200 | dpd_30_count, payment_history_score, missed_payments |
| Bureau | 150 | external_score, num_tradelines, derogatory_marks |
| Time-based | 100 | vintage_months, time_since_delinq, account_age |
| Derived | 150 | payment_to_income, utilization_trend |
| Interaction | 100 | Feature interactions (A * B) |
| Lag | 100 | Historical values (t-1, t-2, t-3) |
| Noise | 100 | Random features for regularization testing |

### Basic Usage

```python
from backend.app.core.data.generator import FinancialDataGenerator, FinancialDataConfig

# Default configuration (100k rows, 1000+ features)
generator = FinancialDataGenerator()
data = generator.generate()

print(f"Shape: {data.shape}")
print(f"Default rate: {data['target'].mean():.2%}")
```

### Custom Configuration

```python
config = FinancialDataConfig(
    n_rows=50000,           # Number of samples
    n_demographic=30,       # Demographic features
    n_financial=100,        # Financial features
    n_behavioral=150,       # Behavioral features
    default_rate=0.12,      # Target default rate
    missing_rate=0.08,      # Missing value rate
    random_state=42,        # Reproducibility
)

generator = FinancialDataGenerator(config)
data = generator.generate()
```

### Data Splitting

```python
from backend.app.core.data.splitter import DataSplitter, SplitConfig

config = SplitConfig(
    train_ratio=0.60,   # Training set
    test_ratio=0.20,    # Test set
    oot_ratio=0.10,     # Out-of-Time
    etrc_ratio=0.10,    # Economic Through-the-Cycle
    stratify=True,      # Preserve class balance
)

splitter = DataSplitter(config)
splits = splitter.split(data, target_column='target')

# Access splits
X_train, y_train = splits['train']
X_test, y_test = splits['test']
X_oot, y_oot = splits['oot']
X_etrc, y_etrc = splits['etrc']
```

---

## Feature Engineering

### Weight of Evidence (WOE) Encoding

WOE transforms categorical variables into continuous values based on their relationship with the target:

```
WOE = ln(% of Events / % of Non-Events)
```

```python
from backend.app.core.features.encoders import WOEEncoder

encoder = WOEEncoder(min_samples=50, regularization=0.5)
encoder.fit(X_train[categorical_cols], y_train)

# Transform data
X_train_woe = encoder.transform(X_train[categorical_cols])
X_test_woe = encoder.transform(X_test[categorical_cols])

# Get Information Value (IV)
iv_scores = encoder.get_iv()
print("IV Scores:")
for col, iv in sorted(iv_scores.items(), key=lambda x: -x[1]):
    print(f"  {col}: {iv:.4f}")
```

### Information Value (IV) Interpretation

| IV Range | Predictive Power |
|----------|------------------|
| < 0.02 | Useless |
| 0.02 - 0.10 | Weak |
| 0.10 - 0.30 | Medium |
| 0.30 - 0.50 | Strong |
| > 0.50 | Suspicious (check for leakage) |

### Target Encoding

For high-cardinality categoricals:

```python
from backend.app.core.features.encoders import TargetEncoder

encoder = TargetEncoder(
    smoothing=1.0,      # Bayesian smoothing
    cv_folds=5,         # Cross-validation folds
    handle_unknown='global_mean'
)

encoder.fit(X_train['merchant_category'], y_train)
X_train['merchant_encoded'] = encoder.transform(X_train['merchant_category'])
```

### Smart Encoding Pipeline

Automatically selects encoding strategy:

```python
from backend.app.core.features.encoders import SmartEncoder

encoder = SmartEncoder(
    woe_threshold=10,       # Use WOE if cardinality <= 10
    target_threshold=100,   # Use target encoding if cardinality <= 100
    # One-hot for cardinality > 100
)

encoder.fit(X_train, y_train)
X_encoded = encoder.transform(X_train)
```

---

## Model Training

### Supported Models

| Model | Key Hyperparameters | Best For |
|-------|---------------------|----------|
| LogisticRegression | C, penalty, solver | Interpretability, baseline |
| RandomForest | n_estimators, max_depth | Robust performance |
| GradientBoosting | n_estimators, learning_rate | Balanced performance |
| XGBoost | max_depth, colsample_bytree | High performance |
| LightGBM | num_leaves, min_child_samples | Speed, large datasets |

### Basic Training

```python
from backend.app.core.models.trainer import ModelTrainer, TrainingConfig
from backend.app.core.models.definitions import ModelType

config = TrainingConfig(
    model_types=[ModelType.LIGHTGBM, ModelType.XGBOOST, ModelType.LOGISTIC],
    cv_folds=5,
    enable_tuning=False,    # Use default parameters
)

trainer = ModelTrainer(config)
result = trainer.train(X_train, y_train, X_test, y_test)

# View leaderboard
print(result.leaderboard)
```

### Training with Hyperparameter Tuning

```python
config = TrainingConfig(
    model_types=[ModelType.LIGHTGBM, ModelType.XGBOOST],
    cv_folds=5,
    enable_tuning=True,
    tuning_trials=50,       # Optuna trials
    scoring='roc_auc',      # Optimization metric
    save_artifacts=True,    # Save models to disk
)

trainer = ModelTrainer(config)
result = trainer.train(X_train, y_train, X_test, y_test)

# Best model
best_model = result.models[ModelType.LIGHTGBM]
print(f"Best params: {best_model.params}")
print(f"CV AUC: {np.mean(best_model.cv_scores):.4f}")
```

### Quick Tuning

For rapid iteration:

```python
from backend.app.core.models.tuning import quick_tune

result = quick_tune(
    ModelType.LIGHTGBM,
    X_train.values,
    y_train.values,
    n_trials=20,
    cv_folds=3,
)

print(f"Best params: {result.best_params}")
print(f"Best score: {result.best_score:.4f}")
```

### Model Artifact Management

```python
# Save model
import joblib
joblib.dump(trained_model.model, 'model.joblib')

# Load model
model = joblib.load('model.joblib')
predictions = model.predict_proba(X_new)[:, 1]
```

---

## Model Evaluation

### Standard Metrics

```python
from backend.app.core.evaluation.metrics import compute_all_metrics

# Get predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Compute all metrics
metrics = compute_all_metrics(y_test, y_pred, y_proba)

print(f"Accuracy:  {metrics.accuracy:.4f}")
print(f"Precision: {metrics.precision:.4f}")
print(f"Recall:    {metrics.recall:.4f}")
print(f"F1 Score:  {metrics.f1:.4f}")
print(f"ROC-AUC:   {metrics.auc_roc:.4f}")
print(f"KS Stat:   {metrics.ks_statistic:.4f}")
print(f"Gini:      {metrics.gini:.4f}")
```

### KS Statistic

The Kolmogorov-Smirnov statistic measures the maximum separation between cumulative distributions of positive and negative classes:

```python
from backend.app.core.evaluation.metrics import compute_ks_statistic

ks_stat, curve_data = compute_ks_statistic(y_test, y_proba, return_curve=True)
print(f"KS Statistic: {ks_stat:.4f}")

# Plot KS curve
import matplotlib.pyplot as plt
plt.plot(curve_data['thresholds'], curve_data['tpr'], label='TPR')
plt.plot(curve_data['thresholds'], curve_data['fpr'], label='FPR')
plt.fill_between(curve_data['thresholds'],
                  curve_data['tpr'], curve_data['fpr'],
                  alpha=0.3, label=f'KS = {ks_stat:.3f}')
plt.legend()
plt.show()
```

### Gini Coefficient

```python
from backend.app.core.evaluation.metrics import compute_gini

gini = compute_gini(y_test, y_proba)
print(f"Gini Coefficient: {gini:.4f}")
# Gini = 2 * AUC - 1
```

### Cross-Split Stability

```python
# Evaluate on all splits
splits = {
    'train': (X_train, y_train),
    'test': (X_test, y_test),
    'oot': (X_oot, y_oot),
    'etrc': (X_etrc, y_etrc),
}

stability = trainer.evaluate_on_splits(result.models, splits)

# Check for degradation
for split_name, model_metrics in stability.items():
    for model_type, metrics in model_metrics.items():
        print(f"{split_name} - {model_type.value}: AUC={metrics.auc_roc:.4f}, KS={metrics.ks_statistic:.4f}")
```

---

## Explainability & Monitoring

### SHAP Explanations

#### Global Importance

```python
from backend.app.core.explainability.shap_engine import ShapExplainer

explainer = ShapExplainer(model)
explainer.fit(X_train.sample(1000))  # Background data

# Global importance
global_exp = explainer.explain_global(X_test)
print("Top 10 Features:")
for feat, imp in list(global_exp.feature_importance.items())[:10]:
    print(f"  {feat}: {imp:.4f}")
```

#### Local Explanations

```python
# Explain a single prediction
local_exp = explainer.explain_local(X_test.iloc[[0]])
print(f"Prediction: {local_exp.prediction:.4f}")
print("Feature Contributions:")
for feat, contrib in local_exp.contributions.items():
    print(f"  {feat}: {contrib:+.4f}")
```

### Population Stability Index (PSI)

PSI measures drift in score distributions:

```python
from backend.app.core.explainability.psi import compute_psi, PSIMonitor

# Compare two distributions
psi_result = compute_psi(
    expected=train_scores,
    actual=test_scores,
    n_bins=10,
)

print(f"PSI: {psi_result.psi:.4f}")
print(f"Status: {psi_result.interpretation}")

# Continuous monitoring
monitor = PSIMonitor(baseline_scores=train_scores)
current_psi = monitor.check(production_scores)
```

### PSI Interpretation

| PSI Value | Interpretation | Action |
|-----------|----------------|--------|
| < 0.10 | No significant shift | Continue monitoring |
| 0.10 - 0.25 | Moderate shift | Investigate causes |
| > 0.25 | Significant shift | Model retraining recommended |

### Characteristic Stability Index (CSI)

CSI monitors feature-level drift:

```python
from backend.app.core.explainability.csi import CSIMonitor

monitor = CSIMonitor()
monitor.set_baseline(X_train)

# Check current data
csi_result = monitor.check_all(X_production)
print(f"Overall CSI: {csi_result.overall_csi:.4f}")
print(f"Drifting features: {len(csi_result.drifting_features)}")

# Get most drifted features
for feat, csi in sorted(csi_result.feature_csi.items(), key=lambda x: -x[1])[:10]:
    print(f"  {feat}: CSI={csi:.4f}")
```

---

## API Reference

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{"status": "healthy", "version": "1.0.0"}
```

### Train Models

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "n_samples": 10000,
    "n_features": 50,
    "model_types": ["lightgbm", "xgboost"],
    "enable_tuning": true,
    "tuning_trials": 20
  }'
```

Response:
```json
{
  "experiment_id": "abc123",
  "leaderboard": [...],
  "metrics": {...},
  "training_time": 45.2
}
```

### Generate Dataset

```bash
curl -X POST http://localhost:8000/datasets/generate \
  -H "Content-Type: application/json" \
  -d '{
    "n_rows": 100000,
    "n_features": 1000,
    "default_rate": 0.15
  }'
```

### Get SHAP Explanations

```bash
curl http://localhost:8000/explain/shap/model_123?type=global
```

### Get PSI Drift

```bash
curl http://localhost:8000/monitor/psi/model_123
```

---

## Docker Deployment

### Build and Run

```bash
cd docker
docker-compose up -d --build
```

### Service Architecture

```
┌─────────────────┐     ┌─────────────────┐
│    Frontend     │────▶│    Backend      │
│   (port 5173)   │     │   (port 8000)   │
└─────────────────┘     └─────────────────┘
```

### Environment Variables

```bash
# Backend
APP_ENV=production
LOG_LEVEL=INFO
WORKERS=4

# Resource limits
OMP_NUM_THREADS=4
OPENBLAS_NUM_THREADS=4
```

### Scaling

```bash
# Scale backend workers
docker-compose up -d --scale backend=3
```

---

## Configuration

### Application Settings

Create `.env` file:

```bash
APP_ENV=development
LOG_LEVEL=DEBUG
ARTIFACT_DIR=./artifacts
DEFAULT_CV_FOLDS=5
DEFAULT_TUNING_TRIALS=50
```

### Training Presets

```yaml
# configs/training/production.yaml
model_types:
  - lightgbm
  - xgboost
cv_folds: 5
enable_tuning: true
tuning_trials: 100
scoring: roc_auc
save_artifacts: true
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Import errors | Ensure virtual environment is activated |
| Memory errors | Reduce n_rows or process in batches |
| Slow training | Reduce tuning_trials or cv_folds |
| SHAP timeout | Sample fewer background points |

### Memory Optimization

```python
# For large datasets, use chunked processing
config = FinancialDataConfig(
    n_rows=1000000,
    chunk_size=50000,  # Process in chunks
)
```

### Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Best Practices

### Data Preparation

1. **Validate data quality** before training
2. **Handle missing values** appropriately
3. **Check for data leakage** in features
4. **Use stratified splitting** to preserve class balance

### Model Training

1. **Start with simple models** (LogisticRegression) as baseline
2. **Use cross-validation** to avoid overfitting
3. **Monitor training time** vs. performance trade-offs
4. **Save model artifacts** for reproducibility

### Production Deployment

1. **Monitor PSI regularly** (weekly or daily)
2. **Set alerting thresholds** for drift detection
3. **Document model decisions** for audit trails
4. **Version control** all configurations

### Regulatory Compliance

1. **Generate SHAP explanations** for interpretability
2. **Track IV scores** for feature selection rationale
3. **Maintain stability reports** across data splits
4. **Document champion-challenger comparisons**

---

## Support

For issues and feature requests, please refer to:
- **CLAUDE.md**: Development guidelines
- **AGENTS.md**: Agent architecture documentation
- **skills.md**: Skills matrix and competency levels
