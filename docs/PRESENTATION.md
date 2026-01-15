# ML on the Go
## Production-Grade Machine Learning Platform for Financial Risk Analytics

---

# Agenda

1. **Problem Statement** - Why we built this
2. **Solution Overview** - What ML on the Go offers
3. **Architecture** - How it's built
4. **Key Features** - Deep dive into capabilities
5. **Demo Walkthrough** - See it in action
6. **Technical Specifications** - Under the hood
7. **Deployment & Operations** - Production readiness
8. **Roadmap** - What's next

---

# Problem Statement

## Challenges in Financial ML

| Challenge | Impact |
|-----------|--------|
| **Data Complexity** | Credit data has 100s of features with complex relationships |
| **Regulatory Compliance** | SR 11-7, IFRS 9, Basel III require model documentation |
| **Model Interpretability** | Regulators demand explainable decisions |
| **Production Monitoring** | Models degrade over time (concept drift) |
| **Reproducibility** | Audit trails needed for every decision |

---

# Problem Statement (cont.)

## Traditional Approach Issues

- Manual feature engineering is time-consuming
- No standardized metrics for credit risk (KS, Gini)
- Limited drift monitoring capabilities
- Scattered tools without integration
- Difficult to maintain audit trails

**ML on the Go solves these problems with a unified platform.**

---

# Solution Overview

## ML on the Go

A **production-grade ML platform** designed specifically for **financial risk analytics**

### Core Capabilities

| Capability | Description |
|------------|-------------|
| Data Generation | 1000+ feature synthetic datasets |
| Feature Engineering | WOE, target encoding, IV calculation |
| Model Training | Multi-model with Optuna tuning |
| Evaluation | AUC, KS, Gini, stability metrics |
| Explainability | SHAP global & local explanations |
| Monitoring | PSI & CSI drift detection |

---

# Solution Overview (cont.)

## Target Users

- **Data Scientists** - Build and validate credit models
- **Risk Analysts** - Evaluate model performance
- **ML Engineers** - Deploy and monitor production models
- **Compliance Teams** - Generate regulatory documentation

## Use Cases

- Credit scoring & loan approval
- Default probability prediction (PD)
- Portfolio risk assessment
- Model validation & backtesting

---

# Architecture

## High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend Dashboard                        │
│                    (HTML/JS + Plotly Charts)                    │
└────────────────────────────┬────────────────────────────────────┘
                             │ REST API
┌────────────────────────────▼────────────────────────────────────┐
│                      FastAPI Backend                             │
├─────────────┬─────────────┬──────────────┬─────────────────────┤
│    Data     │   Feature   │    Model     │    Explainability   │
│  Generator  │  Engineering│   Training   │    & Monitoring     │
├─────────────┼─────────────┼──────────────┼─────────────────────┤
│  100k rows  │  WOE/Target │ LightGBM     │   SHAP/PSI/CSI     │
│  1000+ feat │  Encoding   │ XGBoost      │                     │
└─────────────┴─────────────┴──────────────┴─────────────────────┘
```

---

# Architecture (cont.)

## Backend Module Structure

```
backend/app/
├── core/
│   ├── data/           # Synthetic data generation
│   │   └── generator.py
│   ├── features/       # Feature engineering
│   │   └── encoders.py # WOE, Target, Smart encoders
│   ├── models/         # Model training
│   │   ├── definitions.py
│   │   ├── tuning.py   # Optuna integration
│   │   └── trainer.py
│   ├── evaluation/     # Metrics computation
│   │   └── metrics.py  # KS, Gini, standard metrics
│   └── explainability/ # Model explanations
│       ├── shap_engine.py
│       ├── psi.py
│       └── csi.py
```

---

# Key Feature 1: Data Generation

## 1000+ Feature Synthetic Data

### Feature Categories

| Category | Features | Examples |
|----------|----------|----------|
| Demographic | 50 | age, income, employment_tenure |
| Financial | 150 | credit_limit, utilization_ratio, DTI |
| Behavioral | 200 | dpd_30_count, payment_history |
| Bureau | 150 | external_score, tradelines |
| Time-based | 100 | vintage_months, account_age |
| Derived | 150 | payment_to_income ratio |
| Interaction | 100 | Feature A * Feature B |
| Lag | 100 | Historical values (t-1, t-2) |

---

# Key Feature 1: Data Generation (cont.)

## Realistic Characteristics

- **Class Imbalance**: Configurable default rate (~15% typical)
- **Missing Values**: MCAR and MAR patterns
- **Correlations**: Realistic feature dependencies
- **Non-linear Effects**: Complex target relationships

### Usage

```python
from backend.app.core.data.generator import FinancialDataGenerator

generator = FinancialDataGenerator()
data = generator.generate()  # 100k rows, 1000+ features
```

---

# Key Feature 2: Feature Engineering

## Weight of Evidence (WOE) Encoding

**Formula:**
```
WOE = ln(% of Goods / % of Bads)
```

**Information Value (IV):**
```
IV = Σ (% Goods - % Bads) × WOE
```

### IV Interpretation

| IV Range | Predictive Power |
|----------|------------------|
| < 0.02 | Useless |
| 0.02 - 0.10 | Weak |
| 0.10 - 0.30 | Medium |
| 0.30 - 0.50 | Strong |
| > 0.50 | Suspicious |

---

# Key Feature 2: Feature Engineering (cont.)

## Available Encoders

| Encoder | Best For | Key Features |
|---------|----------|--------------|
| **WOEEncoder** | Low cardinality categoricals | IV calculation, binning |
| **TargetEncoder** | High cardinality | Bayesian smoothing, CV |
| **SmartEncoder** | Mixed data | Auto-selection based on cardinality |

### Usage

```python
from backend.app.core.features.encoders import WOEEncoder

encoder = WOEEncoder()
encoder.fit(X_train, y_train)
X_woe = encoder.transform(X_train)
iv_scores = encoder.get_iv()
```

---

# Key Feature 3: Model Training

## Supported Models

| Model | Strengths | Use Case |
|-------|-----------|----------|
| LogisticRegression | Interpretable, fast | Regulatory scorecards |
| RandomForest | Robust, handles noise | General purpose |
| GradientBoosting | Good performance | Balanced accuracy/speed |
| **XGBoost** | State-of-the-art | High performance |
| **LightGBM** | Fast, memory efficient | Large datasets |

---

# Key Feature 3: Model Training (cont.)

## Optuna Hyperparameter Tuning

- **Bayesian Optimization** with TPE sampler
- **Pruning** for early stopping of poor trials
- **50 trials default** with 5-fold CV
- **Parameter importance** analysis

### Key Hyperparameters Tuned

```
LightGBM:
- n_estimators: [50, 500]
- learning_rate: [0.01, 0.3]
- num_leaves: [15, 127]
- min_child_samples: [5, 100]
- reg_alpha, reg_lambda: [1e-8, 10]
```

---

# Key Feature 4: Evaluation Metrics

## Financial Risk Metrics

### KS Statistic (Kolmogorov-Smirnov)

**Maximum separation** between cumulative distributions of goods and bads

- **Good model**: KS > 0.3
- **Excellent model**: KS > 0.5

### Gini Coefficient

```
Gini = 2 × AUC - 1
```

- **Good model**: Gini > 0.4
- **Excellent model**: Gini > 0.6

---

# Key Feature 4: Evaluation Metrics (cont.)

## Cross-Split Stability

### Data Splits

| Split | Purpose | % of Data |
|-------|---------|-----------|
| Train | Model fitting | 60% |
| Test | Validation | 20% |
| OOT | Out-of-Time stability | 10% |
| ETRC | Economic cycle stress | 10% |

### Stability Check

Monitor AUC/KS/Gini across all splits
- **Degradation > 10%** = Warning
- **Degradation > 20%** = Action required

---

# Key Feature 5: Explainability

## SHAP Integration

### Global Explanations

- Feature importance ranking
- Average impact on predictions
- Direction of influence

### Local Explanations

- Individual prediction breakdown
- "Why did this customer get rejected?"
- Regulatory requirement for adverse action notices

---

# Key Feature 5: Explainability (cont.)

## SHAP Usage

```python
from backend.app.core.explainability.shap_engine import ShapExplainer

explainer = ShapExplainer(model)
explainer.fit(X_background)

# Global importance
global_exp = explainer.explain_global(X_test)

# Local explanation
local_exp = explainer.explain_local(X_test.iloc[[0]])
```

---

# Key Feature 6: Drift Monitoring

## Population Stability Index (PSI)

Measures shift in **score distributions**

```
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
```

### PSI Thresholds

| PSI | Interpretation | Action |
|-----|----------------|--------|
| < 0.10 | Stable | Continue monitoring |
| 0.10-0.25 | Moderate drift | Investigate |
| > 0.25 | Significant drift | Retrain model |

---

# Key Feature 6: Drift Monitoring (cont.)

## Characteristic Stability Index (CSI)

Measures shift in **feature distributions**

- Identifies which features are drifting
- Early warning before model degradation
- Actionable insights for data engineering

### Usage

```python
from backend.app.core.explainability.psi import PSIMonitor
from backend.app.core.explainability.csi import CSIMonitor

psi_monitor = PSIMonitor(baseline_scores)
csi_monitor = CSIMonitor(baseline_features)

# Check current data
psi_result = psi_monitor.check(current_scores)
csi_result = csi_monitor.check_all(current_features)
```

---

# Demo Walkthrough

## 1. Generate Data

```python
generator = FinancialDataGenerator()
data = generator.generate()
# Shape: (100000, 1000+)
```

## 2. Split Data

```python
splitter = DataSplitter()
splits = splitter.split(data, target_column='target')
```

## 3. Train Models

```python
result = train_models(X_train, y_train, X_test, y_test)
print(result.leaderboard)
```

---

# Demo Walkthrough (cont.)

## Sample Leaderboard Output

| Model | CV AUC | Test AUC | Test KS | Test Gini |
|-------|--------|----------|---------|-----------|
| LightGBM | 0.892 | 0.885 | 0.62 | 0.77 |
| XGBoost | 0.889 | 0.881 | 0.60 | 0.76 |
| LogisticRegression | 0.845 | 0.841 | 0.51 | 0.68 |

## 4. Generate Explanations

```python
explainer = ShapExplainer(best_model)
global_importance = explainer.explain_global(X_test)
```

---

# Technical Specifications

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.9+ | 3.11+ |
| RAM | 8 GB | 16 GB+ |
| CPU | 4 cores | 8+ cores |
| Disk | 2 GB | 10 GB |

## Key Dependencies

- FastAPI, Uvicorn (Web framework)
- pandas, numpy (Data processing)
- scikit-learn, XGBoost, LightGBM (ML)
- Optuna (Tuning)
- SHAP (Explainability)

---

# Deployment

## Docker Support

```bash
cd docker
docker-compose up -d

# Services:
# - Backend: localhost:8000
# - Frontend: localhost:5173
```

## Production Configuration

```yaml
APP_ENV: production
LOG_LEVEL: INFO
WORKERS: 4
OMP_NUM_THREADS: 4
```

---

# Deployment (cont.)

## Multi-stage Docker Build

1. **Builder stage**: Install dependencies, compile extensions
2. **Runtime stage**: Minimal image with only runtime deps
3. **Non-root user**: Security best practice
4. **Health checks**: Automatic container monitoring

### Benefits

- **Smaller image size**: ~500MB vs 2GB+
- **Faster startup**: Pre-compiled dependencies
- **Secure**: Non-root execution
- **Reliable**: Built-in health monitoring

---

# Regulatory Compliance

## SR 11-7 / OCC 2011-12 Alignment

| Requirement | How ML on the Go Addresses |
|-------------|---------------------------|
| Model Documentation | SHAP explanations, IV scores |
| Validation | Cross-split stability metrics |
| Ongoing Monitoring | PSI/CSI drift detection |
| Audit Trail | Artifact versioning, training logs |
| Reproducibility | Deterministic pipelines |

---

# Roadmap

## Current State (v1.0)

- Synthetic data generation
- Feature engineering (WOE, target encoding)
- Multi-model training with Optuna
- Evaluation metrics (KS, Gini)
- SHAP explainability
- PSI/CSI monitoring
- Docker deployment

---

# Roadmap (cont.)

## Planned Enhancements

| Phase | Features |
|-------|----------|
| **v1.1** | React/Next.js frontend migration |
| **v1.2** | Model registry with versioning |
| **v1.3** | Real-time inference API |
| **v2.0** | Feature store integration |
| **v2.1** | Automated retraining pipelines |
| **v2.2** | A/B testing framework |

---

# Summary

## ML on the Go Delivers

| Capability | Value |
|------------|-------|
| **Speed** | Minutes from data to deployed model |
| **Quality** | State-of-the-art models with tuning |
| **Compliance** | Built-in regulatory support |
| **Monitoring** | Proactive drift detection |
| **Scalability** | Docker-ready for production |

---

# Getting Started

## Quick Start Commands

```bash
# 1. Setup environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Start services
uvicorn backend.app.main:app --reload --port 8000

# 3. Open dashboard
open http://localhost:5173
```

## Documentation

- `docs/USER_GUIDE.md` - Comprehensive user guide
- `CLAUDE.md` - Development guidelines
- `AGENTS.md` - Agent architecture
- `skills.md` - Skills matrix

---

# Thank You

## Questions?

### Resources

- **Documentation**: `docs/USER_GUIDE.md`
- **Architecture**: `CLAUDE.md`
- **Agent Design**: `AGENTS.md`

### Contact

For support and feature requests, please refer to the project repository.

---

# Appendix: Code Examples

## Complete Training Pipeline

```python
from backend.app.core.data.generator import FinancialDataGenerator
from backend.app.core.features.encoders import WOEEncoder
from backend.app.core.models.trainer import train_models
from backend.app.core.explainability.shap_engine import ShapExplainer
from backend.app.core.explainability.psi import PSIMonitor

# 1. Generate data
generator = FinancialDataGenerator()
data = generator.generate()

# 2. Prepare features
X = data.drop('target', axis=1)
y = data['target']

# 3. Train models
result = train_models(X, y, model_types=['lightgbm', 'xgboost'])

# 4. Explain best model
best_model = result.models[ModelType.LIGHTGBM].model
explainer = ShapExplainer(best_model)
importance = explainer.explain_global(X)

# 5. Setup monitoring
monitor = PSIMonitor(baseline_scores)
```

---

# Appendix: API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/train` | POST | Train models |
| `/datasets/generate` | POST | Generate data |
| `/datasets/{id}/split` | POST | Split data |
| `/models` | GET | List models |
| `/explain/shap/{id}` | GET | SHAP values |
| `/monitor/psi/{id}` | GET | PSI drift |
| `/monitor/csi/{id}` | GET | CSI drift |

---

# Appendix: Metrics Formulas

## KS Statistic

```
KS = max|TPR(t) - FPR(t)|
```

## Gini Coefficient

```
Gini = 2 × AUC - 1
```

## Information Value

```
IV = Σ (% Goods - % Bads) × ln(% Goods / % Bads)
```

## PSI

```
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
```
