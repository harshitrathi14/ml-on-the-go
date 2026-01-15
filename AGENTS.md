# Agent-Based Architecture for ML Risk Analytics Platform

This document defines the autonomous agents that power the ML on the Go platform. Each agent has specific responsibilities, interfaces, and failure handling mechanisms.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Orchestration Layer                            │
│                     (API Gateway / Event Bus / Scheduler)                │
└─────────────────────────────────────────────────────────────────────────┘
        │           │           │           │           │           │
        ▼           ▼           ▼           ▼           ▼           ▼
┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐
│   Data    │ │  Feature  │ │   Model   │ │Evaluation │ │Explainab- │ │Deployment │
│ Ingestion │ │Engineering│ │  Training │ │   Agent   │ │  ility    │ │   Agent   │
│   Agent   │ │   Agent   │ │   Agent   │ │           │ │   Agent   │ │           │
└───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────┘
        │           │           │           │           │           │
        └───────────┴───────────┴───────────┴───────────┴───────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │ Monitoring│   │   Model   │   │  Artifact │
            │   Agent   │   │  Registry │   │   Store   │
            └───────────┘   └───────────┘   └───────────┘
```

---

## Data Ingestion Agent

**Purpose**: Validate, transform, and prepare raw data for ML pipelines

### Responsibilities

| Function | Description |
|----------|-------------|
| **Schema Validation** | Verify column names, data types, and required fields |
| **Data Quality Checks** | Detect missing values, outliers, and data anomalies |
| **Format Conversion** | Convert CSV, Parquet, JSON to internal format |
| **Data Splitting** | Create train/test/OOT/ETRC cohorts with stratification |
| **Metadata Generation** | Create data profile with statistics and distributions |

### Inputs

```python
@dataclass
class IngestionRequest:
    source: str  # File path, S3 URI, or database connection
    format: str  # csv, parquet, json
    schema: Optional[DataSchema]  # Expected schema
    target_column: str
    split_config: SplitConfig
    validation_rules: List[ValidationRule]
```

### Outputs

```python
@dataclass
class IngestionResult:
    dataset_id: str
    splits: Dict[str, str]  # Split name -> file path
    profile: DataProfile
    validation_report: ValidationReport
    metadata: Dict[str, Any]
```

### Failure Handling

| Error Type | Action |
|------------|--------|
| Schema mismatch | Return detailed error with column-level diff |
| File not found | Retry with exponential backoff, then fail gracefully |
| Data quality threshold breach | Generate warning report, optionally halt pipeline |
| Memory overflow | Switch to chunked processing |

### Extensibility

- Plugin architecture for new data sources (Snowflake, BigQuery, etc.)
- Custom validation rules via configuration
- Webhook notifications for ingestion events

---

## Feature Engineering Agent

**Purpose**: Transform raw features into ML-ready representations

### Responsibilities

| Function | Description |
|----------|-------------|
| **Encoding** | WOE, target encoding, one-hot for categoricals |
| **Scaling** | StandardScaler, MinMax, RobustScaler for numerics |
| **Imputation** | Handle missing values with configurable strategies |
| **Feature Selection** | IV-based selection, correlation filtering |
| **Transformation Tracking** | Log all transformations for reproducibility |

### Inputs

```python
@dataclass
class FeatureEngineeringRequest:
    dataset_id: str
    target_column: str
    feature_config: FeatureConfig
    encoding_strategy: str  # woe, target, onehot, smart
    scaling_strategy: str  # standard, minmax, robust
    imputation_strategy: str  # mean, median, mode, knn
    selection_config: Optional[SelectionConfig]
```

### Outputs

```python
@dataclass
class FeatureEngineeringResult:
    pipeline_id: str
    feature_names: List[str]
    dropped_features: List[str]
    iv_scores: Dict[str, float]
    transformation_log: List[TransformationStep]
    pipeline_artifact: str  # Serialized sklearn Pipeline
```

### Failure Handling

| Error Type | Action |
|------------|--------|
| Unsupported dtype | Fall back to passthrough or raise with guidance |
| High cardinality | Auto-switch to target encoding |
| All NaN column | Drop with warning, log to report |
| Memory issues | Process features in batches |

### Extensibility

- Custom transformer plugins
- Feature store integration
- Domain-specific feature libraries (e.g., credit bureau features)

---

## Model Training Agent

**Purpose**: Train, tune, and validate ML models

### Responsibilities

| Function | Description |
|----------|-------------|
| **Model Training** | Fit multiple model types in parallel |
| **Hyperparameter Tuning** | Optuna-based Bayesian optimization |
| **Cross-Validation** | K-fold CV with stratification |
| **Early Stopping** | Prevent overfitting with validation monitoring |
| **Artifact Management** | Save models, configs, and training logs |

### Inputs

```python
@dataclass
class TrainingRequest:
    dataset_id: str
    pipeline_id: str
    model_types: List[str]  # logistic, xgboost, lightgbm, etc.
    tuning_config: TuningConfig
    cv_folds: int
    random_state: int
```

### Outputs

```python
@dataclass
class TrainingResult:
    experiment_id: str
    models: List[TrainedModel]
    cv_scores: Dict[str, List[float]]
    best_params: Dict[str, Dict]
    training_time: float
    leaderboard: pd.DataFrame
```

### Failure Handling

| Error Type | Action |
|------------|--------|
| Convergence failure | Log warning, reduce iterations, retry |
| OOM during training | Reduce batch size, enable data sampling |
| Tuning timeout | Return best params found so far |
| Model divergence | Early stop, flag for review |

### Extensibility

- Custom model definitions via registry
- Distributed training backends (Ray, Dask)
- Custom loss functions and metrics

---

## Evaluation Agent

**Purpose**: Compute comprehensive model metrics and comparisons

### Responsibilities

| Function | Description |
|----------|-------------|
| **Metric Computation** | AUC, KS, Gini, precision, recall, F1 |
| **Stability Analysis** | Compare metrics across train/test/OOT/ETRC |
| **Threshold Optimization** | Find optimal decision threshold |
| **Model Comparison** | Rank models, generate leaderboard |
| **Visualization Data** | Prepare ROC, KS, lift, gains curves |

### Inputs

```python
@dataclass
class EvaluationRequest:
    experiment_id: str
    model_ids: List[str]
    splits: List[str]  # train, test, oot, etrc
    metrics: List[str]
    threshold_strategy: str  # f1_optimal, ks_optimal, fixed
```

### Outputs

```python
@dataclass
class EvaluationResult:
    model_metrics: Dict[str, Dict[str, ModelMetrics]]
    stability_report: StabilityReport
    leaderboard: pd.DataFrame
    curve_data: Dict[str, Dict]  # ROC, KS, lift, gains
    recommendations: List[str]
```

### Failure Handling

| Error Type | Action |
|------------|--------|
| Missing predictions | Skip split, log warning |
| Single class in split | Return partial metrics with warning |
| NaN in predictions | Impute with 0.5 or flag for investigation |
| Metric computation error | Return None for failed metric, continue others |

### Extensibility

- Custom metric definitions
- Business-specific thresholds
- Fairness and bias metrics

---

## Explainability Agent

**Purpose**: Generate model explanations for governance and debugging

### Responsibilities

| Function | Description |
|----------|-------------|
| **Global Importance** | SHAP-based feature ranking |
| **Local Explanations** | Individual prediction explanations |
| **PSI Monitoring** | Score distribution stability |
| **CSI Monitoring** | Feature distribution drift |
| **Report Generation** | Regulatory-ready explanation documents |

### Inputs

```python
@dataclass
class ExplainabilityRequest:
    model_id: str
    dataset_id: str
    explanation_type: str  # global, local, drift
    sample_indices: Optional[List[int]]  # For local explanations
    baseline_dataset_id: Optional[str]  # For drift analysis
```

### Outputs

```python
@dataclass
class ExplainabilityResult:
    global_importance: Optional[Dict[str, float]]
    local_explanations: Optional[List[LocalExplanation]]
    psi_report: Optional[PSIResult]
    csi_report: Optional[CSIResult]
    visualization_data: Dict[str, Any]
```

### Failure Handling

| Error Type | Action |
|------------|--------|
| SHAP computation timeout | Sample fewer data points, warn user |
| Incompatible model type | Fall back to permutation importance |
| Missing baseline data | Skip drift analysis, return available results |
| Memory overflow | Compute in batches |

### Extensibility

- Additional explainability methods (LIME, partial dependence)
- Custom drift detectors
- Regulatory report templates

---

## Deployment Agent

**Purpose**: Manage model lifecycle from staging to production

### Responsibilities

| Function | Description |
|----------|-------------|
| **Model Registration** | Version control with metadata |
| **Stage Promotion** | Move models through staging → production |
| **A/B Testing** | Champion-challenger deployments |
| **Rollback** | Revert to previous versions |
| **Inference Serving** | Batch and real-time predictions |

### Inputs

```python
@dataclass
class DeploymentRequest:
    model_id: str
    target_stage: str  # staging, production
    deployment_config: DeploymentConfig
    rollback_on_failure: bool
    validation_checks: List[str]
```

### Outputs

```python
@dataclass
class DeploymentResult:
    deployment_id: str
    endpoint_url: Optional[str]
    status: str  # deployed, failed, pending
    version: str
    validation_results: Dict[str, bool]
    rollback_version: Optional[str]
```

### Failure Handling

| Error Type | Action |
|------------|--------|
| Validation failure | Block deployment, return detailed failure report |
| Endpoint creation failure | Retry with backoff, alert on-call |
| Performance degradation | Auto-rollback if configured |
| Model loading error | Verify artifact integrity, retry |

### Extensibility

- Custom deployment targets (SageMaker, Vertex, custom K8s)
- Blue-green and canary deployments
- Custom validation checks

---

## Monitoring Agent

**Purpose**: Continuous model and data quality monitoring

### Responsibilities

| Function | Description |
|----------|-------------|
| **Performance Tracking** | Monitor AUC, KS over time |
| **Drift Detection** | PSI/CSI alerts for score and feature drift |
| **Data Quality** | Track missing rates, outliers, schema changes |
| **Alerting** | Threshold-based alerts to Slack, email, PagerDuty |
| **Reporting** | Scheduled performance reports |

### Inputs

```python
@dataclass
class MonitoringConfig:
    model_id: str
    metrics: List[str]
    psi_threshold: float
    csi_threshold: float
    alert_channels: List[str]
    schedule: str  # Cron expression
```

### Outputs

```python
@dataclass
class MonitoringResult:
    timestamp: datetime
    metrics: Dict[str, float]
    drift_status: Dict[str, str]
    alerts_triggered: List[Alert]
    recommendations: List[str]
```

### Failure Handling

| Error Type | Action |
|------------|--------|
| Missing production data | Alert data engineering, skip cycle |
| Metric computation failure | Log error, use last known value |
| Alert delivery failure | Retry with backoff, escalate |
| Database connection failure | Buffer results, retry |

### Extensibility

- Custom monitoring metrics
- Integration with external monitoring (Datadog, New Relic)
- Automated retraining triggers

---

## Agent Communication Protocol

### Message Format

```python
@dataclass
class AgentMessage:
    message_id: str
    source_agent: str
    target_agent: str
    action: str
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: str  # For tracing
    retry_count: int
```

### Event Types

| Event | Producer | Consumers |
|-------|----------|-----------|
| `data.ingested` | Data Ingestion | Feature Engineering, Monitoring |
| `features.ready` | Feature Engineering | Model Training |
| `model.trained` | Model Training | Evaluation, Deployment |
| `model.evaluated` | Evaluation | Deployment, Monitoring |
| `model.deployed` | Deployment | Monitoring |
| `drift.detected` | Monitoring | All Agents (for awareness) |

---

## Quality Standards

All agents must adhere to:

1. **Reproducibility**: All operations must be deterministic given the same inputs
2. **Auditability**: Complete logging of inputs, outputs, and decisions
3. **Idempotency**: Safe to retry failed operations
4. **Graceful Degradation**: Partial failures should not crash the pipeline
5. **Documentation**: Clear API contracts and error messages
