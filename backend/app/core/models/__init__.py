# =============================================================================
# ML on the Go - Models Module
# =============================================================================

from .definitions import (
    ModelType,
    ModelDefinition,
    MODEL_REGISTRY,
    get_model_definition,
    get_available_models,
    create_model,
)
from .tuning import (
    TuningConfig,
    TuningResult,
    HyperparameterTuner,
    quick_tune,
)
from .trainer import (
    TrainingConfig,
    TrainedModel,
    TrainingResult,
    ModelTrainer,
    train_models,
)

__all__ = [
    # Definitions
    "ModelType",
    "ModelDefinition",
    "MODEL_REGISTRY",
    "get_model_definition",
    "get_available_models",
    "create_model",
    # Tuning
    "TuningConfig",
    "TuningResult",
    "HyperparameterTuner",
    "quick_tune",
    # Training
    "TrainingConfig",
    "TrainedModel",
    "TrainingResult",
    "ModelTrainer",
    "train_models",
]
