"""
Dynamic Neural Network Python API

A neural network library that automatically adapts its architecture during training.

Features:
- Automatic architecture adaptation (add/remove layers and nodes)
- Cancer/Alzheimer detection for healthy network growth
- Efficiency-based early stopping
- Trainable node scheduling (100% -> 1%)
- Adaptive batch sizing (small -> large)
- Three-model generation (efficient, balanced, accurate)

Example:
    >>> from pydnn import DynamicNetwork
    >>>
    >>> # Create network with just seed, input shape, and output size
    >>> network = DynamicNetwork(
    ...     input_shape=(784,),
    ...     output_size=10,
    ...     seed=42,
    ...     cost_function="CrossEntropy"
    ... )
    >>>
    >>> # Train - all parameters auto-determined
    >>> result = network.fit(X_train, y_train)
    >>>
    >>> # Predict
    >>> predictions = network.predict(X_test)
    >>>
    >>> # Check network health
    >>> health = network.health_status()
    >>> print(f"Health: {health.diagnosis}")

Three-Model Generation:
    >>> from pydnn import generate_models
    >>>
    >>> efficient, balanced, accurate = generate_models(
    ...     X, y,
    ...     seed=42,
    ...     output_dir="./models"
    ... )
"""

__version__ = "1.0.0"
__author__ = "DNN Team"

# Import core components when C++ bindings are available
try:
    from ._dnn_core import (
        Tensor,
        Network,
        Trainer,
        NetworkConfig,
        TrainerConfig,
        TrainingResult as CppTrainingResult,
        HealthReport as CppHealthReport,
        CostFunction,
        Activation,
        HealthState,
        save_model,
        load_model,
    )
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False

# Pure Python components (always available)
from .network import DynamicNetwork, TrainingResult, HealthReport

# Visualization (requires matplotlib or plotly)
try:
    from .visualization import (
        TrainingPlotter,
        NetworkVisualizer,
        HealthTimeline,
        TrainingVisualization,
        auto_generate_plots,
    )
    _VIZ_AVAILABLE = True
except ImportError:
    _VIZ_AVAILABLE = False

# Model generator
from .model_generator import (
    ThreeModelGenerator,
    ModelVariant,
    ModelStrategy,
    GeneratorConfig,
    generate_models,
)

# Available cost functions
COST_FUNCTIONS = [
    "MSE",
    "MAE",
    "CrossEntropy",
    "BinaryCrossEntropy",
    "Huber",
    "LogCosh",
    "KLDivergence",
    "CosineSimilarity",
]

__all__ = [
    # Core
    "DynamicNetwork",
    "TrainingResult",
    "HealthReport",
    # Model generation
    "ThreeModelGenerator",
    "ModelVariant",
    "ModelStrategy",
    "GeneratorConfig",
    "generate_models",
    # Constants
    "COST_FUNCTIONS",
    # Version
    "__version__",
]

# Add visualization if available
if _VIZ_AVAILABLE:
    __all__.extend([
        "TrainingPlotter",
        "NetworkVisualizer",
        "HealthTimeline",
        "TrainingVisualization",
        "auto_generate_plots",
    ])

# Add C++ bindings if available
if _CPP_AVAILABLE:
    __all__.extend([
        "Tensor",
        "Network",
        "Trainer",
        "NetworkConfig",
        "TrainerConfig",
        "CostFunction",
        "Activation",
        "HealthState",
        "save_model",
        "load_model",
    ])
