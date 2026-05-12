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

__version__ = "0.0.1"
__author__ = "DNN Team"

import sys
import platform

# Store C++ loading error for diagnostics
_CPP_ERROR = None

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
        Device,
        cuda_available as _cpp_cuda_available,
        save_model,
        load_model,
    )
    # set_cuda_memory_mode / get_cuda_memory_mode were added alongside the
    # CudaMemoryPool managed-memory mode (VRAM<->RAM<->disk paging). They
    # are present on .so artefacts built after that change; if a caller has
    # an older .so we degrade to no-ops so `pydnn.set_cuda_memory_mode(...)`
    # doesn't break the import path.
    try:
        from ._dnn_core import (
            set_cuda_memory_mode as _cpp_set_cuda_memory_mode,
            get_cuda_memory_mode as _cpp_get_cuda_memory_mode,
        )
    except ImportError:
        _cpp_set_cuda_memory_mode = None
        _cpp_get_cuda_memory_mode = None
    # OpenMP thread-count controls were added alongside the SIMD/CUDA perf
    # refactor; older .so builds may not expose them, so degrade gracefully.
    try:
        from ._dnn_core import (
            set_num_threads as _cpp_set_num_threads,
            get_num_threads as _cpp_get_num_threads,
            openmp_available as _cpp_openmp_available,
        )
    except ImportError:
        _cpp_set_num_threads = None
        _cpp_get_num_threads = None
        _cpp_openmp_available = None
    _CPP_AVAILABLE = True
except ImportError as e:
    _CPP_AVAILABLE = False
    _CPP_ERROR = str(e)
    _cpp_set_cuda_memory_mode = None
    _cpp_get_cuda_memory_mode = None
    _cpp_set_num_threads = None
    _cpp_get_num_threads = None
    _cpp_openmp_available = None


def cuda_available() -> bool:
    """
    Check if CUDA is available for GPU acceleration.

    Returns:
        True if CUDA is available, False otherwise.

    Example:
        >>> import pydnn
        >>> if pydnn.cuda_available():
        ...     network = pydnn.DynamicNetwork(..., device="cuda")
        ... else:
        ...     network = pydnn.DynamicNetwork(..., device="cpu")
    """
    if not _CPP_AVAILABLE:
        return False
    return _cpp_cuda_available()


def set_cuda_memory_mode(mode: str) -> None:
    """
    Configure how the CUDA memory pool obtains GPU buffers.

    Args:
        mode: Either ``"device"`` or ``"managed"``.

            - ``"device"`` — ``cudaMalloc``. Pinned to VRAM. Hard OOM if
              VRAM is exhausted. This is the default and matches the
              behaviour of older builds.
            - ``"managed"`` — ``cudaMallocManaged`` (Unified Memory).
              The CUDA driver pages cold pages out to host RAM when VRAM
              is under pressure, and the host page cache will fall back
              to OS swap (disk) if RAM is exhausted. This gives a
              transparent VRAM -> RAM -> disk fallback for models that
              don't fit in VRAM, at the cost of slower memory access
              when the working set exceeds VRAM.

    The mode applies to *future* allocations only. Call this before
    constructing a CUDA network so the network's GPU tensors are
    allocated in the chosen mode.

    No-op when CUDA isn't available (older .so build, or CPU-only build).

    Example:
        >>> import pydnn
        >>> if pydnn.cuda_available():
        ...     pydnn.set_cuda_memory_mode("managed")
        ...     net = pydnn.DynamicNetwork(..., device="cuda")
    """
    if not _CPP_AVAILABLE or _cpp_set_cuda_memory_mode is None:
        return
    _cpp_set_cuda_memory_mode(mode)


def get_cuda_memory_mode() -> str:
    """Return the current CUDA memory pool allocation mode.

    Returns ``"device"`` when CUDA is unavailable or the binding is too
    old to expose this information.
    """
    if not _CPP_AVAILABLE or _cpp_get_cuda_memory_mode is None:
        return "device"
    return _cpp_get_cuda_memory_mode()


def set_num_threads(n: int) -> None:
    """Set the OpenMP thread count used by CPU training.

    Routes to ``omp_set_num_threads`` inside the loaded ``_dnn_core``
    extension. Call this once before ``fit()`` to engage all cores on
    machines where ``OMP_NUM_THREADS`` isn't set in the environment.

    No-op when the loaded ``.so`` was built without OpenMP, or when
    the C++ extension isn't available.

    Example:
        >>> import os, pydnn
        >>> pydnn.set_num_threads(os.cpu_count())
        >>> net = pydnn.DynamicNetwork(..., device="cpu")
    """
    if not _CPP_AVAILABLE or _cpp_set_num_threads is None:
        return
    _cpp_set_num_threads(int(n))


def get_num_threads() -> int:
    """Return the current OpenMP max thread count.

    Returns ``1`` when OpenMP is disabled or the binding is too old to
    expose this information.
    """
    if not _CPP_AVAILABLE or _cpp_get_num_threads is None:
        return 1
    return _cpp_get_num_threads()


def openmp_available() -> bool:
    """True iff the loaded ``_dnn_core`` was built with OpenMP."""
    if not _CPP_AVAILABLE or _cpp_openmp_available is None:
        return False
    return _cpp_openmp_available()

# Pure Python components (always available)
from .network import (
    DynamicNetwork,
    TrainingResult,
    HealthReport,
    # Configuration classes for advanced users
    TrainingPhaseConfig,
    ArchitectureConfig,
    EfficiencyConfig,
    HealthScoreConfig,
    GradientConfig,
    PerturbationConfig,
    EarlyStoppingConfig,
    RewardPenaltyConfig,
    NormalizationConfig,
    SigmoidThresholdConfig,
)

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

# Transformer / MoE / LLM building blocks (pure-Python, NumPy-backed).
# Importing the submodule also makes it accessible as ``pydnn.transformer``.
from . import transformer
from .transformer import (
    TransformerConfig,
    EncoderOnlyModel,
    DecoderOnlyModel,
    Seq2SeqModel,
    Transformer,
    DynamicTransformer,
    MultiHeadAttention,
    GroupedQueryAttention,
    MixtureOfExperts,
    CausalLMTrainer,
    AdamW,
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
    # CUDA support
    "cuda_available",
    "set_cuda_memory_mode",
    "get_cuda_memory_mode",
    # OpenMP / CPU multi-core support
    "set_num_threads",
    "get_num_threads",
    "openmp_available",
    # Configuration classes for advanced ML engineers
    "TrainingPhaseConfig",
    "ArchitectureConfig",
    "EfficiencyConfig",
    "HealthScoreConfig",
    "GradientConfig",
    "PerturbationConfig",
    "EarlyStoppingConfig",
    "RewardPenaltyConfig",
    "NormalizationConfig",
    "SigmoidThresholdConfig",
    # Model generation
    "ThreeModelGenerator",
    "ModelVariant",
    "ModelStrategy",
    "GeneratorConfig",
    "generate_models",
    # Transformer / MoE
    "transformer",
    "TransformerConfig",
    "EncoderOnlyModel",
    "DecoderOnlyModel",
    "Seq2SeqModel",
    "Transformer",
    "DynamicTransformer",
    "MultiHeadAttention",
    "GroupedQueryAttention",
    "MixtureOfExperts",
    "CausalLMTrainer",
    "AdamW",
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
        "Device",
        "save_model",
        "load_model",
    ])


def validate_environment():
    """
    Validate environment for C++ extension compatibility.

    Use this function to debug C++ extension loading issues.
    It returns detailed information about the Python environment
    and C++ extension status.

    Returns:
        Dict with environment info:
        - python_version: Python version string
        - platform: Operating system platform
        - architecture: 32bit or 64bit
        - numpy_version: NumPy version (if available)
        - cpp_available: Whether C++ extensions loaded
        - cpp_error: Error message if C++ failed to load
        - expected_binary: Expected binary file name
        - binary_path: Path where binary should be located

    Example:
        >>> from pydnn import validate_environment
        >>> env = validate_environment()
        >>> if not env['cpp_available']:
        ...     print(f"C++ unavailable: {env['cpp_error']}")
        ...     print(f"Expected: {env['expected_binary']}")
    """
    import os

    # Determine expected binary extension
    if sys.platform == "win32":
        ext = ".pyd"
    else:
        ext = ".so"

    package_dir = os.path.dirname(__file__)
    expected_binary = f"_dnn_core{ext}"

    # Check for numpy
    try:
        import numpy as np
        numpy_version = np.__version__
    except ImportError:
        numpy_version = "NOT INSTALLED"

    # Check for pybind11
    try:
        import pybind11
        pybind11_version = pybind11.__version__
    except ImportError:
        pybind11_version = "NOT INSTALLED"

    return {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": sys.platform,
        "architecture": platform.architecture()[0],
        "machine": platform.machine(),
        "numpy_version": numpy_version,
        "pybind11_version": pybind11_version,
        "cpp_available": _CPP_AVAILABLE,
        "cpp_error": _CPP_ERROR if not _CPP_AVAILABLE else None,
        "expected_binary": expected_binary,
        "binary_path": package_dir,
        "visualization_available": _VIZ_AVAILABLE,
    }


# Add validate_environment to exports
__all__.append("validate_environment")
