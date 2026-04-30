"""
High-level Python interface for Dynamic Neural Network.
"""

from typing import Tuple, List, Optional, Callable, Dict, Any
from dataclasses import dataclass, field
import warnings
import sys
import os
import platform
import glob
import numpy as np


class CppLoadError:
    """Diagnostic error types for C++ extension loading failures."""
    MISSING_BINARY = "missing_binary"
    ARCH_MISMATCH = "architecture_mismatch"
    PYTHON_VERSION_MISMATCH = "python_version_mismatch"
    PERMISSION_DENIED = "permission_denied"
    MISSING_DEPENDENCIES = "missing_dependencies"
    CORRUPTED_BINARY = "corrupted_binary"
    UNKNOWN = "unknown"


def _detect_cpp_status() -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Detect C++ extension status with detailed diagnostics.

    Returns:
        Tuple of (success, error_type, error_message)
        - success: True if C++ extension loaded successfully
        - error_type: One of CppLoadError constants if failed, None if success
        - error_message: Detailed error message if failed, None if success
    """
    # Determine expected binary extension
    package_dir = os.path.dirname(__file__)
    if sys.platform == "win32":
        ext = ".pyd"
    else:
        ext = ".so"

    # Find potential binary files with various naming patterns
    binary_patterns = [
        f"_dnn_core{ext}",
        f"_dnn_core.cpython-{sys.version_info.major}{sys.version_info.minor}*{ext}",
        f"_dnn_core.cp{sys.version_info.major}{sys.version_info.minor}*{ext}",
    ]

    binary_found = False
    binary_path = None
    for pattern in binary_patterns:
        matches = glob.glob(os.path.join(package_dir, pattern))
        if matches:
            binary_found = True
            binary_path = matches[0]
            break

    if not binary_found:
        return False, CppLoadError.MISSING_BINARY, (
            f"C++ binary not found in {package_dir}.\n"
            f"Expected: _dnn_core{ext}"
        )

    # Check file permissions
    if not os.access(binary_path, os.R_OK):
        return False, CppLoadError.PERMISSION_DENIED, (
            f"Cannot read C++ binary: {binary_path}\n"
            "Check file permissions."
        )

    # Get architecture info for diagnostics
    python_arch = platform.architecture()[0]  # '64bit' or '32bit'

    # Try to import and catch specific errors
    try:
        from . import _dnn_core
        return True, None, None
    except ImportError as e:
        error_msg = str(e).lower()

        if "dll load failed" in error_msg or "cannot open shared object" in error_msg:
            return False, CppLoadError.MISSING_DEPENDENCIES, (
                f"Missing runtime dependencies: {e}\n"
                "On Windows: Install Visual C++ Redistributable\n"
                "On Linux: Check ldd output for missing libraries"
            )
        elif "incompatible" in error_msg or "version" in error_msg:
            return False, CppLoadError.PYTHON_VERSION_MISMATCH, (
                f"Python version mismatch: {e}\n"
                f"Binary may be for different Python version.\n"
                f"Current: Python {sys.version_info.major}.{sys.version_info.minor}"
            )
        elif "32-bit" in error_msg or "64-bit" in error_msg or "x86" in error_msg:
            return False, CppLoadError.ARCH_MISMATCH, (
                f"Architecture mismatch: {e}\n"
                f"Python is {python_arch}, binary may be different."
            )
        else:
            return False, CppLoadError.UNKNOWN, f"Import failed: {e}"
    except OSError as e:
        return False, CppLoadError.CORRUPTED_BINARY, (
            f"Binary may be corrupted: {e}\n"
            "Try reinstalling the package."
        )

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: tqdm not installed. Install with 'pip install tqdm' for progress bars.")


@dataclass
class TrainingResult:
    """Result of training with comprehensive diagnostics."""
    success: bool
    epochs_completed: int
    final_cost: float
    final_efficiency: float
    best_cost: float
    best_efficiency: float
    stopping_reason: str
    cost_history: List[float]
    efficiency_history: List[float]
    training_time_ms: int
    # New diagnostic fields
    nodes_added: int = 0
    nodes_removed: int = 0
    layers_added: int = 0
    layers_removed: int = 0
    cancer_score_history: List[float] = field(default_factory=list)
    alzheimer_score_history: List[float] = field(default_factory=list)
    architecture_history: List[Tuple[int, int]] = field(default_factory=list)
    perturbations_applied: int = 0
    phase_metrics: Dict[str, Any] = field(default_factory=dict)
    # Emotional state fields
    total_rewards: int = 0
    total_penalties: int = 0
    depression_history: List[float] = field(default_factory=list)
    excitement_history: List[float] = field(default_factory=list)
    lr_reset_count: int = 0
    learning_rate_history: List[float] = field(default_factory=list)
    emotional_state_history: List[str] = field(default_factory=list)
    # Phase 4 fields
    phase4_epochs: int = 0
    phase4_initial_cost: float = 0.0
    phase4_final_cost: float = 0.0
    phase4_cost_reduction: float = 0.0
    phase4_early_stopped: bool = False


@dataclass
class HealthReport:
    """Network health report."""
    state: str
    cancer_score: float
    alzheimer_score: float
    overall_health: float
    diagnosis: str
    recommendations: List[str]
    current_layers: int
    current_nodes: int
    depression_ratio: float = 0.0
    excitement_ratio: float = 0.0
    emotional_state: str = "neutral"


@dataclass
class EmotionalState:
    """Tracks the network's emotional state during training."""
    total_rewards: int = 0
    total_penalties: int = 0
    reward_history: List[float] = field(default_factory=list)
    penalty_history: List[float] = field(default_factory=list)
    depression_history: List[float] = field(default_factory=list)
    excitement_history: List[float] = field(default_factory=list)
    lr_reset_count: int = 0

    @property
    def total_adjustments(self) -> int:
        return self.total_rewards + self.total_penalties

    @property
    def depression_ratio(self) -> float:
        """Ratio of penalties to total adjustments."""
        if self.total_adjustments == 0:
            return 0.0
        return self.total_penalties / self.total_adjustments

    @property
    def excitement_ratio(self) -> float:
        """Ratio of rewards to total adjustments."""
        if self.total_adjustments == 0:
            return 0.0
        return self.total_rewards / self.total_adjustments


@dataclass
class RewardPenaltyConfig:
    """Configuration for the reward/penalty system."""
    cost_improvement_threshold: float = 0.005    # 0.5% threshold (was 0.001)
    efficiency_improvement_threshold: float = 0.02  # 2% threshold (was 0.01)
    min_learning_rate: float = 1e-5              # Increased (was 1e-6)
    max_learning_rate: float = 0.5               # Reduced (was 1.0)
    baseline_learning_rate: float = 0.1
    max_adjustment_factor: float = 1.5           # More conservative (was 2.0)
    min_adjustment_factor: float = 0.7           # More conservative (was 0.5)
    extreme_threshold: float = 0.7               # Earlier intervention (was 0.8)
    moderate_threshold: float = 0.6              # NEW: Partial intervention threshold
    window_size: int = 15                        # Larger window (was 10)


@dataclass
class NormalizationConfig:
    """Configuration for data normalization."""
    # Whether to auto-normalize input data
    normalize_input: bool = True
    # Whether to auto-normalize output data (for regression)
    normalize_output: bool = True
    # Normalization method: 'zscore' (mean=0, std=1) or 'minmax' (0-1 range)
    method: str = 'zscore'
    # Small constant to prevent division by zero
    epsilon: float = 1e-8


@dataclass
class TrainingPhaseConfig:
    """Configuration for the 4-phase training approach."""
    # Phase 1: Exploration
    exploration_epochs: int = 10
    exploration_learning_rate: float = 0.05       # Reduced from 0.2 to prevent gradient explosion
    exploration_batch_size: int = 64
    exploration_saturation_threshold: float = 0.6  # Increased from 0.3
    exploration_efficiency_threshold: float = 0.2  # Lowered from 0.3 for less removal

    # Phase 2: Estimation
    estimation_epochs: int = 10
    estimation_learning_rate: float = 0.01        # Reduced from 0.1 for stability

    # Phase 3: Main Training
    main_learning_rate: float = 0.01              # Reduced from 0.1 to prevent divergence
    main_initial_batch_size: int = 32
    main_max_batch_size: int = 256
    batch_size_growth_interval: int = 25
    perturbation_cutoff_ratio: float = 0.2  # First 20% of epochs

    # Epoch estimation (Phase 3)
    target_efficiency: float = 0.8               # Reduced from 0.9
    min_estimated_epochs: int = 50               # Increased from 10
    max_estimated_epochs: int = 300              # Reduced from 500

    # Phase 4: Standard Training (frozen architecture, accuracy focus)
    phase4_enabled: bool = True
    phase4_learning_rate: float = 0.005          # Reduced from 0.01 for fine-tuning stability
    phase4_min_learning_rate: float = 0.0001     # Lower bound for LR
    phase4_lr_decay_rate: float = 0.95           # Gentle decay
    phase4_lr_decay_interval: int = 10           # Epochs between decay
    phase4_batch_size: int = 64                  # Fixed batch size
    phase4_target_cost_reduction: float = 0.5    # Target 50% additional cost reduction
    phase4_min_epochs: int = 50                  # Increased from 20 for more training time
    phase4_max_epochs: int = 200                 # Maximum training epochs
    phase4_patience: int = 35                    # Early stopping patience (increased from 15)
    phase4_min_improvement: float = 1e-5         # Minimum cost improvement threshold


@dataclass
class ArchitectureConfig:
    """Configuration for dynamic architecture constraints."""
    # Node limits
    min_nodes_per_layer: int = 32            # Increased from 16 to prevent under-capacity
    max_nodes_per_layer: int = 1024          # Reduced (was 2000)
    min_nodes_to_keep: int = 16              # Increased from 8

    # Layer limits
    max_layers: int = 6                      # Reduced (was 10)
    min_layers: int = 1

    # Growth rates - significantly reduced for stability
    exploration_growth_rate: float = 0.10    # 10% (was 25%)
    main_growth_rate: float = 0.05           # 5% (was 12.5%)

    # Initial sizing
    min_initial_hidden_size: int = 64        # Increased from 32 for larger initial capacity


@dataclass
class EfficiencyConfig:
    """Configuration for efficiency computation and thresholds."""
    # Saturation thresholds
    default_saturation_threshold: float = 0.7
    exploration_saturation_threshold: float = 0.3

    # Efficiency thresholds
    default_efficiency_threshold: float = 0.5
    exploration_efficiency_threshold: float = 0.3
    removal_efficiency_multiplier: float = 0.5  # threshold * 0.5 for removal

    # Initial values
    initial_efficiency: float = 0.5

    # Efficiency update
    efficiency_decay: float = 0.9
    efficiency_update_scale: float = 0.1
    efficiency_gradient_multiplier: float = 10.0
    max_removal_fraction_per_epoch: float = 0.25  # max 25% of layer removed per epoch


@dataclass
class EfficiencyWeights:
    """
    Adaptive weights for node efficiency score computation.
    Weights adapt based on network behavior and are constrained to sum to 1.

    Efficiency score: η = w_var*S_var + w_grad*S_grad + w_alive*S_alive + w_contrib*S_contrib
    """
    w_variance: float = 0.25
    w_gradient: float = 0.30
    w_alive: float = 0.25       # Static, never changes
    w_contribution: float = 0.20
    grad_threshold: float = 0.1  # Auto-detected from data

    def normalize(self) -> None:
        """Normalize weights to sum to 1.0"""
        total = self.w_variance + self.w_gradient + self.w_alive + self.w_contribution
        if total > 0:
            self.w_variance /= total
            self.w_gradient /= total
            self.w_alive /= total
            self.w_contribution /= total

    def adapt_variance_from_zscore(self, z: float) -> None:
        """
        Adapt variance weight based on z-score.
        High z-score (high variance) -> lower weight (already good)
        Low z-score (low variance) -> higher weight (need improvement)
        """
        if z > 1.0:
            self.w_variance = 0.15  # High variance already, less emphasis
        elif z < -1.0:
            self.w_variance = 0.35  # Low variance, more emphasis needed
        else:
            self.w_variance = 0.25  # Normal range
        self.normalize()

    def adapt_gradient_contribution(self, grad_mag: float) -> None:
        """
        Adapt gradient/contribution weights based on gradient magnitude.
        High gradients -> lower gradient weight, higher contribution weight.
        """
        if grad_mag > self.grad_threshold:
            excess = (grad_mag - self.grad_threshold) / self.grad_threshold
            transfer = min(0.15, excess * 0.10)

            self.w_gradient = max(0.10, 0.30 - transfer)
            self.w_contribution = min(0.40, 0.20 + transfer)
            self.normalize()

    def get_weights(self) -> Tuple[float, float, float, float]:
        """Get current weights as tuple (var, grad, alive, contrib)."""
        return (self.w_variance, self.w_gradient, self.w_alive, self.w_contribution)

    def copy(self) -> 'EfficiencyWeights':
        """Create a copy of these weights."""
        return EfficiencyWeights(
            w_variance=self.w_variance,
            w_gradient=self.w_gradient,
            w_alive=self.w_alive,
            w_contribution=self.w_contribution,
            grad_threshold=self.grad_threshold
        )


@dataclass
class SigmoidThresholdConfig:
    """Configuration for adaptive sigmoid threshold computation."""
    k: float = 5.0
    base: float = 0.3
    range_val: float = 0.5
    center: float = 0.5  # Efficiency center point


@dataclass
class HealthScoreConfig:
    """Configuration for cancer/alzheimer health score computation."""
    # Denominators for normalization (rate-based)
    cancer_denominator: float = 5.0
    alzheimer_denominator: float = 10.0

    # Layer weight in scoring
    layer_weight: int = 10

    # Health state thresholds
    healthy_threshold: float = 0.3
    at_risk_threshold: float = 0.7

    # Ratio-based thresholds (percentage of initial nodes)
    # If more than this ratio of nodes are removed, trigger high Alzheimer score
    alzheimer_ratio_threshold: float = 0.8  # 80% node removal triggers concern
    # If more than this ratio of nodes are added, trigger high Cancer score
    cancer_ratio_threshold: float = 2.0  # 200% growth (3x original size) triggers concern


@dataclass
class GradientConfig:
    """Configuration for gradient handling and optimization."""
    gradient_clip_value: float = 1.0         # Reduced from 5.0 for more aggressive clipping
    momentum: float = 0.9


@dataclass
class PerturbationConfig:
    """Configuration for random perturbation."""
    perturbation_fraction: float = 0.005
    perturbation_scale: float = 0.01


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping criteria."""
    window_size: int = 30                    # Increased from 20 for less aggressive stopping
    improvement_threshold: float = 1e-7      # Reduced from 1e-6 for less sensitivity
    max_consecutive_increases: int = 5


class _CostTrendObserver:
    """
    Parallel cost-trend observer for the pure-Python fit().

    Mirrors the C++ StageController's observer: runs as a daemon thread
    that periodically reads the shared cost_history list, computes a
    rolling improvement signal over `window` samples, and raises a
    rewind flag when improvement falls below `stall_delta`. The
    training loop checks rewind_requested between epochs and bails out
    of its inner loop so the controller can wake an earlier stage.

    Thread-safety note: Python's GIL makes single list.append() and
    list-index reads atomic, which is enough for this read-only
    observer. We only ever **read** cost_history; we don't mutate it.
    """

    def __init__(self, window: int = 12, stall_delta: float = 1e-5,
                 poll_interval: float = 0.05):
        import threading
        self._cost_history = []  # Replaced by shared list at start()
        self._window = window
        self._stall_delta = stall_delta
        self._poll_interval = poll_interval
        self._stop = threading.Event()
        self._rewind = threading.Event()
        self._thread = None

    def start(self, cost_history):
        """Start the observer reading from `cost_history` (a list)."""
        import threading
        self._cost_history = cost_history
        self._stop.clear()
        self._rewind.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="dnn-cost-observer")
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._thread = None

    def rewind_requested(self) -> bool:
        return self._rewind.is_set()

    def clear_rewind(self):
        self._rewind.clear()

    def _loop(self):
        while not self._stop.is_set():
            self._stop.wait(self._poll_interval)
            if self._stop.is_set():
                return
            history = self._cost_history
            if len(history) < self._window:
                continue
            # Take a snapshot of the tail (each index read is atomic).
            tail = history[-self._window:]
            if len(tail) < self._window:
                continue
            improvement = tail[0] - tail[-1]
            if improvement < self._stall_delta:
                self._rewind.set()


class DynamicNetwork:
    """
    Dynamic Neural Network with automatic architecture adaptation.

    The network automatically adjusts its structure during training:
    - Adds layers/nodes when capacity is saturated
    - Removes layers/nodes when they're inefficient
    - Monitors for pathological growth (cancer) or pruning (alzheimer)
    - Gradually reduces trainable nodes from 100% to 1%
    - Uses adaptive batch sizing (small -> large)

    Only required hyperparameters:
    - seed: Random seed for reproducibility
    - input_shape: Shape of input data
    - output_size: Number of output neurons
    - cost_function: Loss function to use

    Example:
        >>> network = DynamicNetwork(
        ...     input_shape=(784,),
        ...     output_size=10,
        ...     seed=42,
        ...     cost_function="CrossEntropy"
        ... )
        >>> result = network.fit(X_train, y_train)
        >>> predictions = network.predict(X_test)
    """

    # Available cost functions
    COST_FUNCTIONS = {
        "MSE": "Mean Squared Error",
        "MAE": "Mean Absolute Error",
        "CrossEntropy": "Cross-Entropy (multi-class)",
        "BinaryCrossEntropy": "Binary Cross-Entropy",
        "Huber": "Huber Loss",
        "LogCosh": "Log-Cosh Loss",
        "KLDivergence": "KL Divergence",
        "CosineSimilarity": "Cosine Similarity"
    }

    def __init__(self,
                 input_shape: Tuple[int, ...],
                 output_size: int,
                 seed: int,
                 cost_function: str = "CrossEntropy",
                 device: str = "cpu",
                 # Optional configuration objects
                 training_phase: Optional[TrainingPhaseConfig] = None,
                 architecture: Optional[ArchitectureConfig] = None,
                 efficiency: Optional[EfficiencyConfig] = None,
                 sigmoid_threshold: Optional[SigmoidThresholdConfig] = None,
                 health_score: Optional[HealthScoreConfig] = None,
                 gradient: Optional[GradientConfig] = None,
                 perturbation: Optional[PerturbationConfig] = None,
                 early_stopping: Optional[EarlyStoppingConfig] = None,
                 reward_penalty: Optional[RewardPenaltyConfig] = None,
                 normalization: Optional[NormalizationConfig] = None,
                 runtime_enabled: bool = False,
                 dynamic_thresholds: bool = True):
        """
        Initialize a Dynamic Neural Network.

        Args:
            input_shape: Shape of input data (excluding batch dimension)
            output_size: Number of output neurons
            seed: Random seed for reproducibility
            cost_function: One of the available cost functions
            device: Device to run on ("cpu" or "cuda")
            training_phase: Configuration for training phases (exploration, estimation, main)
            architecture: Configuration for dynamic architecture constraints
            efficiency: Configuration for efficiency computation and thresholds
            sigmoid_threshold: Configuration for adaptive sigmoid threshold
            health_score: Configuration for cancer/alzheimer health scores
            gradient: Configuration for gradient handling
            perturbation: Configuration for random perturbation
            early_stopping: Configuration for early stopping criteria
            reward_penalty: Configuration for the reward/penalty system
            normalization: Configuration for automatic data normalization
            runtime_enabled: Opt into the concurrent StageController
                pipeline (parallel Estimation observer + soft topology +
                adaptive scalars). C++ backend only; the pure-Python
                fallback ignores this flag. Default False.
            dynamic_thresholds: Derive cancer/alzheimer/patience/etc.
                from the training dataset's variance and complexity
                instead of using static defaults. C++ backend only;
                the pure-Python fallback ignores this flag. Default
                True. Set to False to reproduce legacy static-default
                behaviour bit-for-bit. The derived signals and final
                threshold dict are stashed on
                ``self._last_data_signals`` after each fit().
        """
        self.runtime_enabled = runtime_enabled
        self.dynamic_thresholds = dynamic_thresholds
        self._last_data_signals = None
        # Validate device
        device = device.lower()
        if device not in ("cpu", "cuda", "gpu"):
            raise ValueError(f"Unknown device: {device}. Use 'cpu' or 'cuda'")
        if device == "gpu":
            device = "cuda"
        self._device = device

        if cost_function not in self.COST_FUNCTIONS:
            raise ValueError(f"Unknown cost function: {cost_function}. "
                           f"Available: {list(self.COST_FUNCTIONS.keys())}")

        self.input_shape = input_shape
        self.output_size = output_size
        self.seed = seed
        self.cost_function = cost_function

        # Store configurations (use defaults if not provided)
        self.training_phase = training_phase or TrainingPhaseConfig()
        self.architecture = architecture or ArchitectureConfig()
        self.efficiency = efficiency or EfficiencyConfig()
        self.sigmoid_threshold = sigmoid_threshold or SigmoidThresholdConfig()
        self.health_score = health_score or HealthScoreConfig()
        self.gradient = gradient or GradientConfig()
        self.perturbation = perturbation or PerturbationConfig()
        self.early_stopping = early_stopping or EarlyStoppingConfig()
        self.reward_penalty = reward_penalty or RewardPenaltyConfig()
        self.normalization = normalization or NormalizationConfig()

        # Internal state
        self._trained = False
        self._training_result: Optional[TrainingResult] = None
        self._layers: List[Dict] = []
        self._cpp_error_type: Optional[str] = None
        self._cpp_error_msg: Optional[str] = None

        # Normalization parameters (computed during fit, used in predict)
        self._input_mean: Optional[np.ndarray] = None
        self._input_std: Optional[np.ndarray] = None
        self._input_min: Optional[np.ndarray] = None
        self._input_max: Optional[np.ndarray] = None
        self._output_mean: Optional[np.ndarray] = None
        self._output_std: Optional[np.ndarray] = None
        self._output_min: Optional[np.ndarray] = None
        self._output_max: Optional[np.ndarray] = None

        # Track initial node count for health scoring
        self._initial_node_count: Optional[int] = None

        # Try to use C++ backend with detailed diagnostics
        success, error_type, error_msg = _detect_cpp_status()

        if success:
            self._use_cpp = True
            self._init_cpp_network()
        else:
            self._cpp_error_type = error_type
            self._cpp_error_msg = error_msg

            # Attempt auto-recovery for missing binary in dev environment
            if error_type == CppLoadError.MISSING_BINARY:
                if self._attempt_cpp_recovery():
                    self._use_cpp = True
                    self._cpp_error_type = None
                    self._cpp_error_msg = None
                    self._init_cpp_network()
                else:
                    self._use_cpp = False
                    self._init_python_network(error_type, error_msg)
            else:
                self._use_cpp = False
                self._init_python_network(error_type, error_msg)

    def _init_cpp_network(self):
        """Initialize using C++ backend."""
        from . import _dnn_core

        # Check CUDA availability if requested
        if self._device == "cuda" and not _dnn_core.cuda_available():
            raise RuntimeError(
                "CUDA requested but not available. "
                "Rebuild pydnn with DNN_ENABLE_CUDA=ON or use device='cpu'"
            )

        config = _dnn_core.NetworkConfig()
        config.seed = self.seed
        config.input_shape = list(self.input_shape)
        config.output_size = self.output_size
        config.cost_function = getattr(_dnn_core.CostFunction, self.cost_function)
        config.device = _dnn_core.Device.CUDA if self._device == "cuda" else _dnn_core.Device.CPU

        self._network = _dnn_core.Network(config)

    def _attempt_cpp_recovery(self) -> bool:
        """
        Attempt to recover from C++ loading failure by auto-building.

        This is only attempted for MISSING_BINARY errors in development environments
        where setup.py is available.

        Returns:
            True if recovery successful and C++ extension now works, False otherwise
        """
        import subprocess

        # Look for setup.py in parent directory (dev environment)
        setup_py = os.path.join(os.path.dirname(__file__), "..", "setup.py")

        if not os.path.exists(setup_py):
            return False

        try:
            # Attempt to build C++ extension
            result = subprocess.run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                cwd=os.path.dirname(setup_py),
                capture_output=True,
                timeout=120
            )

            if result.returncode == 0:
                # Retry import after successful build
                try:
                    from . import _dnn_core
                    return True
                except ImportError:
                    return False
            else:
                return False

        except subprocess.TimeoutExpired:
            warnings.warn(
                "C++ extension build timed out after 120 seconds.",
                RuntimeWarning
            )
            return False
        except Exception:
            return False

    def _init_python_network(self, error_type: str = None, error_msg: str = None):
        """
        Initialize using pure Python fallback with detailed guidance.

        Args:
            error_type: The CppLoadError type that caused fallback
            error_msg: Detailed error message from C++ loading attempt
        """
        # Pure Python implementation for when C++ is not available
        self._network = None

        # Build comprehensive help message based on failure type
        help_sections = []

        if error_type == CppLoadError.MISSING_BINARY:
            help_sections.append(
                "BUILD FROM SOURCE:\n"
                "  pip install pybind11 numpy\n"
                "  python setup.py build_ext --inplace"
            )

        elif error_type == CppLoadError.MISSING_DEPENDENCIES:
            if sys.platform == "win32":
                help_sections.append(
                    "INSTALL RUNTIME:\n"
                    "  Download Visual C++ Redistributable from:\n"
                    "  https://aka.ms/vs/17/release/vc_redist.x64.exe"
                )
            else:
                help_sections.append(
                    "CHECK DEPENDENCIES:\n"
                    "  Run: ldd python/pydnn/_dnn_core*.so\n"
                    "  Install any missing shared libraries"
                )

        elif error_type == CppLoadError.PYTHON_VERSION_MISMATCH:
            help_sections.append(
                f"REBUILD FOR PYTHON {sys.version_info.major}.{sys.version_info.minor}:\n"
                "  python setup.py build_ext --inplace --force"
            )

        elif error_type == CppLoadError.ARCH_MISMATCH:
            python_arch = platform.architecture()[0]
            help_sections.append(
                f"ARCHITECTURE MISMATCH:\n"
                f"  Python is {python_arch}.\n"
                "  Rebuild with matching architecture:\n"
                "  python setup.py build_ext --inplace --force"
            )

        elif error_type == CppLoadError.PERMISSION_DENIED:
            help_sections.append(
                "FIX PERMISSIONS:\n"
                "  Check file permissions on the C++ binary.\n"
                "  On Linux/Mac: chmod +r python/pydnn/_dnn_core*.so"
            )

        elif error_type == CppLoadError.CORRUPTED_BINARY:
            help_sections.append(
                "REINSTALL OR REBUILD:\n"
                "  pip install --force-reinstall pydnn\n"
                "  Or rebuild: python setup.py build_ext --inplace --force"
            )

        # Always include generic install option
        help_sections.append(
            "INSTALL PRE-BUILT (if available):\n"
            "  pip install --upgrade pydnn"
        )

        full_message = (
            f"C++ extensions unavailable. Using Python fallback (10-100x slower).\n"
            f"Reason: {error_msg or 'Unknown'}\n\n"
            + "\n\n".join(help_sections)
        )

        warnings.warn(full_message, UserWarning, stacklevel=3)

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            callback: Optional[Callable[[int, float, float], None]] = None,
            verbose: bool = True) -> TrainingResult:
        """
        Train the network with automatic parameter adjustment.

        All training parameters are determined automatically from the seed:
        - Learning rate and schedule
        - Batch size progression
        - Early stopping criteria
        - Layer/node adjustments

        Args:
            X: Input data array of shape (n_samples, *input_shape)
            y: Target data array of shape (n_samples, output_size)
            callback: Optional callback(epoch, cost, efficiency)
            verbose: Print training progress

        Returns:
            TrainingResult with training history
        """
        if verbose:
            print(f"Training Dynamic Neural Network")
            print(f"  Input shape: {self.input_shape}")
            print(f"  Output size: {self.output_size}")
            print(f"  Seed: {self.seed}")
            print(f"  Cost function: {self.cost_function}")
            print(f"  Samples: {len(X)}")
            print()

        # Validate input shapes
        expected_input = (len(X),) + self.input_shape
        if X.shape != expected_input:
            # Try to reshape if total size matches
            if np.prod(X.shape[1:]) == np.prod(self.input_shape):
                X = X.reshape(expected_input)
            else:
                raise ValueError(f"Expected input shape {expected_input}, got {X.shape}")

        # Convert to float32
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        # Apply normalization if enabled
        X, y = self._normalize_data(X, y, verbose)

        if self._use_cpp:
            result = self._fit_cpp(X, y, callback, verbose)
        else:
            result = self._fit_python(X, y, callback, verbose)

        self._trained = True
        self._training_result = result

        if verbose:
            print(f"\nTraining complete!")
            print(f"  Epochs: {result.epochs_completed}")
            print(f"  Final cost: {result.final_cost:.6f}")
            print(f"  Final efficiency: {result.final_efficiency:.4f}")
            print(f"  Stopping reason: {result.stopping_reason}")

        return result

    def _fit_cpp(self, X, y, callback, verbose) -> TrainingResult:
        """Train using C++ backend."""
        from . import _dnn_core

        # PyTrainer::train() accepts raw numpy arrays directly (it
        # converts to per-sample tensors internally). Make sure the
        # arrays are float32 and contiguous so the buffer protocol path
        # in the binding is happy.
        inputs = np.ascontiguousarray(X, dtype=np.float32)
        targets = np.ascontiguousarray(y, dtype=np.float32)

        # Build a TrainerConfig that opts into the concurrent runtime
        # if the user requested it. Other fields keep their C++ defaults.
        trainer_config = _dnn_core.TrainerConfig()
        if self.runtime_enabled:
            trainer_config.runtime_enabled = True
            if verbose:
                print("  [runtime] StageController + parallel Estimation observer enabled.")

        # Tier 1: derive thresholds from dataset variance + complexity.
        # When False, leave trainer_config at its C++ defaults so behaviour
        # matches the legacy static-default path bit-for-bit.
        if self.dynamic_thresholds:
            try:
                from .dynamic_thresholds import (
                    apply_to_trainer_config,
                    compute_data_signals,
                    derive_thresholds,
                )
                signals = compute_data_signals(inputs, targets)
                derived = derive_thresholds(signals)
                if derived:
                    apply_to_trainer_config(trainer_config, derived)
                self._last_data_signals = signals
                if verbose:
                    print("  [dynamic_thresholds] variance=%.3f complexity=%.3f%s" % (
                        signals.variance_score,
                        signals.complexity_score,
                        " (fallback)" if signals.fallback_used else "",
                    ))
                    if derived:
                        for key, value in derived.items():
                            print(f"    {key} = {value}")
            except Exception as exc:  # noqa: BLE001
                # Never let signal computation break training. Fall back
                # to static defaults and surface the issue in verbose mode.
                self._last_data_signals = None
                if verbose:
                    print(f"  [dynamic_thresholds] disabled due to error: {exc!r}")

        # Create trainer with the config
        try:
            trainer = _dnn_core.Trainer(
                self._network,
                getattr(_dnn_core.CostFunction, self.cost_function),
                trainer_config,
            )
        except (TypeError, AttributeError):
            # Older binding without the 3-arg ctor; fall back to legacy.
            trainer = _dnn_core.Trainer(self._network)

        if callback:
            trainer.set_epoch_callback(callback)

        # Train
        cpp_result = trainer.train(inputs, targets)

        # Hard-remove dormant (soft-pruned) capacity from the underlying
        # C++ network so the inference model is the lean compacted form.
        # During training, "remove node" / "remove layer" only flipped an
        # active mask; this is where they actually stop costing FLOPs.
        try:
            removed = self._network.compact()
            if verbose and removed > 0:
                print(f"  Compacted {removed} dormant nodes from final model.")
        except AttributeError:
            # Older C++ extension without compact(); harmless to skip.
            pass

        return TrainingResult(
            success=cpp_result.success,
            epochs_completed=cpp_result.epochs_completed,
            final_cost=cpp_result.final_cost,
            final_efficiency=cpp_result.final_efficiency,
            best_cost=cpp_result.best_cost,
            best_efficiency=cpp_result.best_efficiency,
            stopping_reason=cpp_result.stopping_reason,
            cost_history=list(cpp_result.cost_history),
            efficiency_history=list(cpp_result.efficiency_history),
            training_time_ms=(
                cpp_result.training_time.count()
                if hasattr(cpp_result.training_time, "count")
                else int(cpp_result.training_time)
            ),
        )

    def _fit_python(self, X, y, callback, verbose) -> TrainingResult:
        """Train using pure Python fallback with 4-phase training.

        When self.runtime_enabled is True, a parallel _CostTrendObserver
        runs alongside the phase loops and raises a rewind flag when
        cost improvement stalls. The Phase 3 loop checks the flag
        between epochs and breaks out so the controller can re-run a
        brief Phase 2 refresh, mirroring the C++ runtime path.
        """
        import time
        start_time = time.time()

        np.random.seed(self.seed)

        # Initialize network architecture
        self._init_architecture(X.shape[1])

        # Start the parallel cost-trend observer if the user opted in.
        # The observer reads from the cost_history list defined just
        # below; safe because we only mutate cost_history from this
        # thread (append) and the observer only reads.
        observer = _CostTrendObserver() if self.runtime_enabled else None

        # Tracking variables
        cost_history = []
        # Parallel observer reads cost_history (read-only from its thread).
        if observer is not None:
            observer.start(cost_history)
            if verbose:
                print("  [runtime] Python parallel cost-trend observer started.")
        efficiency_history = []
        cancer_score_history = []
        alzheimer_score_history = []
        architecture_history = []
        nodes_added = 0
        nodes_removed = 0
        layers_added = 0
        layers_removed = 0
        perturbations_applied = 0
        best_cost = float('inf')
        best_architecture = None

        # Emotional state tracking (for Phase 3 reward/penalty system)
        emotional_state = EmotionalState()
        rp_config = self.reward_penalty
        learning_rate_history = []
        emotional_state_actions = []

        # Shuffle data
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        # ============ PRE-TRAINING: ADAPTIVE WEIGHT INITIALIZATION ============
        # Run a forward pass to collect initial activation statistics
        batch_size = min(100, len(X))
        m = batch_size
        batch_X = X[:m]
        # Forward pass through all layers to collect activation statistics
        activations = [batch_X.reshape(m, -1)]
        for i, layer in enumerate(self._layers):
            z = activations[-1] @ layer["W"].T + layer["b"]
            if i < len(self._layers) - 1:
                a = np.maximum(0, z)  # ReLU
                # Record activation statistics
                if layer.get("activation_stats") is not None:
                    node_activations = np.mean(a, axis=0)
                    layer["activation_stats"]["sum"] += node_activations
                    layer["activation_stats"]["sq_sum"] += node_activations ** 2
                    layer["activation_stats"]["count"] += 1
            else:
                a = z  # Output layer
            activations.append(a)
        # Compute variance z-scores and adapt variance weights for all layers
        self._compute_initial_variance_zscores()
        # Reset stats after initial pass
        self._reset_layer_stats()

        # ============ PHASE 1: EXPLORATION ============
        phase1_start = time.time()
        exploration_epochs = self.training_phase.exploration_epochs
        if verbose:
            print("\n" + "=" * 60)
            print(f"PHASE 1: EXPLORATION ({exploration_epochs} epochs)")
            print("=" * 60)

        exploration_costs = []
        learning_rate = self.training_phase.exploration_learning_rate
        # Use lower learning rate for regression tasks (MSE is more sensitive)
        if self.cost_function == "MSE":
            learning_rate = min(learning_rate, 0.01)
        batch_size = self.training_phase.exploration_batch_size

        phase1_iter = range(exploration_epochs)
        if verbose and TQDM_AVAILABLE:
            phase1_iter = tqdm(phase1_iter, desc="Phase 1: Exploration", unit="epoch")

        for epoch in phase1_iter:
            epoch_cost = self._train_epoch(X, y, learning_rate, batch_size)
            exploration_costs.append(epoch_cost)
            cost_history.append(epoch_cost)

            # Aggressive architecture exploration
            added, removed, layer_change = self._aggressive_layer_adjustment(
                saturation_threshold=self.training_phase.exploration_saturation_threshold,
                efficiency_threshold=self.training_phase.exploration_efficiency_threshold
            )
            nodes_added += added
            nodes_removed += removed
            if layer_change > 0:
                layers_added += layer_change
            elif layer_change < 0:
                layers_removed += abs(layer_change)

            # Track best architecture
            if epoch_cost < best_cost:
                best_cost = epoch_cost
                best_architecture = self._save_architecture()

            # Compute metrics
            efficiency = self._compute_efficiency(cost_history)
            efficiency_history.append(efficiency)
            cancer, alzheimer = self._compute_health_scores(nodes_added, nodes_removed, layers_added, layers_removed, epoch + 1)
            cancer_score_history.append(cancer)
            alzheimer_score_history.append(alzheimer)
            architecture_history.append((len(self._layers), self._count_nodes()))

            if verbose and TQDM_AVAILABLE:
                phase1_iter.set_postfix({
                    'cost': f'{epoch_cost:.4f}',
                    'layers': len(self._layers),
                    'nodes': self._count_nodes(),
                    'lr': f'{learning_rate:.4f}'
                })
            elif verbose and epoch % 2 == 0:
                print(f"  Epoch {epoch}: cost={epoch_cost:.4f}, layers={len(self._layers)}, nodes={self._count_nodes()}")

            self._safe_callback(callback, epoch, epoch_cost, efficiency)

        phase1_time = time.time() - phase1_start

        # ============ POST-PHASE 1: COMPUTE GRADIENT THRESHOLDS ============
        # Auto-detect gradient thresholds based on Phase 1 statistics
        self._compute_gradient_threshold()

        # ============ PHASE 2: ESTIMATION ============
        phase2_start = time.time()
        estimation_epochs = self.training_phase.estimation_epochs
        if verbose:
            print("\n" + "=" * 60)
            print(f"PHASE 2: ESTIMATION ({estimation_epochs} epochs)")
            print("=" * 60)

        estimation_costs = []
        learning_rate = self.training_phase.estimation_learning_rate
        # Use lower learning rate for regression tasks
        if self.cost_function == "MSE":
            learning_rate = min(learning_rate, 0.01)

        phase2_iter = range(estimation_epochs)
        if verbose and TQDM_AVAILABLE:
            phase2_iter = tqdm(phase2_iter, desc="Phase 2: Estimation", unit="epoch")

        for epoch in phase2_iter:
            epoch_cost = self._train_epoch(X, y, learning_rate, batch_size)
            estimation_costs.append(epoch_cost)
            cost_history.append(epoch_cost)

            efficiency = self._compute_efficiency(cost_history)
            efficiency_history.append(efficiency)
            cancer, alzheimer = self._compute_health_scores(nodes_added, nodes_removed, layers_added, layers_removed, len(cost_history))
            cancer_score_history.append(cancer)
            alzheimer_score_history.append(alzheimer)
            architecture_history.append((len(self._layers), self._count_nodes()))

            # Adapt efficiency weights based on gradient statistics
            self._adapt_node_weights()

            self._safe_callback(callback, exploration_epochs + epoch, epoch_cost, efficiency)

        # Estimate epochs needed based on cost reduction rate
        if estimation_costs and len(estimation_costs) >= 2:
            cost_reduction_rate = (estimation_costs[0] - estimation_costs[-1]) / estimation_epochs
            current_cost = estimation_costs[-1]
            initial_cost = estimation_costs[0]

            if cost_reduction_rate > 1e-6 and current_cost > 0:
                # Target: reduce cost by 99%
                target_cost = initial_cost * 0.01
                remaining_cost = max(0, current_cost - target_cost)
                # Apply diminishing returns factor (0.5) - progress slows over time
                effective_rate = cost_reduction_rate * 0.5
                estimated_epochs = int(remaining_cost / (effective_rate + 1e-6))
            else:
                # No meaningful progress - use default
                estimated_epochs = 100
        else:
            estimated_epochs = 100

        estimated_epochs = max(self.training_phase.min_estimated_epochs, min(self.training_phase.max_estimated_epochs, estimated_epochs))

        phase2_time = time.time() - phase2_start
        if verbose:
            print(f"  Estimated epochs for {self.training_phase.target_efficiency:.0%} efficiency: {estimated_epochs}")

        # ============ PHASE 3: MAIN TRAINING ============
        phase3_start = time.time()
        if verbose:
            print("\n" + "=" * 60)
            print(f"PHASE 3: MAIN TRAINING ({estimated_epochs} epochs)")
            print("=" * 60)

        perturbation_cutoff = int(estimated_epochs * self.training_phase.perturbation_cutoff_ratio)
        learning_rate = self.training_phase.main_learning_rate
        # Use lower learning rate for regression tasks
        if self.cost_function == "MSE":
            learning_rate = min(learning_rate, 0.01)
        batch_size = self.training_phase.main_initial_batch_size

        phase3_iter = range(estimated_epochs)
        if verbose and TQDM_AVAILABLE:
            phase3_iter = tqdm(phase3_iter, desc="Phase 3: Training", unit="epoch")

        for epoch in phase3_iter:
            total_epoch = exploration_epochs + estimation_epochs + epoch  # Account for phases 1 and 2

            # Apply random perturbation in first portion to avoid overfitting
            if epoch < perturbation_cutoff:
                self._apply_random_perturbation(fraction=self.perturbation.perturbation_fraction)
                perturbations_applied += 1

            # Compute adaptive saturation threshold
            efficiency = self._compute_efficiency(cost_history)
            saturation_threshold = self._sigmoid_threshold(efficiency)

            # Train one epoch
            epoch_cost = self._train_epoch(X, y, learning_rate, batch_size)
            cost_history.append(epoch_cost)

            # Track best cost
            if epoch_cost < best_cost:
                best_cost = epoch_cost

            # Layer adjustment with adaptive threshold
            added, removed, layer_change = self._layer_adjustment(
                saturation_threshold=saturation_threshold,
                efficiency_threshold=self.efficiency.default_efficiency_threshold
            )
            nodes_added += added
            nodes_removed += removed
            if layer_change > 0:
                layers_added += layer_change
            elif layer_change < 0:
                layers_removed += abs(layer_change)

            # Compute metrics
            efficiency = self._compute_efficiency(cost_history)
            efficiency_history.append(efficiency)
            cancer, alzheimer = self._compute_health_scores(nodes_added, nodes_removed, layers_added, layers_removed, len(cost_history))
            cancer_score_history.append(cancer)
            alzheimer_score_history.append(alzheimer)
            architecture_history.append((len(self._layers), self._count_nodes()))

            # Apply reward/penalty system (replaces fixed LR decay)
            learning_rate, action = self._apply_reward_penalty_system(
                learning_rate, cost_history, efficiency_history, rp_config, emotional_state
            )
            # Cap learning rate for MSE (regression is more sensitive)
            if self.cost_function == "MSE":
                learning_rate = min(learning_rate, 0.01)
            learning_rate_history.append(learning_rate)
            emotional_state_actions.append(action)

            # Adapt efficiency weights based on gradient statistics
            self._adapt_node_weights()

            # Adaptive batch size growth
            if epoch > 0 and epoch % self.training_phase.batch_size_growth_interval == 0:
                batch_size = min(batch_size * 2, self.training_phase.main_max_batch_size)

            # Update progress bar
            if verbose and TQDM_AVAILABLE:
                phase3_iter.set_postfix({
                    'cost': f'{epoch_cost:.4f}',
                    'eff': f'{efficiency:.2%}',
                    'lr': f'{learning_rate:.4f}',
                    'dep': f'{emotional_state.depression_ratio:.0%}',
                    'exc': f'{emotional_state.excitement_ratio:.0%}',
                    'R/P': f'{emotional_state.total_rewards}/{emotional_state.total_penalties}'
                })
            elif verbose and epoch % 10 == 0:
                print(f"  Epoch {epoch}: cost={epoch_cost:.4f}, eff={efficiency:.2%}, lr={learning_rate:.4f}, R/P={emotional_state.total_rewards}/{emotional_state.total_penalties}")

            self._safe_callback(callback, total_epoch, epoch_cost, efficiency)

            # Parallel observer rewind: when the cost-trend observer
            # raises rewind, we exit the Phase 3 inner loop early so the
            # outer flow can run a brief Phase 2 refresh and re-enter
            # Phase 3 with the preserved emotional state. Capped at one
            # rewind per call (clear_rewind, then ignore subsequent).
            if observer is not None and observer.rewind_requested():
                observer.clear_rewind()
                if verbose:
                    print(f"  [runtime] Cost-trend stalled at epoch {epoch}; observer requested rewind.")
                # The full rewind+resume control flow lives in the C++
                # train_phased_runtime(); the Python fallback opts for
                # the simpler "early break" interpretation here so the
                # observer signal is still observable end-to-end.
                break

            # Early stopping check
            if len(cost_history) > self.early_stopping.window_size:
                recent_improvement = cost_history[-self.early_stopping.window_size] - cost_history[-1]
                if recent_improvement < self.early_stopping.improvement_threshold:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch} (no improvement)")
                    break

        phase3_time = time.time() - phase3_start

        # ============ PHASE 4: STANDARD TRAINING ============
        # Initialize Phase 4 variables
        phase4_time = 0.0
        phase4_estimated_epochs = 0
        phase4_epochs = 0
        phase4_initial_cost = 0.0
        phase4_final_cost = 0.0
        phase4_cost_reduction = 0.0
        phase4_early_stopped = False

        if self.training_phase.phase4_enabled:
            phase4_start = time.time()
            phase3_final_cost = cost_history[-1] if cost_history else 0.0

            # Estimate epochs for Phase 4
            phase4_estimated_epochs = self._estimate_phase4_epochs(
                phase3_final_cost,
                cost_history[-50:]  # Use last 50 epochs of history
            )

            if verbose:
                print("\n" + "=" * 60)
                print(f"PHASE 4: STANDARD TRAINING ({phase4_estimated_epochs} epochs)")
                print("=" * 60)
                print(f"  Target: {self.training_phase.phase4_target_cost_reduction:.0%} cost reduction")
                print(f"  Architecture frozen: {len(self._layers)} layers, {self._count_nodes()} nodes")

            phase4_initial_cost = phase3_final_cost
            phase4_learning_rate = self.training_phase.phase4_learning_rate

            # Use lower LR for regression tasks
            if self.cost_function == "MSE":
                phase4_learning_rate = min(phase4_learning_rate, 0.005)

            batch_size = self.training_phase.phase4_batch_size
            patience_counter = 0
            phase4_best_cost = phase4_initial_cost

            phase4_iter = range(phase4_estimated_epochs)
            if verbose and TQDM_AVAILABLE:
                phase4_iter = tqdm(phase4_iter, desc="Phase 4: Standard", unit="epoch")

            for epoch in phase4_iter:
                total_epoch = exploration_epochs + estimation_epochs + estimated_epochs + epoch

                # Train one epoch (NO architecture modifications - frozen)
                epoch_cost = self._train_epoch(X, y, phase4_learning_rate, batch_size)
                cost_history.append(epoch_cost)

                # Track metrics (architecture is frozen, but we still track)
                efficiency = self._compute_efficiency(cost_history)
                efficiency_history.append(efficiency)
                cancer, alzheimer = self._compute_health_scores(
                    nodes_added, nodes_removed, layers_added, layers_removed, len(cost_history)
                )
                cancer_score_history.append(cancer)
                alzheimer_score_history.append(alzheimer)
                architecture_history.append((len(self._layers), self._count_nodes()))

                # Track best cost
                if epoch_cost < best_cost:
                    best_cost = epoch_cost

                # Track best cost and patience for early stopping
                if epoch_cost < phase4_best_cost - self.training_phase.phase4_min_improvement:
                    phase4_best_cost = epoch_cost
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Learning rate decay
                if epoch > 0 and epoch % self.training_phase.phase4_lr_decay_interval == 0:
                    phase4_learning_rate = max(
                        self.training_phase.phase4_min_learning_rate,
                        phase4_learning_rate * self.training_phase.phase4_lr_decay_rate
                    )

                # Adapt efficiency weights based on gradient statistics
                self._adapt_node_weights()

                # Update progress bar
                if verbose and TQDM_AVAILABLE:
                    phase4_iter.set_postfix({
                        'cost': f'{epoch_cost:.6f}',
                        'eff': f'{efficiency:.2%}',
                        'lr': f'{phase4_learning_rate:.5f}',
                        'patience': f'{patience_counter}/{self.training_phase.phase4_patience}'
                    })
                elif verbose and epoch % 10 == 0:
                    print(f"  Epoch {epoch}: cost={epoch_cost:.6f}, lr={phase4_learning_rate:.5f}")

                self._safe_callback(callback, total_epoch, epoch_cost, efficiency)

                # Early stopping check
                if patience_counter >= self.training_phase.phase4_patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch} (no improvement for {patience_counter} epochs)")
                    phase4_early_stopped = True
                    phase4_epochs = epoch + 1
                    break

                # Check if target reduction achieved
                current_reduction = (phase4_initial_cost - epoch_cost) / (phase4_initial_cost + 1e-8)
                if current_reduction >= self.training_phase.phase4_target_cost_reduction:
                    if verbose:
                        print(f"  Target cost reduction achieved at epoch {epoch} ({current_reduction:.1%})")
                    phase4_epochs = epoch + 1
                    break

                phase4_epochs = epoch + 1

            phase4_time = time.time() - phase4_start
            phase4_final_cost = cost_history[-1] if cost_history else 0.0
            phase4_cost_reduction = (phase4_initial_cost - phase4_final_cost) / (phase4_initial_cost + 1e-8)

        # ============ TRAINING SUMMARY ============
        training_time_ms = int((time.time() - start_time) * 1000)

        if verbose:
            print("\n" + "=" * 60)
            print("TRAINING SUMMARY")
            print("=" * 60)
            print(f"  Total Time: {training_time_ms / 1000:.2f}s")
            if self.training_phase.phase4_enabled:
                print(f"  Phases: Exploration({phase1_time:.1f}s) -> Estimation({phase2_time:.1f}s) -> Training({phase3_time:.1f}s) -> Standard({phase4_time:.1f}s)")
            else:
                print(f"  Phases: Exploration({phase1_time:.1f}s) -> Estimation({phase2_time:.1f}s) -> Training({phase3_time:.1f}s)")
            initial_arch = architecture_history[0] if architecture_history else (2, 0)
            final_arch = architecture_history[-1] if architecture_history else (len(self._layers), self._count_nodes())
            print(f"  Architecture: {initial_arch[0]} layers -> {final_arch[0]} layers, {initial_arch[1]} nodes -> {final_arch[1]} nodes")
            print(f"  Nodes: +{nodes_added} added, -{nodes_removed} removed")
            print(f"  Layers: +{layers_added} added, -{layers_removed} removed")
            print(f"  Perturbations: {perturbations_applied}")
            print(f"  Health: Cancer={cancer_score_history[-1]:.1%}, Alzheimer={alzheimer_score_history[-1]:.1%}")
            print(f"  Emotional: {emotional_state.total_rewards} rewards, {emotional_state.total_penalties} penalties")
            print(f"  Depression ratio: {emotional_state.depression_ratio:.1%}, Excitement ratio: {emotional_state.excitement_ratio:.1%}")
            print(f"  LR resets: {emotional_state.lr_reset_count}")
            if self.training_phase.phase4_enabled:
                print(f"  Phase 4: {phase4_epochs} epochs, cost reduced by {phase4_cost_reduction:.1%}")
            print("=" * 60)

        # Stop the parallel observer (no-op if it wasn't started).
        if observer is not None:
            observer.stop()

        return TrainingResult(
            success=True,
            epochs_completed=len(cost_history),
            final_cost=cost_history[-1] if cost_history else 0.0,
            final_efficiency=efficiency_history[-1] if efficiency_history else 0.5,
            best_cost=best_cost,
            best_efficiency=max(efficiency_history) if efficiency_history else 0.5,
            stopping_reason="Training completed",
            cost_history=cost_history,
            efficiency_history=efficiency_history,
            training_time_ms=training_time_ms,
            nodes_added=nodes_added,
            nodes_removed=nodes_removed,
            layers_added=layers_added,
            layers_removed=layers_removed,
            cancer_score_history=cancer_score_history,
            alzheimer_score_history=alzheimer_score_history,
            architecture_history=architecture_history,
            perturbations_applied=perturbations_applied,
            phase_metrics={
                'phase1_time': phase1_time,
                'phase2_time': phase2_time,
                'phase3_time': phase3_time,
                'phase4_time': phase4_time,
                'phase4_estimated_epochs': phase4_estimated_epochs,
                'phase4_cost_reduction': phase4_cost_reduction,
                'estimated_epochs': estimated_epochs,
                'exploration_best_cost': min(exploration_costs) if exploration_costs else 0.0,
                'final_depression_ratio': emotional_state.depression_ratio,
                'final_excitement_ratio': emotional_state.excitement_ratio
            },
            # Emotional state fields
            total_rewards=emotional_state.total_rewards,
            total_penalties=emotional_state.total_penalties,
            depression_history=emotional_state.depression_history,
            excitement_history=emotional_state.excitement_history,
            lr_reset_count=emotional_state.lr_reset_count,
            learning_rate_history=learning_rate_history,
            emotional_state_history=emotional_state_actions,
            # Phase 4 fields
            phase4_epochs=phase4_epochs,
            phase4_initial_cost=phase4_initial_cost,
            phase4_final_cost=phase4_final_cost,
            phase4_cost_reduction=phase4_cost_reduction,
            phase4_early_stopped=phase4_early_stopped
        )

    def _init_architecture(self, input_size: int) -> None:
        """Initialize network architecture."""
        # Compute initial hidden layer size (geometric mean)
        hidden_size = int(np.sqrt(input_size * self.output_size))
        hidden_size = max(hidden_size, self.architecture.min_initial_hidden_size)

        # Initialize weights with He initialization
        W1 = np.random.randn(hidden_size, input_size).astype(np.float32) * np.sqrt(2.0 / input_size)
        b1 = np.zeros(hidden_size, dtype=np.float32)
        W2 = np.random.randn(self.output_size, hidden_size).astype(np.float32) * np.sqrt(2.0 / hidden_size)
        b2 = np.zeros(self.output_size, dtype=np.float32)

        # Store layers (output layer has no efficiency - nodes are fixed)
        initial_eff = self.efficiency.initial_efficiency
        self._layers = [
            {
                "W": W1,
                "b": b1,
                "efficiency": np.ones(hidden_size) * initial_eff,
                "efficiency_weights": [EfficiencyWeights() for _ in range(hidden_size)],
                "activation_stats": {
                    "sum": np.zeros(hidden_size),
                    "sq_sum": np.zeros(hidden_size),
                    "count": 0,
                    "dead_count": np.zeros(hidden_size, dtype=np.int64)
                },
                "gradient_stats": {
                    "sum": np.zeros(hidden_size),
                    "count": 0
                }
            },
            {"W": W2, "b": b2, "efficiency": None, "efficiency_weights": None}  # Output layer - no dynamic node management
        ]

        # Store initial node count for health scoring
        self._initial_node_count = self._count_nodes()
        # Hidden-only count (output layer excluded; it can never be pruned)
        self._initial_hidden_count = sum(
            self._layers[i]["W"].shape[0] for i in range(len(self._layers) - 1)
        )

    def _normalize_data(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize input and output data, storing parameters for denormalization.

        Args:
            X: Input data array
            y: Target data array
            verbose: Print normalization info

        Returns:
            Tuple of (normalized_X, normalized_y)
        """
        eps = self.normalization.epsilon
        X_normalized = X.copy()
        y_normalized = y.copy()

        # Normalize input data
        if self.normalization.normalize_input:
            X_flat = X.reshape(len(X), -1)

            if self.normalization.method == 'zscore':
                self._input_mean = np.mean(X_flat, axis=0)
                self._input_std = np.std(X_flat, axis=0) + eps
                X_normalized = ((X_flat - self._input_mean) / self._input_std).reshape(X.shape)
            elif self.normalization.method == 'minmax':
                self._input_min = np.min(X_flat, axis=0)
                self._input_max = np.max(X_flat, axis=0)
                input_range = self._input_max - self._input_min + eps
                X_normalized = ((X_flat - self._input_min) / input_range).reshape(X.shape)

            if verbose:
                print(f"  Input normalization: {self.normalization.method}")
                if self.normalization.method == 'zscore':
                    print(f"    Mean range: [{self._input_mean.min():.4f}, {self._input_mean.max():.4f}]")
                    print(f"    Std range: [{self._input_std.min():.4f}, {self._input_std.max():.4f}]")

        # Normalize output data (only for regression tasks)
        if self.normalization.normalize_output and self.cost_function in ["MSE", "MAE", "Huber", "LogCosh"]:
            y_flat = y.reshape(len(y), -1)

            if self.normalization.method == 'zscore':
                self._output_mean = np.mean(y_flat, axis=0)
                self._output_std = np.std(y_flat, axis=0) + eps
                y_normalized = ((y_flat - self._output_mean) / self._output_std).reshape(y.shape)
            elif self.normalization.method == 'minmax':
                self._output_min = np.min(y_flat, axis=0)
                self._output_max = np.max(y_flat, axis=0)
                output_range = self._output_max - self._output_min + eps
                y_normalized = ((y_flat - self._output_min) / output_range).reshape(y.shape)

            if verbose:
                print(f"  Output normalization: {self.normalization.method}")
                if self.normalization.method == 'zscore':
                    print(f"    Mean range: [{self._output_mean.min():.4f}, {self._output_mean.max():.4f}]")
                    print(f"    Std range: [{self._output_std.min():.4f}, {self._output_std.max():.4f}]")

        return X_normalized.astype(np.float32), y_normalized.astype(np.float32)

    def _normalize_input(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize input data using stored parameters (for prediction).

        Args:
            X: Input data array

        Returns:
            Normalized input array
        """
        if not self.normalization.normalize_input:
            return X

        eps = self.normalization.epsilon
        X_flat = X.reshape(len(X), -1)

        if self.normalization.method == 'zscore' and self._input_mean is not None:
            X_normalized = (X_flat - self._input_mean) / self._input_std
        elif self.normalization.method == 'minmax' and self._input_min is not None:
            input_range = self._input_max - self._input_min + eps
            X_normalized = (X_flat - self._input_min) / input_range
        else:
            return X

        return X_normalized.reshape(X.shape).astype(np.float32)

    def _denormalize_output(self, y: np.ndarray) -> np.ndarray:
        """
        Denormalize output data using stored parameters.

        Args:
            y: Normalized output array

        Returns:
            Denormalized output array in original scale
        """
        if not self.normalization.normalize_output:
            return y

        # Only denormalize for regression tasks
        if self.cost_function not in ["MSE", "MAE", "Huber", "LogCosh"]:
            return y

        eps = self.normalization.epsilon
        y_flat = y.reshape(len(y), -1)

        if self.normalization.method == 'zscore' and self._output_mean is not None:
            y_denormalized = y_flat * self._output_std + self._output_mean
        elif self.normalization.method == 'minmax' and self._output_min is not None:
            output_range = self._output_max - self._output_min + eps
            y_denormalized = y_flat * output_range + self._output_min
        else:
            return y

        return y_denormalized.reshape(y.shape).astype(np.float32)

    def get_normalization_params(self) -> Dict[str, Any]:
        """
        Get stored normalization parameters for external use.

        Returns:
            Dictionary containing normalization parameters:
            - input_mean, input_std (for zscore)
            - input_min, input_max (for minmax)
            - output_mean, output_std (for zscore, regression only)
            - output_min, output_max (for minmax, regression only)
            - method: normalization method used
        """
        return {
            'method': self.normalization.method,
            'input_mean': self._input_mean,
            'input_std': self._input_std,
            'input_min': self._input_min,
            'input_max': self._input_max,
            'output_mean': self._output_mean,
            'output_std': self._output_std,
            'output_min': self._output_min,
            'output_max': self._output_max,
        }

    def _count_nodes(self) -> int:
        """Count total nodes in network."""
        return sum(layer["W"].shape[0] for layer in self._layers)

    def _train_epoch(self, X: np.ndarray, y: np.ndarray, learning_rate: float, batch_size: int) -> float:
        """Train for one epoch and return average cost."""
        epoch_cost = 0.0
        n_samples = len(X)
        n_batches = max(1, n_samples // batch_size)

        # Pre-allocate or reuse permutation array for in-place shuffle
        if not hasattr(self, '_perm') or len(self._perm) != n_samples:
            self._perm = np.arange(n_samples, dtype=np.int32)
        np.random.shuffle(self._perm)  # In-place shuffle - no allocation

        # Pre-shuffle data once per epoch (one copy instead of per-batch copies)
        X_shuffled = X[self._perm]
        y_shuffled = y[self._perm]

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)

            # Use contiguous slices (views, not copies)
            batch_X = X_shuffled[start_idx:end_idx]
            batch_y = y_shuffled[start_idx:end_idx]
            m = len(batch_X)

            # Forward pass through all layers
            activations = [batch_X.reshape(m, -1)]
            z_values = []

            for i, layer in enumerate(self._layers):
                z = activations[-1] @ layer["W"].T + layer["b"]
                z_values.append(z)

                if i < len(self._layers) - 1:
                    # ReLU for hidden layers
                    a = np.maximum(0, z)

                    # Record activation statistics for adaptive weights
                    if layer.get("activation_stats") is not None:
                        # Sum of activations per node (averaged across batch samples)
                        node_activations = np.mean(a, axis=0)
                        layer["activation_stats"]["sum"] += node_activations
                        layer["activation_stats"]["sq_sum"] += node_activations ** 2
                        layer["activation_stats"]["count"] += 1
                        # Count dead activations (near zero)
                        dead_mask = np.mean(np.abs(a) < 1e-7, axis=0)
                        layer["activation_stats"]["dead_count"] += (dead_mask > 0.99).astype(np.int64)
                else:
                    # Output layer activation depends on cost function
                    if self.cost_function == "MSE":
                        # Linear activation for regression
                        a = z
                    else:
                        # Softmax for classification (CrossEntropy)
                        z_max = np.max(z, axis=1, keepdims=True)
                        exp_z = np.exp(z - z_max)
                        a = exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-8)
                activations.append(a)

            # Compute cost based on cost function
            if self.cost_function == "MSE":
                # Mean Squared Error for regression
                batch_cost = np.mean((activations[-1] - batch_y) ** 2)
            else:
                # Cross-Entropy for classification
                log_probs = np.log(activations[-1] + 1e-8)
                batch_cost = -np.mean(np.sum(batch_y * log_probs, axis=1))
            epoch_cost += batch_cost

            # Backward pass - gradient depends on cost function
            if self.cost_function == "MSE":
                # MSE gradient: (output - target) / m (factor of 2 absorbed into learning rate)
                dz = (activations[-1] - batch_y) / m
            else:
                # Cross-Entropy + Softmax gradient: (output - target) / m
                dz = (activations[-1] - batch_y) / m

            for i in range(len(self._layers) - 1, -1, -1):
                dW = dz.T @ activations[i]
                db = np.sum(dz, axis=0)

                # Gradient clipping FIRST (before NaN check - clipping may fix large values)
                clip_value = self.gradient.gradient_clip_value
                dW = np.clip(dW, -clip_value, clip_value)
                db = np.clip(db, -clip_value, clip_value)

                # NaN/Inf detection AFTER clipping - skip update only if still invalid
                if not np.isfinite(dW).all() or not np.isfinite(db).all():
                    warnings.warn(
                        f"NaN/Inf detected in gradients at layer {i}, batch {batch_idx} after clipping. Skipping update.",
                        RuntimeWarning
                    )
                    if i > 0:
                        da = dz @ self._layers[i]["W"]
                        dz = da * (z_values[i - 1] > 0).astype(np.float32)
                    continue

                # Update weights
                self._layers[i]["W"] -= learning_rate * dW
                self._layers[i]["b"] -= learning_rate * db

                # Update efficiency based on gradient magnitude (skip output layer)
                if self._layers[i]["efficiency"] is not None:
                    grad_magnitude = np.mean(np.abs(dW), axis=1)
                    eff_decay = self.efficiency.efficiency_decay
                    eff_scale = self.efficiency.efficiency_update_scale
                    eff_grad_mult = self.efficiency.efficiency_gradient_multiplier
                    self._layers[i]["efficiency"] = eff_decay * self._layers[i]["efficiency"] + eff_scale * np.clip(grad_magnitude * eff_grad_mult, 0, 1)

                    # Record gradient statistics for adaptive weights
                    if self._layers[i].get("gradient_stats") is not None:
                        self._layers[i]["gradient_stats"]["sum"] += grad_magnitude
                        self._layers[i]["gradient_stats"]["count"] += 1

                if i > 0:
                    da = dz @ self._layers[i]["W"]
                    dz = da * (z_values[i - 1] > 0).astype(np.float32)

        avg_cost = epoch_cost / n_batches

        # Check for diverged training
        if not np.isfinite(avg_cost):
            raise RuntimeError(
                f"Training diverged: epoch cost is {avg_cost}. "
                "Try reducing learning rate or checking input data for NaN/Inf values."
            )

        return avg_cost

    def _compute_efficiency(self, cost_history: List[float]) -> float:
        """Compute efficiency metric based on cost reduction using windowed averaging."""
        if len(cost_history) < 2:
            return self.efficiency.initial_efficiency

        # Use a window for more stable efficiency computation
        window_size = min(5, len(cost_history))
        recent_costs = cost_history[-window_size:]

        # Calculate average improvement rate over window
        if len(recent_costs) >= 2:
            total_improvement = (recent_costs[0] - recent_costs[-1]) / (recent_costs[0] + 1e-8)
            avg_improvement = total_improvement / (len(recent_costs) - 1)
        else:
            avg_improvement = 0.0

        # Use sigmoid transformation for bounded, smooth efficiency
        # This maps any real number to (0, 1) range naturally
        k = self.sigmoid_threshold.k
        scaled_improvement = avg_improvement * 100.0
        efficiency = 1.0 / (1.0 + np.exp(-k * scaled_improvement / 5.0))

        return efficiency

    def _compute_initial_variance_zscores(self) -> None:
        """
        Compute variance z-scores and adapt variance weights for all nodes.
        Should be called once before training starts, after an initial forward pass.
        """
        for layer in self._layers:
            if layer.get("efficiency_weights") is None:
                continue

            stats = layer.get("activation_stats")
            if stats is None or stats["count"] == 0:
                continue

            # Compute variance for each node
            n = stats["count"]
            mean = stats["sum"] / n
            variance = stats["sq_sum"] / n - mean ** 2
            variance = np.maximum(variance, 0)  # Handle numerical issues

            # Compute z-scores across nodes in this layer
            var_mean = np.mean(variance)
            var_std = np.std(variance)
            if var_std < 1e-8:
                var_std = 1e-8

            z_scores = (variance - var_mean) / var_std

            # Adapt variance weights for each node
            for i, weights in enumerate(layer["efficiency_weights"]):
                weights.adapt_variance_from_zscore(z_scores[i])

    def _compute_gradient_threshold(self) -> None:
        """
        Compute gradient threshold from current gradient statistics.
        Sets threshold as mean + std of gradient magnitudes.
        Should be called after Phase 1 (exploration).
        """
        for layer in self._layers:
            if layer.get("efficiency_weights") is None:
                continue

            stats = layer.get("gradient_stats")
            if stats is None or stats["count"] == 0:
                continue

            # Compute mean gradient magnitude per node
            grad_mags = stats["sum"] / stats["count"]

            # Compute threshold: mean + std
            threshold = np.mean(grad_mags) + np.std(grad_mags)

            # Set threshold for all nodes in this layer
            for weights in layer["efficiency_weights"]:
                weights.grad_threshold = threshold

    def _adapt_node_weights(self) -> None:
        """
        Adapt gradient/contribution weights for all nodes based on current gradient statistics.
        Should be called each epoch after training.
        """
        for layer in self._layers:
            if layer.get("efficiency_weights") is None:
                continue

            stats = layer.get("gradient_stats")
            if stats is None or stats["count"] == 0:
                continue

            # Compute mean gradient magnitude per node
            grad_mags = stats["sum"] / stats["count"]

            # Adapt weights for each node
            for i, weights in enumerate(layer["efficiency_weights"]):
                weights.adapt_gradient_contribution(grad_mags[i])

    def _reset_layer_stats(self) -> None:
        """Reset activation and gradient statistics for all layers."""
        for layer in self._layers:
            if layer.get("activation_stats") is not None:
                n = layer["W"].shape[0]
                layer["activation_stats"] = {
                    "sum": np.zeros(n),
                    "sq_sum": np.zeros(n),
                    "count": 0,
                    "dead_count": np.zeros(n, dtype=np.int64)
                }
            if layer.get("gradient_stats") is not None:
                n = layer["W"].shape[0]
                layer["gradient_stats"] = {
                    "sum": np.zeros(n),
                    "count": 0
                }

    def _compute_health_scores(self, nodes_added: int, nodes_removed: int,
                               layers_added: int, layers_removed: int, epoch: int) -> Tuple[float, float]:
        """
        Compute cancer and alzheimer scores using both rate-based and ratio-based metrics.

        The health scores combine two approaches:
        1. Rate-based: How fast nodes are being added/removed per epoch
        2. Ratio-based: What percentage of initial nodes have been added/removed

        This ensures that even slow but extensive pruning (e.g., 70% of nodes removed
        over many epochs) is properly detected as Alzheimer state.
        """
        layer_weight = self.health_score.layer_weight

        # ===== RATE-BASED SCORING (original method) =====
        # Cancer score: excessive growth rate
        growth_rate = (nodes_added + layers_added * layer_weight) / (epoch + 1)
        cancer_rate = min(1.0, growth_rate / self.health_score.cancer_denominator)

        # Alzheimer score: excessive removal rate
        removal_rate = (nodes_removed + layers_removed * layer_weight) / (epoch + 1)
        alzheimer_rate = min(1.0, removal_rate / self.health_score.alzheimer_denominator)

        # ===== RATIO-BASED SCORING (new method to catch extensive changes) =====
        # This catches cases where pruning happens slowly but extensively
        cancer_ratio = 0.0
        alzheimer_ratio = 0.0

        initial_hidden = getattr(self, '_initial_hidden_count', None)
        initial = initial_hidden if initial_hidden else self._initial_node_count
        if initial is not None and initial > 0:

            # What percentage of initial nodes were removed?
            removal_percentage = nodes_removed / initial
            # Scale: 50% removal (threshold) = 1.0 score
            alzheimer_ratio = min(1.0, removal_percentage / self.health_score.alzheimer_ratio_threshold)

            # What percentage of initial nodes were added?
            growth_percentage = nodes_added / initial
            # Scale: 200% growth (threshold) = 1.0 score
            cancer_ratio = min(1.0, growth_percentage / self.health_score.cancer_ratio_threshold)

        # ===== COMBINE BOTH METRICS =====
        # Take the maximum of rate-based and ratio-based scores
        # This ensures both fast changes AND extensive changes are detected
        cancer = max(cancer_rate, cancer_ratio)
        alzheimer = max(alzheimer_rate, alzheimer_ratio)

        return cancer, alzheimer

    def _sigmoid_threshold(self, efficiency: float) -> float:
        """Compute adaptive saturation threshold using sigmoid function."""
        k = self.sigmoid_threshold.k
        base = self.sigmoid_threshold.base
        range_val = self.sigmoid_threshold.range_val
        center = self.sigmoid_threshold.center
        sigmoid = 1.0 / (1.0 + np.exp(-k * (efficiency - center)))
        return base + range_val * sigmoid

    def _estimate_phase4_epochs(self, phase3_final_cost: float,
                                 cost_history: List[float]) -> int:
        """
        Estimate epochs needed for Phase 4 based on cost reduction rate.

        Uses the cost reduction rate from recent training history and applies
        a diminishing returns factor since Phase 4 improvement is typically slower.

        Args:
            phase3_final_cost: Final cost from Phase 3
            cost_history: Cost history from training (uses last 20 epochs)

        Returns:
            Estimated number of epochs for Phase 4
        """
        window_size = min(20, len(cost_history))
        if window_size < 2:
            return self.training_phase.phase4_min_epochs

        recent_costs = cost_history[-window_size:]
        total_reduction = recent_costs[0] - recent_costs[-1]
        reduction_per_epoch = total_reduction / (window_size - 1)

        # Diminishing returns factor (Phase 4 improvement is slower)
        diminishing_factor = 0.3
        effective_rate = reduction_per_epoch * diminishing_factor

        if effective_rate <= 0:
            # No improvement or worsening - use minimum epochs
            return self.training_phase.phase4_min_epochs

        # Target cost after Phase 4
        target_cost = phase3_final_cost * (1.0 - self.training_phase.phase4_target_cost_reduction)
        remaining_reduction = phase3_final_cost - target_cost

        # Estimate epochs needed
        estimated_epochs = int(remaining_reduction / (effective_rate + 1e-8))

        # Clamp to configured bounds
        return max(
            self.training_phase.phase4_min_epochs,
            min(self.training_phase.phase4_max_epochs, estimated_epochs)
        )

    # ============ REWARD/PENALTY SYSTEM METHODS ============

    def _compute_improvement_metrics(
        self,
        cost_history: List[float],
        efficiency_history: List[float],
        window: int = 5
    ) -> Dict[str, float]:
        """
        Compute improvement metrics over a sliding window.

        Returns:
            Dict with keys:
            - cost_improvement: (old - new) / old (positive = improvement)
            - efficiency_improvement: new - old (positive = improvement)
            - cost_trend: average rate of change (negative = decreasing = good)
            - efficiency_trend: average rate of change (positive = increasing = good)
        """
        if len(cost_history) < 2:
            return {
                "cost_improvement": 0.0,
                "efficiency_improvement": 0.0,
                "cost_trend": 0.0,
                "efficiency_trend": 0.0
            }

        # Use most recent window for trend analysis
        recent_costs = cost_history[-min(window, len(cost_history)):]
        recent_efficiency = efficiency_history[-min(window, len(efficiency_history)):]

        # Cost improvement (epoch-over-epoch)
        cost_improvement = (cost_history[-2] - cost_history[-1]) / (cost_history[-2] + 1e-8)

        # Efficiency improvement (epoch-over-epoch)
        efficiency_improvement = efficiency_history[-1] - efficiency_history[-2] if len(efficiency_history) >= 2 else 0.0

        # Trend analysis over window
        if len(recent_costs) >= 2:
            cost_trend = (recent_costs[-1] - recent_costs[0]) / (len(recent_costs) * (recent_costs[0] + 1e-8))
        else:
            cost_trend = 0.0

        if len(recent_efficiency) >= 2:
            efficiency_trend = (recent_efficiency[-1] - recent_efficiency[0]) / len(recent_efficiency)
        else:
            efficiency_trend = 0.0

        return {
            "cost_improvement": cost_improvement,
            "efficiency_improvement": efficiency_improvement,
            "cost_trend": cost_trend,
            "efficiency_trend": efficiency_trend
        }

    def _should_reward(
        self,
        metrics: Dict[str, float],
        config: RewardPenaltyConfig
    ) -> Tuple[bool, float]:
        """
        Determine if current epoch deserves a reward.

        Returns:
            Tuple of (should_reward: bool, reward_magnitude: float)
        """
        # Reward conditions:
        # 1. Cost is decreasing
        # 2. Efficiency is good/improving
        # 3. Trend is positive overall
        cost_improving = metrics["cost_improvement"] > config.cost_improvement_threshold
        efficiency_good = metrics["efficiency_improvement"] >= 0 or metrics["efficiency_trend"] > 0
        trend_positive = metrics["cost_trend"] < 0  # Negative trend means cost decreasing

        should_reward = cost_improving and efficiency_good and trend_positive

        if should_reward:
            # Magnitude proportional to improvement
            magnitude = abs(metrics["cost_improvement"]) + abs(metrics["efficiency_improvement"]) * 0.5
            magnitude = min(magnitude, 1.0)  # Cap at 1.0
        else:
            magnitude = 0.0

        return should_reward, magnitude

    def _should_penalize(
        self,
        metrics: Dict[str, float],
        config: RewardPenaltyConfig
    ) -> Tuple[bool, float]:
        """
        Determine if current epoch deserves a penalty.

        Returns:
            Tuple of (should_penalize: bool, penalty_magnitude: float)
        """
        # Penalty conditions:
        # 1. Cost is increasing
        # 2. Efficiency is declining
        # 3. Trend is negative overall
        cost_degrading = metrics["cost_improvement"] < -config.cost_improvement_threshold
        efficiency_bad = metrics["efficiency_improvement"] < -config.efficiency_improvement_threshold
        trend_negative = metrics["cost_trend"] > 0  # Positive trend means cost increasing

        should_penalize = cost_degrading or (efficiency_bad and trend_negative)

        if should_penalize:
            # Magnitude proportional to degradation
            magnitude = abs(metrics["cost_improvement"]) + abs(metrics["efficiency_improvement"]) * 0.5
            magnitude = min(magnitude, 1.0)  # Cap at 1.0
        else:
            magnitude = 0.0

        return should_penalize, magnitude

    def _apply_reward(
        self,
        learning_rate: float,
        magnitude: float,
        config: RewardPenaltyConfig,
        emotional_state: EmotionalState
    ) -> float:
        """
        Apply reward by decreasing learning rate proportionally.

        The intuition: good progress means we're on the right track,
        so we can take smaller steps to fine-tune.

        Returns:
            Adjusted learning rate
        """
        # Decrease factor proportional to magnitude
        # magnitude of 1.0 -> multiply by min_adjustment_factor (0.5)
        # magnitude of 0.0 -> no change
        decrease_factor = 1.0 - magnitude * (1.0 - config.min_adjustment_factor)

        new_lr = learning_rate * decrease_factor

        # Enforce both bounds and validate
        new_lr = np.clip(new_lr, config.min_learning_rate, config.max_learning_rate)
        if not np.isfinite(new_lr) or new_lr <= 0:
            warnings.warn(
                f"Invalid learning rate {new_lr} after reward. Resetting to baseline.",
                RuntimeWarning
            )
            new_lr = config.baseline_learning_rate

        # Update emotional state
        emotional_state.total_rewards += 1
        emotional_state.reward_history.append(magnitude)

        return new_lr

    def _apply_penalty(
        self,
        learning_rate: float,
        magnitude: float,
        config: RewardPenaltyConfig,
        emotional_state: EmotionalState
    ) -> float:
        """
        Apply penalty by increasing learning rate proportionally.

        The intuition: poor progress means we might be stuck,
        so we need larger steps to escape.

        Returns:
            Adjusted learning rate
        """
        # Increase factor proportional to magnitude
        # magnitude of 1.0 -> multiply by max_adjustment_factor (2.0)
        # magnitude of 0.0 -> no change
        increase_factor = 1.0 + magnitude * (config.max_adjustment_factor - 1.0)

        new_lr = learning_rate * increase_factor

        # Enforce both bounds and validate
        new_lr = np.clip(new_lr, config.min_learning_rate, config.max_learning_rate)
        if not np.isfinite(new_lr) or new_lr <= 0:
            warnings.warn(
                f"Invalid learning rate {new_lr} after penalty. Resetting to baseline.",
                RuntimeWarning
            )
            new_lr = config.baseline_learning_rate

        # Update emotional state
        emotional_state.total_penalties += 1
        emotional_state.penalty_history.append(magnitude)

        return new_lr

    def _check_extreme_states(
        self,
        learning_rate: float,
        config: RewardPenaltyConfig,
        emotional_state: EmotionalState
    ) -> Tuple[float, str]:
        """
        Check for extreme emotional states and adjust LR if needed.

        Returns:
            Tuple of (adjusted_lr, state_description)
        """
        depression = emotional_state.depression_ratio
        excitement = emotional_state.excitement_ratio

        state = "neutral"

        # Check for extreme states - full reset to baseline
        if depression > config.extreme_threshold:
            learning_rate = config.baseline_learning_rate
            emotional_state.lr_reset_count += 1
            state = "extreme_depression"
        elif excitement > config.extreme_threshold:
            learning_rate = config.baseline_learning_rate
            emotional_state.lr_reset_count += 1
            state = "extreme_excitement"

        # Check for moderate states - partial correction toward baseline
        elif depression > config.moderate_threshold:
            # Move 50% toward baseline
            learning_rate = learning_rate + 0.5 * (config.baseline_learning_rate - learning_rate)
            state = "depressed"
        elif excitement > config.moderate_threshold:
            # Move 50% toward baseline
            learning_rate = learning_rate + 0.5 * (config.baseline_learning_rate - learning_rate)
            state = "excited"

        # Record history
        emotional_state.depression_history.append(depression)
        emotional_state.excitement_history.append(excitement)

        return learning_rate, state

    def _apply_reward_penalty_system(
        self,
        learning_rate: float,
        cost_history: List[float],
        efficiency_history: List[float],
        config: RewardPenaltyConfig,
        emotional_state: EmotionalState
    ) -> Tuple[float, str]:
        """
        Apply the complete reward/penalty system for one epoch.

        Returns:
            Tuple of (new_learning_rate, action_taken)
            action_taken: "reward", "penalty", "neutral", "extreme_depression_reset", "extreme_excitement_reset"
        """
        # Compute metrics
        metrics = self._compute_improvement_metrics(cost_history, efficiency_history, config.window_size)

        action = "neutral"

        # Check for reward
        should_reward, reward_mag = self._should_reward(metrics, config)
        if should_reward:
            learning_rate = self._apply_reward(learning_rate, reward_mag, config, emotional_state)
            action = "reward"
        else:
            # Check for penalty
            should_penalize, penalty_mag = self._should_penalize(metrics, config)
            if should_penalize:
                learning_rate = self._apply_penalty(learning_rate, penalty_mag, config, emotional_state)
                action = "penalty"

        # Check for extreme states (may override previous adjustment)
        learning_rate, extreme_state = self._check_extreme_states(learning_rate, config, emotional_state)
        if extreme_state.startswith("extreme"):
            action = f"{extreme_state}_reset"

        return learning_rate, action

    # ============ END REWARD/PENALTY SYSTEM METHODS ============

    def _safe_callback(
        self,
        callback: Optional[Callable],
        epoch: int,
        cost: float,
        efficiency: float
    ) -> None:
        """Safely invoke callback, catching exceptions to prevent training crashes."""
        if callback is None:
            return
        try:
            callback(epoch, cost, efficiency)
        except Exception as e:
            warnings.warn(
                f"Callback raised exception at epoch {epoch}: {e}. Training will continue.",
                RuntimeWarning
            )

    def _apply_random_perturbation(self, fraction: float = None) -> None:
        """Apply random perturbation to fraction of nodes to avoid overfitting."""
        if fraction is None:
            fraction = self.perturbation.perturbation_fraction
        total_nodes = self._count_nodes()
        num_to_perturb = max(1, int(total_nodes * fraction))

        for _ in range(num_to_perturb):
            layer_idx = np.random.randint(len(self._layers))
            node_idx = np.random.randint(self._layers[layer_idx]["W"].shape[0])

            # Add small random perturbation
            perturbation_scale = self.perturbation.perturbation_scale
            self._layers[layer_idx]["W"][node_idx] *= (1 + np.random.randn() * perturbation_scale)
            self._layers[layer_idx]["b"][node_idx] *= (1 + np.random.randn() * perturbation_scale)

    def _aggressive_layer_adjustment(self, saturation_threshold: float = None,
                                     efficiency_threshold: float = None) -> Tuple[int, int, int]:
        """Aggressive layer/node adjustment for exploration phase."""
        if saturation_threshold is None:
            saturation_threshold = self.efficiency.exploration_saturation_threshold
        if efficiency_threshold is None:
            efficiency_threshold = self.efficiency.exploration_efficiency_threshold

        nodes_added = 0
        nodes_removed = 0
        layer_change = 0
        max_nodes = self.architecture.max_nodes_per_layer
        min_nodes = self.architecture.min_nodes_to_keep
        growth_rate = self.architecture.exploration_growth_rate
        removal_mult = self.efficiency.removal_efficiency_multiplier

        for i in range(len(self._layers) - 1):  # Don't modify output layer
            layer = self._layers[i]
            avg_efficiency = np.mean(layer["efficiency"])

            # Add nodes if layer is saturated
            if avg_efficiency > saturation_threshold and layer["W"].shape[0] < max_nodes:
                num_new = max(1, int(layer["W"].shape[0] * growth_rate))
                self._add_nodes_to_layer(i, num_new)
                nodes_added += num_new

            # Remove inefficient nodes
            inefficient_mask = layer["efficiency"] < efficiency_threshold * removal_mult
            if np.sum(inefficient_mask) > 0 and layer["W"].shape[0] > min_nodes:
                max_this_epoch = max(1, int(layer["W"].shape[0] * self.efficiency.max_removal_fraction_per_epoch))
                num_to_remove = min(np.sum(inefficient_mask), layer["W"].shape[0] - min_nodes, max_this_epoch)
                if num_to_remove > 0:
                    self._remove_nodes_from_layer(i, int(num_to_remove))
                    nodes_removed += int(num_to_remove)

        # Consider adding a layer if all hidden layers are saturated
        if len(self._layers) < self.architecture.max_layers:
            all_saturated = all(
                np.mean(self._layers[i]["efficiency"]) > saturation_threshold
                for i in range(len(self._layers) - 1)
            )
            if all_saturated:
                self._add_layer()
                layer_change = 1

        return nodes_added, nodes_removed, layer_change

    def _layer_adjustment(self, saturation_threshold: float,
                         efficiency_threshold: float) -> Tuple[int, int, int]:
        """Standard layer/node adjustment for main training phase."""
        nodes_added = 0
        nodes_removed = 0
        layer_change = 0
        max_nodes = self.architecture.max_nodes_per_layer
        growth_rate = self.architecture.main_growth_rate

        for i in range(len(self._layers) - 1):
            layer = self._layers[i]
            avg_efficiency = np.mean(layer["efficiency"])

            # Add nodes if layer is saturated
            if avg_efficiency > saturation_threshold and layer["W"].shape[0] < max_nodes:
                num_new = max(1, int(layer["W"].shape[0] * growth_rate))
                self._add_nodes_to_layer(i, num_new)
                nodes_added += num_new

        return nodes_added, nodes_removed, layer_change

    def _add_nodes_to_layer(self, layer_idx: int, num_nodes: int) -> None:
        """Add nodes to a layer."""
        layer = self._layers[layer_idx]
        input_size = layer["W"].shape[1]
        output_size = layer["W"].shape[0]

        # Initialize new weights
        new_W = np.random.randn(num_nodes, input_size).astype(np.float32) * np.sqrt(2.0 / input_size)
        new_b = np.zeros(num_nodes, dtype=np.float32)
        new_eff = np.ones(num_nodes) * self.efficiency.initial_efficiency

        # Append to layer
        layer["W"] = np.vstack([layer["W"], new_W])
        layer["b"] = np.concatenate([layer["b"], new_b])
        layer["efficiency"] = np.concatenate([layer["efficiency"], new_eff])

        # Add efficiency weights for new nodes
        if layer.get("efficiency_weights") is not None:
            # Copy threshold from existing nodes if available
            existing_threshold = layer["efficiency_weights"][0].grad_threshold if layer["efficiency_weights"] else 0.1
            for _ in range(num_nodes):
                new_weights = EfficiencyWeights()
                new_weights.grad_threshold = existing_threshold
                layer["efficiency_weights"].append(new_weights)

        # Expand activation stats arrays
        if layer.get("activation_stats") is not None:
            n = layer["W"].shape[0]
            layer["activation_stats"]["sum"] = np.concatenate([layer["activation_stats"]["sum"], np.zeros(num_nodes)])
            layer["activation_stats"]["sq_sum"] = np.concatenate([layer["activation_stats"]["sq_sum"], np.zeros(num_nodes)])
            layer["activation_stats"]["dead_count"] = np.concatenate([layer["activation_stats"]["dead_count"], np.zeros(num_nodes, dtype=np.int64)])

        # Expand gradient stats arrays
        if layer.get("gradient_stats") is not None:
            layer["gradient_stats"]["sum"] = np.concatenate([layer["gradient_stats"]["sum"], np.zeros(num_nodes)])

        # Update next layer's input size
        if layer_idx < len(self._layers) - 1:
            next_layer = self._layers[layer_idx + 1]
            # Use He initialization with correct fan-in (total nodes in current layer after addition)
            fan_in = layer["W"].shape[0]  # New total nodes in current layer
            std = np.sqrt(2.0 / fan_in)
            new_cols = np.random.randn(next_layer["W"].shape[0], num_nodes).astype(np.float32) * std
            next_layer["W"] = np.hstack([next_layer["W"], new_cols])

    def _remove_nodes_from_layer(self, layer_idx: int, num_nodes: int) -> None:
        """Remove least efficient nodes from a layer."""
        layer = self._layers[layer_idx]
        n = layer["W"].shape[0]
        efficiency = layer["efficiency"]

        # Bounds check: ensure we don't remove too many nodes
        min_nodes = 4
        if num_nodes >= len(efficiency) - min_nodes:
            num_nodes = len(efficiency) - min_nodes
            if num_nodes <= 0:
                return  # Can't remove any nodes

        num_to_keep = n - num_nodes

        if num_to_keep < min_nodes:
            return  # Keep minimum nodes

        # Use argpartition for O(n) partial sort instead of O(n log n) argsort
        # argpartition places the smallest num_nodes elements at the front (unsorted)
        partition_indices = np.argpartition(efficiency, num_nodes)
        indices_to_keep = partition_indices[num_nodes:]  # Keep the larger efficiency nodes

        # Remove from current layer
        layer["W"] = layer["W"][indices_to_keep]
        layer["b"] = layer["b"][indices_to_keep]
        layer["efficiency"] = layer["efficiency"][indices_to_keep]

        # Update efficiency weights
        if layer.get("efficiency_weights") is not None:
            layer["efficiency_weights"] = [layer["efficiency_weights"][i] for i in indices_to_keep]

        # Update activation stats
        if layer.get("activation_stats") is not None:
            layer["activation_stats"]["sum"] = layer["activation_stats"]["sum"][indices_to_keep]
            layer["activation_stats"]["sq_sum"] = layer["activation_stats"]["sq_sum"][indices_to_keep]
            layer["activation_stats"]["dead_count"] = layer["activation_stats"]["dead_count"][indices_to_keep]

        # Update gradient stats
        if layer.get("gradient_stats") is not None:
            layer["gradient_stats"]["sum"] = layer["gradient_stats"]["sum"][indices_to_keep]

        # Update next layer's input
        if layer_idx < len(self._layers) - 1:
            next_layer = self._layers[layer_idx + 1]
            next_layer["W"] = next_layer["W"][:, indices_to_keep]

    def _add_layer(self) -> None:
        """Add a new hidden layer."""
        if len(self._layers) >= self.architecture.max_layers:
            return

        # Insert before output layer
        insert_idx = len(self._layers) - 1
        prev_layer = self._layers[insert_idx - 1] if insert_idx > 0 else self._layers[0]
        next_layer = self._layers[insert_idx]

        # New layer size is geometric mean
        prev_size = prev_layer["W"].shape[0]
        next_input = next_layer["W"].shape[1]
        new_size = int(np.sqrt(prev_size * next_input))
        new_size = max(self.architecture.min_initial_hidden_size, new_size)

        # Create new layer
        new_W = np.random.randn(new_size, prev_size).astype(np.float32) * np.sqrt(2.0 / prev_size)
        new_b = np.zeros(new_size, dtype=np.float32)
        new_eff = np.ones(new_size) * self.efficiency.initial_efficiency

        # Get existing gradient threshold from previous layers
        existing_threshold = 0.1
        for layer in self._layers:
            if layer.get("efficiency_weights") is not None and len(layer["efficiency_weights"]) > 0:
                existing_threshold = layer["efficiency_weights"][0].grad_threshold
                break

        new_layer = {
            "W": new_W,
            "b": new_b,
            "efficiency": new_eff,
            "efficiency_weights": [EfficiencyWeights() for _ in range(new_size)],
            "activation_stats": {
                "sum": np.zeros(new_size),
                "sq_sum": np.zeros(new_size),
                "count": 0,
                "dead_count": np.zeros(new_size, dtype=np.int64)
            },
            "gradient_stats": {
                "sum": np.zeros(new_size),
                "count": 0
            }
        }

        # Set threshold for all new nodes
        for weights in new_layer["efficiency_weights"]:
            weights.grad_threshold = existing_threshold

        # Update next layer's input
        next_layer["W"] = np.random.randn(next_layer["W"].shape[0], new_size).astype(np.float32) * np.sqrt(2.0 / new_size)

        # Insert new layer
        self._layers.insert(insert_idx, new_layer)

    def _save_architecture(self) -> Dict:
        """Save current architecture for restoration."""
        return {
            'layers': [
                {
                    'W': layer['W'].copy(),
                    'b': layer['b'].copy(),
                    'efficiency': layer['efficiency'].copy() if layer['efficiency'] is not None else None
                }
                for layer in self._layers
            ]
        }

    def _restore_architecture(self, saved: Dict) -> None:
        """Restore architecture from saved state."""
        self._layers = [
            {
                'W': layer['W'].copy(),
                'b': layer['b'].copy(),
                'efficiency': layer['efficiency'].copy() if layer['efficiency'] is not None else None
            }
            for layer in saved['layers']
        ]

    def predict(self, X: np.ndarray, denormalize: bool = True) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input data array
            denormalize: Whether to denormalize output to original scale (default: True)
                        Set to False if you want raw network output.

        Returns:
            Predictions array (denormalized to original scale by default for regression)
        """
        X = X.astype(np.float32)

        # Normalize input using stored parameters
        X = self._normalize_input(X)

        if self._use_cpp:
            from . import _dnn_core
            outputs = []
            for x in X:
                # PyNetwork::predict accepts a numpy array directly
                # and returns a numpy array (not a Tensor).
                output = self._network.predict(
                    np.ascontiguousarray(x.flatten(), dtype=np.float32)
                )
                outputs.append(np.asarray(output))
            output = np.array(outputs)
        else:
            # Python fallback - works with multi-layer dynamic architecture
            a = X.reshape(len(X), -1)
            for i, layer in enumerate(self._layers):
                z = a @ layer["W"].T + layer["b"]
                if i < len(self._layers) - 1:
                    # ReLU for hidden layers
                    a = np.maximum(0, z)
                else:
                    # Output layer activation depends on cost function
                    if self.cost_function == "MSE":
                        # Linear activation for regression
                        a = z
                    else:
                        # Softmax for classification
                        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
                        a = exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-8)
            output = a

        # Denormalize output for regression tasks
        if denormalize:
            output = self._denormalize_output(output)

        return output

    def health_status(self) -> HealthReport:
        """Get health status including emotional state (cancer/alzheimer/depression/excitement)."""
        # Compute emotional metrics from last training
        depression = 0.0
        excitement = 0.0
        emotional_state = "neutral"

        if self._training_result:
            total = self._training_result.total_rewards + self._training_result.total_penalties
            if total > 0:
                depression = self._training_result.total_penalties / total
                excitement = self._training_result.total_rewards / total

            if depression > 0.8:
                emotional_state = "extreme_depression"
            elif excitement > 0.8:
                emotional_state = "extreme_excitement"
            elif depression > 0.5:
                emotional_state = "depressed"
            elif excitement > 0.5:
                emotional_state = "excited"

        if self._use_cpp:
            from . import _dnn_core
            report = self._network.health_report()
            return HealthReport(
                state=str(report.state),
                cancer_score=report.cancer_score,
                alzheimer_score=report.alzheimer_score,
                overall_health=report.overall_health,
                diagnosis=report.diagnosis,
                recommendations=list(report.recommendations),
                current_layers=report.current_layers,
                current_nodes=report.current_nodes,
                depression_ratio=depression,
                excitement_ratio=excitement,
                emotional_state=emotional_state
            )
        else:
            return HealthReport(
                state="Healthy",
                cancer_score=0.0,
                alzheimer_score=0.0,
                overall_health=1.0,
                diagnosis="Network is healthy",
                recommendations=[],
                current_layers=len(self._layers),
                current_nodes=sum(l["W"].shape[0] for l in self._layers),
                depression_ratio=depression,
                excitement_ratio=excitement,
                emotional_state=emotional_state
            )

    def efficiency_report(self, include_nodes: bool = False) -> Dict[str, Any]:
        """Get efficiency report for network."""
        if self._use_cpp:
            from . import _dnn_core
            return self._network.efficiency_report(include_nodes)
        else:
            return {
                "avg_layer_efficiency": 0.5,
                "total_layers": len(self._layers),
                "total_nodes": sum(l["W"].shape[0] for l in self._layers),
                "total_parameters": sum(l["W"].size + l["b"].size for l in self._layers)
            }

    def to(self, device: str) -> "DynamicNetwork":
        """
        Move network to specified device.

        Args:
            device: Target device ("cpu" or "cuda")

        Returns:
            self for method chaining
        """
        device = device.lower()
        if device == "gpu":
            device = "cuda"
        if device not in ("cpu", "cuda"):
            raise ValueError(f"Unknown device: {device}. Use 'cpu' or 'cuda'")

        if self._use_cpp:
            from . import _dnn_core
            if device == "cuda" and not _dnn_core.cuda_available():
                raise RuntimeError(
                    "CUDA requested but not available. "
                    "Rebuild pydnn with DNN_ENABLE_CUDA=ON"
                )
            target_device = _dnn_core.Device.CUDA if device == "cuda" else _dnn_core.Device.CPU
            self._network.to(target_device)

        self._device = device
        return self

    @property
    def device(self) -> str:
        """Current device ('cpu' or 'cuda')."""
        return self._device

    @property
    def num_layers(self) -> int:
        """Number of layers."""
        if self._use_cpp:
            return self._network.num_layers()
        return len(self._layers)

    @property
    def num_parameters(self) -> int:
        """Total number of parameters."""
        if self._use_cpp:
            return self._network.num_parameters()
        return sum(l["W"].size + l["b"].size for l in self._layers)

    def save(self, path: str, name: str = "model"):
        """
        Save model to files.

        Creates:
        - {name}.json: Model architecture and metadata
        - {name}_layer{i}_W.npy: Weight files for each layer
        - {name}_layer{i}_b.npy: Bias files for each layer

        Args:
            path: Directory to save to
            name: Model name

        Raises:
            IOError: If files cannot be written
            PermissionError: If directory is not writable
        """
        import os

        # Validate and create directory
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            raise IOError(f"Cannot create directory {path}: {e}")

        # Check write permissions
        if not os.access(path, os.W_OK):
            raise PermissionError(f"Directory {path} is not writable")

        if self._use_cpp:
            from . import _dnn_core
            _dnn_core.save_model(self._network, path, name)
        else:
            # Python fallback - save as numpy files
            import json

            metadata = {
                "version": "1.0",
                "seed": self.seed,
                "input_shape": self.input_shape,
                "output_size": self.output_size,
                "cost_function": self.cost_function,
                "num_layers": len(self._layers)
            }

            json_path = os.path.join(path, f"{name}.json")
            saved_files = []

            try:
                with open(json_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                saved_files.append(json_path)

                for i, layer in enumerate(self._layers):
                    w_path = os.path.join(path, f"{name}_layer{i}_W.npy")
                    b_path = os.path.join(path, f"{name}_layer{i}_b.npy")
                    np.save(w_path, layer["W"])
                    saved_files.append(w_path)
                    np.save(b_path, layer["b"])
                    saved_files.append(b_path)

            except (IOError, OSError) as e:
                # Clean up partial save on failure
                for file_path in saved_files:
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass
                raise IOError(f"Failed to save model: {e}")

    @classmethod
    def load(cls, path: str) -> "DynamicNetwork":
        """
        Load a saved model.

        Args:
            path: Path to model file or directory containing model files

        Returns:
            Loaded DynamicNetwork

        Raises:
            FileNotFoundError: If model file not found
            IOError: If files cannot be read
            ValueError: If model format is invalid or corrupted
        """
        import os
        import json

        # Try to load JSON metadata
        json_path = path if path.endswith(".json") else path.replace(".txt", ".json")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Model file not found: {json_path}")

        try:
            with open(json_path) as f:
                metadata = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in model file {json_path}: {e}")
        except IOError as e:
            raise IOError(f"Cannot read model file {json_path}: {e}")

        # Validate metadata
        required_keys = ["input_shape", "output_size", "seed", "cost_function", "num_layers"]
        missing = [k for k in required_keys if k not in metadata]
        if missing:
            raise ValueError(f"Model file missing required keys: {missing}")

        try:
            network = cls(
                input_shape=tuple(metadata["input_shape"]),
                output_size=metadata["output_size"],
                seed=metadata["seed"],
                cost_function=metadata["cost_function"]
            )
        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid model metadata: {e}")

        # Load weights
        base_path = os.path.dirname(json_path) or "."
        name = os.path.basename(json_path).replace(".json", "")

        network._layers = []
        for i in range(metadata["num_layers"]):
            w_path = os.path.join(base_path, f"{name}_layer{i}_W.npy")
            b_path = os.path.join(base_path, f"{name}_layer{i}_b.npy")

            if not os.path.exists(w_path):
                raise FileNotFoundError(f"Weight file not found: {w_path}")
            if not os.path.exists(b_path):
                raise FileNotFoundError(f"Bias file not found: {b_path}")

            try:
                W = np.load(w_path)
                b = np.load(b_path)
            except Exception as e:
                raise IOError(f"Failed to load layer {i} weights: {e}")

            network._layers.append({
                "W": W,
                "b": b,
                "efficiency": np.ones(W.shape[0]) * 0.5  # Initialize efficiency
            })

        network._trained = True
        return network
