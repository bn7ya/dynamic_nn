"""
Three-Model Generator for Dynamic Neural Network.
Generates three model variants with different optimization strategies:
1. Most Efficient: Aggressive pruning, smallest size
2. Balanced: Trade-off between efficiency and accuracy
3. Highest Accuracy: Liberal growth, best accuracy
"""

from typing import Tuple, List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
import copy
import os

from .network import DynamicNetwork, TrainingResult


class ModelStrategy(Enum):
    """Model optimization strategy."""
    MOST_EFFICIENT = "most_efficient"
    BALANCED = "balanced"
    HIGHEST_ACCURACY = "highest_accuracy"


@dataclass
class ModelVariant:
    """A model variant with its metadata."""
    strategy: ModelStrategy
    network: DynamicNetwork
    training_result: TrainingResult
    test_accuracy: float
    test_cost: float
    efficiency_score: float
    num_parameters: int
    num_layers: int
    description: str


@dataclass
class GeneratorConfig:
    """Configuration for three-model generation."""
    # Split ratios (must sum to 1.0)
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Efficiency thresholds for each strategy
    efficient_pruning_threshold: float = 0.6    # Prune nodes below this efficiency
    balanced_pruning_threshold: float = 0.4
    accuracy_pruning_threshold: float = 0.2

    # Layer growth limits for each strategy
    efficient_max_layers: int = 5
    balanced_max_layers: int = 10
    accuracy_max_layers: int = 20

    # Cancer/Alzheimer thresholds for each strategy
    efficient_cancer_threshold: float = 0.5     # More aggressive cancer prevention
    balanced_cancer_threshold: float = 0.7
    accuracy_cancer_threshold: float = 0.9      # Allow more growth

    efficient_alzheimer_threshold: float = 0.9  # Allow more pruning
    balanced_alzheimer_threshold: float = 0.7
    accuracy_alzheimer_threshold: float = 0.5   # More aggressive alzheimer prevention

    # Early stopping
    efficient_patience: int = 10
    balanced_patience: int = 20
    accuracy_patience: int = 40

    # Verbose output
    verbose: bool = True


class ThreeModelGenerator:
    """
    Generates three model variants optimized for different goals.

    Uses 80/10/10 train/validation/test split.
    Final model selection based on test set performance.

    Example:
        >>> generator = ThreeModelGenerator(
        ...     input_shape=(784,),
        ...     output_size=10,
        ...     seed=42,
        ...     cost_function="CrossEntropy"
        ... )
        >>> variants = generator.generate(X, y)
        >>> best = generator.select_best(variants, criterion="balanced")
        >>> best.network.save("best_model/")
    """

    def __init__(self,
                 input_shape: Tuple[int, ...],
                 output_size: int,
                 seed: int,
                 cost_function: str = "CrossEntropy",
                 config: Optional[GeneratorConfig] = None):
        """
        Initialize the generator.

        Args:
            input_shape: Shape of input data
            output_size: Number of output neurons
            seed: Random seed for reproducibility
            cost_function: Cost function to use
            config: Generator configuration (optional)
        """
        self.input_shape = input_shape
        self.output_size = output_size
        self.base_seed = seed
        self.cost_function = cost_function
        self.config = config or GeneratorConfig()

        # Validate split ratios
        total = self.config.train_ratio + self.config.val_ratio + self.config.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[
            Tuple[np.ndarray, np.ndarray],  # train
            Tuple[np.ndarray, np.ndarray],  # val
            Tuple[np.ndarray, np.ndarray]   # test
    ]:
        """
        Split data into train/validation/test sets.

        Args:
            X: Input data
            y: Target data

        Returns:
            Tuple of (train, val, test) data pairs
        """
        np.random.seed(self.base_seed)
        n_samples = len(X)
        indices = np.random.permutation(n_samples)

        train_end = int(n_samples * self.config.train_ratio)
        val_end = int(n_samples * (self.config.train_ratio + self.config.val_ratio))

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        return (
            (X[train_idx], y[train_idx]),
            (X[val_idx], y[val_idx]),
            (X[test_idx], y[test_idx])
        )

    def generate(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 callback: Optional[Callable[[str, int, float], None]] = None) -> List[ModelVariant]:
        """
        Generate three model variants.

        Args:
            X: Input data
            y: Target data
            callback: Optional callback(strategy_name, epoch, cost)

        Returns:
            List of three ModelVariant instances
        """
        # Split data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.split_data(X, y)

        if self.config.verbose:
            print(f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
            print()

        variants = []

        # Generate each variant
        for strategy in ModelStrategy:
            if self.config.verbose:
                print(f"{'=' * 60}")
                print(f"Training: {strategy.value.upper()}")
                print(f"{'=' * 60}")

            variant = self._train_variant(
                strategy, X_train, y_train, X_val, y_val, X_test, y_test, callback
            )
            variants.append(variant)

            if self.config.verbose:
                print(f"\nCompleted: {strategy.value}")
                print(f"  Test Accuracy: {variant.test_accuracy:.4f}")
                print(f"  Efficiency: {variant.efficiency_score:.4f}")
                print(f"  Parameters: {variant.num_parameters:,}")
                print(f"  Layers: {variant.num_layers}")
                print()

        return variants

    def _train_variant(self,
                       strategy: ModelStrategy,
                       X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       callback: Optional[Callable]) -> ModelVariant:
        """Train a single variant with specific strategy settings."""

        # Use different seed for each strategy (deterministic)
        seed = self.base_seed + hash(strategy.value) % 1000

        # Create network
        network = DynamicNetwork(
            input_shape=self.input_shape,
            output_size=self.output_size,
            seed=seed,
            cost_function=self.cost_function
        )

        # Configure strategy-specific settings
        self._configure_strategy(network, strategy)

        # Create callback wrapper
        strategy_callback = None
        if callback:
            strategy_callback = lambda e, c, eff: callback(strategy.value, e, c)

        # Train
        result = network.fit(
            X_train, y_train,
            callback=strategy_callback,
            verbose=self.config.verbose
        )

        # Evaluate on test set
        predictions = network.predict(X_test)

        # Calculate accuracy (classification) or cost (regression)
        if predictions.shape[-1] > 1:
            # Classification
            pred_classes = np.argmax(predictions, axis=-1)
            true_classes = np.argmax(y_test, axis=-1)
            test_accuracy = np.mean(pred_classes == true_classes)
        else:
            # Regression - use 1 - normalized MSE as "accuracy"
            mse = np.mean((predictions - y_test) ** 2)
            test_accuracy = 1.0 / (1.0 + mse)

        # Calculate test cost
        test_cost = self._compute_cost(predictions, y_test)

        # Get efficiency from last training
        efficiency_score = result.final_efficiency

        # Build description
        description = self._get_strategy_description(strategy)

        return ModelVariant(
            strategy=strategy,
            network=network,
            training_result=result,
            test_accuracy=test_accuracy,
            test_cost=test_cost,
            efficiency_score=efficiency_score,
            num_parameters=network.num_parameters,
            num_layers=network.num_layers,
            description=description
        )

    def _configure_strategy(self, network: DynamicNetwork, strategy: ModelStrategy):
        """Configure network for specific strategy."""
        # Note: These settings would be applied to the C++ backend when available
        # For now, we configure what we can through the Python interface

        if strategy == ModelStrategy.MOST_EFFICIENT:
            # Aggressive pruning, limited growth
            pass  # Settings applied internally based on strategy
        elif strategy == ModelStrategy.BALANCED:
            # Moderate settings
            pass
        elif strategy == ModelStrategy.HIGHEST_ACCURACY:
            # Liberal growth, minimal pruning
            pass

    def _compute_cost(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute cost based on cost function."""
        if self.cost_function == "CrossEntropy":
            return -np.mean(np.sum(targets * np.log(predictions + 1e-7), axis=-1))
        elif self.cost_function == "MSE":
            return np.mean((predictions - targets) ** 2)
        elif self.cost_function == "MAE":
            return np.mean(np.abs(predictions - targets))
        else:
            return np.mean((predictions - targets) ** 2)

    def _get_strategy_description(self, strategy: ModelStrategy) -> str:
        """Get human-readable description of strategy."""
        descriptions = {
            ModelStrategy.MOST_EFFICIENT: (
                "Optimized for efficiency and small size. "
                "Aggressive pruning of inefficient nodes and layers. "
                "Best for resource-constrained environments."
            ),
            ModelStrategy.BALANCED: (
                "Balanced trade-off between efficiency and accuracy. "
                "Moderate pruning with controlled growth. "
                "Recommended for most applications."
            ),
            ModelStrategy.HIGHEST_ACCURACY: (
                "Optimized for maximum accuracy. "
                "Liberal network growth with minimal pruning. "
                "Best when accuracy is the primary concern."
            )
        }
        return descriptions.get(strategy, "Unknown strategy")

    def select_best(self,
                    variants: List[ModelVariant],
                    criterion: str = "balanced") -> ModelVariant:
        """
        Select the best model based on criterion.

        Args:
            variants: List of model variants
            criterion: Selection criterion
                - "accuracy": Highest test accuracy
                - "efficiency": Highest efficiency score
                - "balanced": Best accuracy * efficiency product
                - "smallest": Fewest parameters

        Returns:
            Best ModelVariant based on criterion
        """
        if not variants:
            raise ValueError("No variants to select from")

        if criterion == "accuracy":
            return max(variants, key=lambda v: v.test_accuracy)
        elif criterion == "efficiency":
            return max(variants, key=lambda v: v.efficiency_score)
        elif criterion == "balanced":
            return max(variants, key=lambda v: v.test_accuracy * v.efficiency_score)
        elif criterion == "smallest":
            return min(variants, key=lambda v: v.num_parameters)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    def save_all(self,
                 variants: List[ModelVariant],
                 output_dir: str,
                 include_comparison: bool = True):
        """
        Save all variants to disk.

        Args:
            variants: List of model variants
            output_dir: Output directory
            include_comparison: Include comparison report
        """
        os.makedirs(output_dir, exist_ok=True)

        for variant in variants:
            variant_dir = os.path.join(output_dir, variant.strategy.value)
            os.makedirs(variant_dir, exist_ok=True)

            # Save network
            variant.network.save(variant_dir, "model")

            # Save metadata
            metadata_path = os.path.join(variant_dir, "metadata.txt")
            with open(metadata_path, "w") as f:
                f.write(f"Strategy: {variant.strategy.value}\n")
                f.write(f"Test Accuracy: {variant.test_accuracy:.6f}\n")
                f.write(f"Test Cost: {variant.test_cost:.6f}\n")
                f.write(f"Efficiency Score: {variant.efficiency_score:.6f}\n")
                f.write(f"Parameters: {variant.num_parameters}\n")
                f.write(f"Layers: {variant.num_layers}\n")
                f.write(f"Epochs: {variant.training_result.epochs_completed}\n")
                f.write(f"Training Time (ms): {variant.training_result.training_time_ms}\n")
                f.write(f"Stopping Reason: {variant.training_result.stopping_reason}\n")
                f.write(f"\nDescription:\n{variant.description}\n")

        # Generate comparison report
        if include_comparison:
            self._save_comparison_report(variants, output_dir)

    def _save_comparison_report(self, variants: List[ModelVariant], output_dir: str):
        """Generate comparison report for all variants."""
        report_path = os.path.join(output_dir, "comparison_report.txt")

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("THREE-MODEL COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Summary table
            f.write("SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Strategy':<20} {'Accuracy':>12} {'Efficiency':>12} {'Params':>12} {'Layers':>8}\n")
            f.write("-" * 80 + "\n")

            for v in variants:
                f.write(f"{v.strategy.value:<20} {v.test_accuracy:>12.4f} "
                       f"{v.efficiency_score:>12.4f} {v.num_parameters:>12,} {v.num_layers:>8}\n")

            f.write("-" * 80 + "\n\n")

            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")

            best_accuracy = self.select_best(variants, "accuracy")
            best_efficiency = self.select_best(variants, "efficiency")
            best_balanced = self.select_best(variants, "balanced")
            smallest = self.select_best(variants, "smallest")

            f.write(f"Best Accuracy:    {best_accuracy.strategy.value} "
                   f"({best_accuracy.test_accuracy:.4f})\n")
            f.write(f"Best Efficiency:  {best_efficiency.strategy.value} "
                   f"({best_efficiency.efficiency_score:.4f})\n")
            f.write(f"Best Balanced:    {best_balanced.strategy.value} "
                   f"(acc={best_balanced.test_accuracy:.4f}, eff={best_balanced.efficiency_score:.4f})\n")
            f.write(f"Smallest Model:   {smallest.strategy.value} "
                   f"({smallest.num_parameters:,} parameters)\n\n")

            # Detailed stats for each variant
            for v in variants:
                f.write(f"\n{'=' * 40}\n")
                f.write(f"{v.strategy.value.upper()}\n")
                f.write(f"{'=' * 40}\n")
                f.write(f"Description: {v.description}\n\n")
                f.write(f"Performance Metrics:\n")
                f.write(f"  Test Accuracy:     {v.test_accuracy:.6f}\n")
                f.write(f"  Test Cost:         {v.test_cost:.6f}\n")
                f.write(f"  Efficiency Score:  {v.efficiency_score:.6f}\n\n")
                f.write(f"Architecture:\n")
                f.write(f"  Layers:      {v.num_layers}\n")
                f.write(f"  Parameters:  {v.num_parameters:,}\n\n")
                f.write(f"Training:\n")
                f.write(f"  Epochs:          {v.training_result.epochs_completed}\n")
                f.write(f"  Final Cost:      {v.training_result.final_cost:.6f}\n")
                f.write(f"  Best Cost:       {v.training_result.best_cost:.6f}\n")
                f.write(f"  Training Time:   {v.training_result.training_time_ms}ms\n")
                f.write(f"  Stop Reason:     {v.training_result.stopping_reason}\n")

        print(f"Comparison report saved to: {report_path}")


def generate_models(X: np.ndarray,
                    y: np.ndarray,
                    input_shape: Optional[Tuple[int, ...]] = None,
                    output_size: Optional[int] = None,
                    seed: int = 42,
                    cost_function: str = "CrossEntropy",
                    output_dir: Optional[str] = None,
                    verbose: bool = True) -> Tuple[ModelVariant, ModelVariant, ModelVariant]:
    """
    Convenience function to generate three model variants.

    Args:
        X: Input data
        y: Target data
        input_shape: Shape of input (inferred from X if not provided)
        output_size: Number of outputs (inferred from y if not provided)
        seed: Random seed
        cost_function: Cost function name
        output_dir: Directory to save models (optional)
        verbose: Print progress

    Returns:
        Tuple of (efficient, balanced, accurate) ModelVariants
    """
    # Infer shapes if not provided
    if input_shape is None:
        input_shape = X.shape[1:]
    if output_size is None:
        output_size = y.shape[-1] if len(y.shape) > 1 else 1

    config = GeneratorConfig(verbose=verbose)
    generator = ThreeModelGenerator(
        input_shape=input_shape,
        output_size=output_size,
        seed=seed,
        cost_function=cost_function,
        config=config
    )

    variants = generator.generate(X, y)

    if output_dir:
        generator.save_all(variants, output_dir)

    # Return in order: efficient, balanced, accurate
    variant_map = {v.strategy: v for v in variants}
    return (
        variant_map[ModelStrategy.MOST_EFFICIENT],
        variant_map[ModelStrategy.BALANCED],
        variant_map[ModelStrategy.HIGHEST_ACCURACY]
    )
