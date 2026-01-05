#!/usr/bin/env python3
"""
MNIST Classification Example

Demonstrates the Dynamic Neural Network on the MNIST handwritten digit dataset.
The network automatically adapts its architecture during training.

Features demonstrated:
- Automatic architecture adaptation
- Cancer/Alzheimer health monitoring
- Efficiency-based early stopping
- Training visualization
- Three-model generation
"""

import numpy as np
import sys
import os

# Add pydnn to path if not installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from pydnn import (
    DynamicNetwork,
    generate_models,
    auto_generate_plots,
    COST_FUNCTIONS,
)


def load_mnist():
    """
    Load MNIST dataset.

    For this example, we'll generate synthetic MNIST-like data.
    In practice, you would load the actual MNIST dataset.
    """
    print("Generating synthetic MNIST-like data...")

    np.random.seed(42)

    # Generate synthetic data (28x28 images, 10 classes)
    n_train = 5000
    n_test = 1000
    img_size = 28 * 28
    n_classes = 10

    # Create training data
    X_train = np.random.randn(n_train, img_size).astype(np.float32)
    y_train_labels = np.random.randint(0, n_classes, n_train)
    y_train = np.eye(n_classes)[y_train_labels].astype(np.float32)

    # Create test data
    X_test = np.random.randn(n_test, img_size).astype(np.float32)
    y_test_labels = np.random.randint(0, n_classes, n_test)
    y_test = np.eye(n_classes)[y_test_labels].astype(np.float32)

    # Normalize
    X_train = (X_train - X_train.mean()) / (X_train.std() + 1e-7)
    X_test = (X_test - X_test.mean()) / (X_test.std() + 1e-7)

    print(f"  Training samples: {n_train}")
    print(f"  Test samples: {n_test}")
    print(f"  Input shape: ({img_size},)")
    print(f"  Classes: {n_classes}")
    print()

    return (X_train, y_train), (X_test, y_test)


def train_single_network():
    """Train a single dynamic network."""
    print("=" * 60)
    print("Training Single Dynamic Network")
    print("=" * 60)
    print()

    # Load data
    (X_train, y_train), (X_test, y_test) = load_mnist()

    # Create network
    network = DynamicNetwork(
        input_shape=(784,),
        output_size=10,
        seed=42,
        cost_function="CrossEntropy"
    )

    print(f"Network created:")
    print(f"  Cost function: CrossEntropy")
    print(f"  Seed: 42")
    print()

    # Train
    result = network.fit(X_train, y_train, verbose=True)

    # Evaluate on test set
    predictions = network.predict(X_test)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    accuracy = np.mean(pred_classes == true_classes)

    print()
    print("Test Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Parameters: {network.num_parameters:,}")
    print(f"  Layers: {network.num_layers}")

    # Check health
    health = network.health_status()
    print()
    print("Network Health:")
    print(f"  State: {health.state}")
    print(f"  Cancer Score: {health.cancer_score:.4f}")
    print(f"  Alzheimer Score: {health.alzheimer_score:.4f}")
    print(f"  Diagnosis: {health.diagnosis}")

    # Save model
    output_dir = "./mnist_model"
    os.makedirs(output_dir, exist_ok=True)
    network.save(output_dir, "mnist_classifier")
    print()
    print(f"Model saved to: {output_dir}")

    # Generate plots
    try:
        saved_files = auto_generate_plots(
            result,
            network=network,
            output_dir=output_dir,
            prefix="mnist",
            show=False
        )
        print(f"Plots saved: {saved_files}")
    except ImportError:
        print("Visualization libraries not available. Skipping plots.")

    return network, result


def train_three_models():
    """Generate three model variants."""
    print()
    print("=" * 60)
    print("Generating Three Model Variants")
    print("=" * 60)
    print()

    # Load data
    (X_train, y_train), (X_test, y_test) = load_mnist()

    # Combine for three-model generator (it handles splitting internally)
    X = np.vstack([X_train, X_test])
    y = np.vstack([y_train, y_test])

    # Generate three models
    output_dir = "./mnist_three_models"
    efficient, balanced, accurate = generate_models(
        X, y,
        input_shape=(784,),
        output_size=10,
        seed=42,
        cost_function="CrossEntropy",
        output_dir=output_dir,
        verbose=True
    )

    # Print comparison
    print()
    print("=" * 60)
    print("Model Comparison")
    print("=" * 60)
    print()
    print(f"{'Model':<20} {'Accuracy':>12} {'Efficiency':>12} {'Params':>12}")
    print("-" * 60)
    print(f"{'Most Efficient':<20} {efficient.test_accuracy:>12.4f} "
          f"{efficient.efficiency_score:>12.4f} {efficient.num_parameters:>12,}")
    print(f"{'Balanced':<20} {balanced.test_accuracy:>12.4f} "
          f"{balanced.efficiency_score:>12.4f} {balanced.num_parameters:>12,}")
    print(f"{'Highest Accuracy':<20} {accurate.test_accuracy:>12.4f} "
          f"{accurate.efficiency_score:>12.4f} {accurate.num_parameters:>12,}")
    print()
    print(f"Models saved to: {output_dir}")

    return efficient, balanced, accurate


def demonstrate_cost_functions():
    """Demonstrate different cost functions."""
    print()
    print("=" * 60)
    print("Available Cost Functions")
    print("=" * 60)
    print()

    for cf in COST_FUNCTIONS:
        print(f"  - {cf}")

    print()
    print("Example usage:")
    print('  network = DynamicNetwork(..., cost_function="CrossEntropy")')
    print('  network = DynamicNetwork(..., cost_function="MSE")')
    print()


def main():
    """Run all examples."""
    print()
    print("*" * 60)
    print("Dynamic Neural Network - MNIST Example")
    print("*" * 60)
    print()

    # Show available cost functions
    demonstrate_cost_functions()

    # Train single network
    network, result = train_single_network()

    # Generate three models
    efficient, balanced, accurate = train_three_models()

    print()
    print("*" * 60)
    print("All examples completed!")
    print("*" * 60)


if __name__ == "__main__":
    main()
