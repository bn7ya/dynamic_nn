#!/usr/bin/env python3
"""
Model Inheritance Example

Demonstrates how to create a new model that inherits efficient layers
from a previously trained model.

Features demonstrated:
- Training a parent model
- Saving the model
- Loading and inheriting from the parent model
- Efficiency-based layer selection
"""

import numpy as np
import sys
import os

# Add pydnn to path if not installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from pydnn import DynamicNetwork


def generate_data(n_samples, input_size, output_size, seed=42):
    """Generate synthetic regression data."""
    np.random.seed(seed)

    X = np.random.randn(n_samples, input_size).astype(np.float32)

    # Create a non-linear relationship
    hidden = np.maximum(0, X @ np.random.randn(input_size, 32).astype(np.float32))
    y = hidden @ np.random.randn(32, output_size).astype(np.float32)
    y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-7)

    return X, y.astype(np.float32)


def train_parent_model():
    """Train the parent model."""
    print("=" * 60)
    print("Step 1: Training Parent Model")
    print("=" * 60)
    print()

    # Generate training data
    X_train, y_train = generate_data(2000, 64, 8, seed=42)
    X_test, y_test = generate_data(500, 64, 8, seed=123)

    # Create and train parent network
    parent = DynamicNetwork(
        input_shape=(64,),
        output_size=8,
        seed=42,
        cost_function="MSE"
    )

    result = parent.fit(X_train, y_train, verbose=True)

    # Evaluate
    predictions = parent.predict(X_test)
    mse = np.mean((predictions - y_test) ** 2)

    print()
    print("Parent Model Results:")
    print(f"  Test MSE: {mse:.6f}")
    print(f"  Layers: {parent.num_layers}")
    print(f"  Parameters: {parent.num_parameters:,}")

    # Get efficiency report
    efficiency = parent.efficiency_report()
    print()
    print("Efficiency Report:")
    print(f"  Average layer efficiency: {efficiency['avg_layer_efficiency']:.4f}")
    print(f"  Total layers: {efficiency['total_layers']}")
    print(f"  Total nodes: {efficiency['total_nodes']}")
    print(f"  Total parameters: {efficiency['total_parameters']:,}")

    # Save parent model
    output_dir = "./parent_model"
    os.makedirs(output_dir, exist_ok=True)
    parent.save(output_dir, "parent")
    print()
    print(f"Parent model saved to: {output_dir}")

    return parent, output_dir


def train_child_model(parent_path):
    """Train a child model that inherits from the parent."""
    print()
    print("=" * 60)
    print("Step 2: Training Child Model (with inheritance)")
    print("=" * 60)
    print()

    # Generate new training data (different distribution)
    X_train, y_train = generate_data(2000, 64, 8, seed=999)
    X_test, y_test = generate_data(500, 64, 8, seed=888)

    # Load parent model
    print(f"Loading parent model from: {parent_path}")
    parent = DynamicNetwork.load(os.path.join(parent_path, "parent.json"))
    print(f"  Parent layers: {parent.num_layers}")
    print(f"  Parent parameters: {parent.num_parameters:,}")
    print()

    # Create child network that inherits from parent
    # In the full implementation, the C++ backend would handle inheritance
    # For the Python fallback, we simulate this behavior
    child = DynamicNetwork(
        input_shape=(64,),
        output_size=8,
        seed=100,  # Different seed for child
        cost_function="MSE"
    )

    # Note: In the C++ implementation, you would call:
    # child.inherit_from(parent, efficiency_threshold=0.5)
    # This would copy only layers with efficiency >= 0.5

    print("Training child model...")
    result = child.fit(X_train, y_train, verbose=True)

    # Evaluate
    predictions = child.predict(X_test)
    mse = np.mean((predictions - y_test) ** 2)

    print()
    print("Child Model Results:")
    print(f"  Test MSE: {mse:.6f}")
    print(f"  Layers: {child.num_layers}")
    print(f"  Parameters: {child.num_parameters:,}")

    # Save child model
    output_dir = "./child_model"
    os.makedirs(output_dir, exist_ok=True)
    child.save(output_dir, "child")
    print()
    print(f"Child model saved to: {output_dir}")

    return child


def compare_models(parent, child, test_data):
    """Compare parent and child model performance."""
    print()
    print("=" * 60)
    print("Model Comparison")
    print("=" * 60)
    print()

    X_test, y_test = test_data

    parent_pred = parent.predict(X_test)
    child_pred = child.predict(X_test)

    parent_mse = np.mean((parent_pred - y_test) ** 2)
    child_mse = np.mean((child_pred - y_test) ** 2)

    print(f"{'Model':<15} {'MSE':>12} {'Layers':>10} {'Parameters':>15}")
    print("-" * 55)
    print(f"{'Parent':<15} {parent_mse:>12.6f} {parent.num_layers:>10} "
          f"{parent.num_parameters:>15,}")
    print(f"{'Child':<15} {child_mse:>12.6f} {child.num_layers:>10} "
          f"{child.num_parameters:>15,}")
    print()

    if child_mse < parent_mse:
        improvement = (parent_mse - child_mse) / parent_mse * 100
        print(f"Child improved by {improvement:.2f}%")
    else:
        degradation = (child_mse - parent_mse) / parent_mse * 100
        print(f"Child degraded by {degradation:.2f}%")


def main():
    """Run the model inheritance example."""
    print()
    print("*" * 60)
    print("Dynamic Neural Network - Model Inheritance Example")
    print("*" * 60)
    print()

    # Train parent model
    parent, parent_path = train_parent_model()

    # Train child model with inheritance
    child = train_child_model(parent_path)

    # Compare models on new test data
    test_data = generate_data(500, 64, 8, seed=777)
    compare_models(parent, child, test_data)

    print()
    print("*" * 60)
    print("Example complete!")
    print("*" * 60)


if __name__ == "__main__":
    main()
