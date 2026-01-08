"""
Comparison Evaluator for Standard vs Dynamic Transformer

Provides side-by-side comparison metrics, visualizations, and analysis
for comparing the standard transformer with the Dynamic NN-enhanced version.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .standard_transformer import StandardTransformer
from .dynamic_transformer import DynamicTransformer
from .arabic_tokenizer import ArabicTokenizer


@dataclass
class ComparisonMetrics:
    """Side-by-side comparison metrics"""
    # Accuracy
    standard_accuracy: float
    dynamic_accuracy: float

    # Top-K accuracy
    standard_top3_accuracy: float
    dynamic_top3_accuracy: float
    standard_top5_accuracy: float
    dynamic_top5_accuracy: float

    # Loss and perplexity
    standard_loss: float
    dynamic_loss: float
    standard_perplexity: float
    dynamic_perplexity: float

    # Training time
    standard_time_seconds: float
    dynamic_time_seconds: float

    # Parameter efficiency
    standard_params: int
    dynamic_initial_params: int
    dynamic_final_params: int
    dynamic_params_added: int
    dynamic_params_removed: int

    # Architecture changes (dynamic only)
    dynamic_layers_added: int
    dynamic_layers_removed: int
    dynamic_heads_added: int
    dynamic_heads_removed: int
    dynamic_nodes_added: int
    dynamic_nodes_removed: int

    # Health scores (dynamic only)
    dynamic_final_cancer_score: float
    dynamic_final_alzheimer_score: float

    # Efficiency
    standard_params_per_accuracy: float
    dynamic_params_per_accuracy: float

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'accuracy': {
                'standard': self.standard_accuracy,
                'dynamic': self.dynamic_accuracy,
                'improvement': self.dynamic_accuracy - self.standard_accuracy
            },
            'top3_accuracy': {
                'standard': self.standard_top3_accuracy,
                'dynamic': self.dynamic_top3_accuracy
            },
            'top5_accuracy': {
                'standard': self.standard_top5_accuracy,
                'dynamic': self.dynamic_top5_accuracy
            },
            'perplexity': {
                'standard': self.standard_perplexity,
                'dynamic': self.dynamic_perplexity
            },
            'training_time_seconds': {
                'standard': self.standard_time_seconds,
                'dynamic': self.dynamic_time_seconds
            },
            'parameters': {
                'standard': self.standard_params,
                'dynamic_final': self.dynamic_final_params,
                'dynamic_change': self.dynamic_final_params - self.dynamic_initial_params
            },
            'architecture_changes': {
                'layers_added': self.dynamic_layers_added,
                'layers_removed': self.dynamic_layers_removed,
                'heads_added': self.dynamic_heads_added,
                'heads_removed': self.dynamic_heads_removed,
                'nodes_added': self.dynamic_nodes_added,
                'nodes_removed': self.dynamic_nodes_removed
            },
            'health': {
                'cancer_score': self.dynamic_final_cancer_score,
                'alzheimer_score': self.dynamic_final_alzheimer_score
            },
            'efficiency': {
                'standard_params_per_accuracy': self.standard_params_per_accuracy,
                'dynamic_params_per_accuracy': self.dynamic_params_per_accuracy
            }
        }


class TransformerComparisonEvaluator:
    """
    Evaluator for comparing Standard vs Dynamic Transformer.
    """

    def __init__(
        self,
        standard_model: StandardTransformer,
        dynamic_model: DynamicTransformer,
        tokenizer: Optional[ArabicTokenizer] = None
    ):
        self.standard = standard_model
        self.dynamic = dynamic_model
        self.tokenizer = tokenizer
        self.metrics: Optional[ComparisonMetrics] = None

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True
    ) -> ComparisonMetrics:
        """
        Run full comparison evaluation.

        Args:
            X_test: Test input token IDs
            y_test: Test labels
            verbose: Print results

        Returns:
            ComparisonMetrics with all comparison data
        """
        # Evaluate standard model
        std_eval = self.standard.evaluate(X_test, y_test)
        std_history = self.standard.training_history

        # Evaluate dynamic model
        dyn_eval = self.dynamic.evaluate(X_test, y_test)
        dyn_history = self.dynamic.training_history

        # Get architecture info
        std_arch = self.standard.get_architecture_summary()
        dyn_arch = self.dynamic.get_architecture_summary()

        # Health status
        health = self.dynamic.health_status()

        # Calculate initial params for dynamic (before training changes)
        initial_params = (
            self.dynamic.config.initial_num_layers *
            (self.dynamic.config.embed_dim * self.dynamic.config.embed_dim * 4 +  # attention
             self.dynamic.config.embed_dim * self.dynamic.config.initial_ffn_dim * 2)  # ffn
        )

        # Create metrics
        self.metrics = ComparisonMetrics(
            # Accuracy
            standard_accuracy=std_eval['accuracy'],
            dynamic_accuracy=dyn_eval['accuracy'],

            # Top-K
            standard_top3_accuracy=std_eval['top3_accuracy'],
            dynamic_top3_accuracy=dyn_eval['top3_accuracy'],
            standard_top5_accuracy=std_eval['top5_accuracy'],
            dynamic_top5_accuracy=dyn_eval['top5_accuracy'],

            # Loss and perplexity
            standard_loss=std_eval['loss'],
            dynamic_loss=dyn_eval['loss'],
            standard_perplexity=std_eval['perplexity'],
            dynamic_perplexity=dyn_eval['perplexity'],

            # Training time
            standard_time_seconds=std_history.get('training_time_seconds', 0),
            dynamic_time_seconds=dyn_history.get('training_time_seconds', 0),

            # Parameters
            standard_params=std_arch['total_parameters'],
            dynamic_initial_params=initial_params,
            dynamic_final_params=dyn_arch['total_parameters'],
            dynamic_params_added=dyn_history.get('nodes_added', 0),
            dynamic_params_removed=dyn_history.get('nodes_removed', 0),

            # Architecture changes
            dynamic_layers_added=dyn_history.get('layers_added', 0),
            dynamic_layers_removed=dyn_history.get('layers_removed', 0),
            dynamic_heads_added=dyn_history.get('heads_added', 0),
            dynamic_heads_removed=dyn_history.get('heads_removed', 0),
            dynamic_nodes_added=dyn_history.get('nodes_added', 0),
            dynamic_nodes_removed=dyn_history.get('nodes_removed', 0),

            # Health
            dynamic_final_cancer_score=health['cancer_score'],
            dynamic_final_alzheimer_score=health['alzheimer_score'],

            # Efficiency
            standard_params_per_accuracy=std_arch['total_parameters'] / (std_eval['accuracy'] + 1e-8),
            dynamic_params_per_accuracy=dyn_arch['total_parameters'] / (dyn_eval['accuracy'] + 1e-8),
        )

        if verbose:
            self._print_comparison()

        return self.metrics

    def _print_comparison(self) -> None:
        """Print comparison summary"""
        if self.metrics is None:
            return

        m = self.metrics
        print("\n" + "=" * 70)
        print("COMPARISON: Standard Transformer vs Dynamic Transformer")
        print("=" * 70)

        print("\n--- ACCURACY ---")
        print(f"{'Metric':<20} {'Standard':>15} {'Dynamic':>15} {'Diff':>15}")
        print("-" * 65)
        print(f"{'Accuracy':<20} {m.standard_accuracy:>14.2%} {m.dynamic_accuracy:>14.2%} {m.dynamic_accuracy - m.standard_accuracy:>+14.2%}")
        print(f"{'Top-3 Accuracy':<20} {m.standard_top3_accuracy:>14.2%} {m.dynamic_top3_accuracy:>14.2%} {m.dynamic_top3_accuracy - m.standard_top3_accuracy:>+14.2%}")
        print(f"{'Top-5 Accuracy':<20} {m.standard_top5_accuracy:>14.2%} {m.dynamic_top5_accuracy:>14.2%} {m.dynamic_top5_accuracy - m.standard_top5_accuracy:>+14.2%}")
        print(f"{'Perplexity':<20} {m.standard_perplexity:>15.2f} {m.dynamic_perplexity:>15.2f} {m.dynamic_perplexity - m.standard_perplexity:>+15.2f}")

        print("\n--- EFFICIENCY ---")
        print(f"{'Parameters':<20} {m.standard_params:>15,} {m.dynamic_final_params:>15,} {m.dynamic_final_params - m.standard_params:>+15,}")
        print(f"{'Training Time (s)':<20} {m.standard_time_seconds:>15.1f} {m.dynamic_time_seconds:>15.1f} {m.dynamic_time_seconds - m.standard_time_seconds:>+15.1f}")
        print(f"{'Params/Accuracy':<20} {m.standard_params_per_accuracy:>15,.0f} {m.dynamic_params_per_accuracy:>15,.0f}")

        print("\n--- DYNAMIC ARCHITECTURE CHANGES ---")
        print(f"Layers: +{m.dynamic_layers_added} / -{m.dynamic_layers_removed}")
        print(f"Attention Heads: +{m.dynamic_heads_added} / -{m.dynamic_heads_removed}")
        print(f"FFN Nodes: +{m.dynamic_nodes_added} / -{m.dynamic_nodes_removed}")

        print("\n--- DYNAMIC HEALTH STATUS ---")
        print(f"Cancer Score: {m.dynamic_final_cancer_score:.2%}")
        print(f"Alzheimer Score: {m.dynamic_final_alzheimer_score:.2%}")

        print("\n" + "=" * 70)

        # Winner summary
        acc_winner = "Dynamic" if m.dynamic_accuracy > m.standard_accuracy else "Standard"
        param_winner = "Dynamic" if m.dynamic_final_params < m.standard_params else "Standard"
        time_winner = "Dynamic" if m.dynamic_time_seconds < m.standard_time_seconds else "Standard"

        print(f"WINNERS: Accuracy={acc_winner}, Params={param_winner}, Time={time_winner}")
        print("=" * 70 + "\n")

    def plot_training_curves(
        self,
        save_path: Optional[str] = None,
        figsize: tuple = (14, 5)
    ) -> None:
        """
        Plot side-by-side training curves.
        """
        std_history = self.standard.training_history
        dyn_history = self.dynamic.training_history

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Loss curves
        ax1 = axes[0]
        ax1.plot(std_history.get('train_loss', []), label='Standard', color='blue', linewidth=2)
        ax1.plot(dyn_history.get('cost_history', []), label='Dynamic', color='red', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy curves
        ax2 = axes[1]
        ax2.plot(std_history.get('train_acc', []), label='Standard Train', color='blue', linewidth=2)
        ax2.plot(dyn_history.get('train_acc_history', []), label='Dynamic Train', color='red', linewidth=2)
        if std_history.get('val_acc'):
            ax2.plot(std_history.get('val_acc', []), label='Standard Val', color='blue', linestyle='--', alpha=0.7)
        if dyn_history.get('val_acc_history'):
            ax2.plot(dyn_history.get('val_acc_history', []), label='Dynamic Val', color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Final metrics bar chart
        ax3 = axes[2]
        metrics = ['Accuracy', 'Top-3 Acc', 'Top-5 Acc']
        std_values = [
            self.metrics.standard_accuracy if self.metrics else 0,
            self.metrics.standard_top3_accuracy if self.metrics else 0,
            self.metrics.standard_top5_accuracy if self.metrics else 0
        ]
        dyn_values = [
            self.metrics.dynamic_accuracy if self.metrics else 0,
            self.metrics.dynamic_top3_accuracy if self.metrics else 0,
            self.metrics.dynamic_top5_accuracy if self.metrics else 0
        ]

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax3.bar(x - width/2, std_values, width, label='Standard', color='blue', alpha=0.7)
        bars2 = ax3.bar(x + width/2, dyn_values, width, label='Dynamic', color='red', alpha=0.7)

        ax3.set_ylabel('Score')
        ax3.set_title('Final Performance Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax3.annotate(f'{height:.1%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax3.annotate(f'{height:.1%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    def plot_architecture_evolution(
        self,
        save_path: Optional[str] = None,
        figsize: tuple = (14, 4)
    ) -> None:
        """
        Plot architecture changes over time (Dynamic model only).
        """
        arch_history = self.dynamic.training_history.get('architecture_history', [])

        if not arch_history:
            print("No architecture history available")
            return

        epochs = range(len(arch_history))
        layers = [h['num_layers'] for h in arch_history]
        heads = [h['total_heads'] for h in arch_history]
        ffn_nodes = [h['total_ffn_nodes'] for h in arch_history]
        params = [h.get('params', 0) for h in arch_history]

        fig, axes = plt.subplots(1, 4, figsize=figsize)

        # Number of layers
        ax1 = axes[0]
        ax1.plot(epochs, layers, color='green', linewidth=2)
        ax1.axhline(y=self.dynamic.config.initial_num_layers, color='gray', linestyle='--', alpha=0.5, label='Initial')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Count')
        ax1.set_title('Number of Layers')
        ax1.grid(True, alpha=0.3)

        # Total attention heads
        ax2 = axes[1]
        ax2.plot(epochs, heads, color='purple', linewidth=2)
        initial_heads = self.dynamic.config.initial_num_layers * self.dynamic.config.initial_num_heads
        ax2.axhline(y=initial_heads, color='gray', linestyle='--', alpha=0.5, label='Initial')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Count')
        ax2.set_title('Total Attention Heads')
        ax2.grid(True, alpha=0.3)

        # Total FFN nodes
        ax3 = axes[2]
        ax3.plot(epochs, ffn_nodes, color='orange', linewidth=2)
        initial_ffn = self.dynamic.config.initial_num_layers * self.dynamic.config.initial_ffn_dim
        ax3.axhline(y=initial_ffn, color='gray', linestyle='--', alpha=0.5, label='Initial')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Count')
        ax3.set_title('Total FFN Nodes')
        ax3.grid(True, alpha=0.3)

        # Parameters
        ax4 = axes[3]
        ax4.plot(epochs, params, color='blue', linewidth=2)
        ax4.axhline(y=self.standard.count_parameters(), color='red', linestyle='--', alpha=0.7, label='Standard')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Parameters')
        ax4.set_title('Total Parameters')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Add phase markers
        for ax in axes:
            ax.axvline(x=10, color='gray', linestyle=':', alpha=0.5)  # End Phase 1
            ax.axvline(x=20, color='gray', linestyle=':', alpha=0.5)  # End Phase 2

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    def plot_health_scores(
        self,
        save_path: Optional[str] = None,
        figsize: tuple = (10, 4)
    ) -> None:
        """
        Plot health monitoring scores (Dynamic model only).
        """
        cancer_history = self.dynamic.training_history.get('cancer_score_history', [])
        alzheimer_history = self.dynamic.training_history.get('alzheimer_score_history', [])

        if not cancer_history:
            print("No health history available")
            return

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        epochs = range(len(cancer_history))

        # Cancer score
        ax1 = axes[0]
        ax1.plot(epochs, cancer_history, color='red', linewidth=2, label='Cancer Score')
        ax1.axhline(y=0.3, color='yellow', linestyle='--', alpha=0.7, label='Warning')
        ax1.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Critical')
        ax1.fill_between(epochs, cancer_history, alpha=0.3, color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Score')
        ax1.set_title('Cancer Score (Overgrowth)')
        ax1.legend()
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        # Alzheimer score
        ax2 = axes[1]
        ax2.plot(epochs, alzheimer_history, color='blue', linewidth=2, label='Alzheimer Score')
        ax2.axhline(y=0.3, color='yellow', linestyle='--', alpha=0.7, label='Warning')
        ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Critical')
        ax2.fill_between(epochs, alzheimer_history, alpha=0.3, color='blue')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.set_title('Alzheimer Score (Over-pruning)')
        ax2.legend()
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        # Add phase markers
        for ax in axes:
            ax.axvline(x=10, color='gray', linestyle=':', alpha=0.5)
            ax.axvline(x=20, color='gray', linestyle=':', alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    def plot_parameter_efficiency(
        self,
        save_path: Optional[str] = None,
        figsize: tuple = (10, 5)
    ) -> None:
        """
        Compare parameter counts and efficiency.
        """
        if self.metrics is None:
            print("Run evaluate() first")
            return

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Parameter comparison
        ax1 = axes[0]
        models = ['Standard', 'Dynamic\n(Initial)', 'Dynamic\n(Final)']
        params = [
            self.metrics.standard_params,
            self.metrics.dynamic_initial_params,
            self.metrics.dynamic_final_params
        ]
        colors = ['blue', 'lightcoral', 'red']
        bars = ax1.bar(models, params, color=colors, alpha=0.7)
        ax1.set_ylabel('Parameters')
        ax1.set_title('Parameter Count Comparison')
        ax1.grid(True, alpha=0.3, axis='y')

        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:,.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

        # Efficiency comparison (params per accuracy point)
        ax2 = axes[1]
        models = ['Standard', 'Dynamic']
        efficiency = [
            self.metrics.standard_params_per_accuracy,
            self.metrics.dynamic_params_per_accuracy
        ]
        colors = ['blue', 'red']
        bars = ax2.bar(models, efficiency, color=colors, alpha=0.7)
        ax2.set_ylabel('Parameters per Accuracy Point')
        ax2.set_title('Parameter Efficiency\n(Lower is Better)')
        ax2.grid(True, alpha=0.3, axis='y')

        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:,.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    def show_prediction_examples(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_examples: int = 10,
        show_incorrect: bool = True
    ) -> None:
        """
        Show side-by-side prediction examples.
        """
        # Get predictions
        std_preds = self.standard.predict(X_test)
        dyn_preds = self.dynamic.predict(X_test)

        std_probs = self.standard.predict_proba(X_test)
        dyn_probs = self.dynamic.predict_proba(X_test)

        print("\n" + "=" * 80)
        print("PREDICTION EXAMPLES")
        print("=" * 80)

        # Find correct/incorrect predictions
        std_correct = std_preds == y_test
        dyn_correct = dyn_preds == y_test

        if show_incorrect:
            # Show cases where models differ
            differ_mask = std_preds != dyn_preds
            indices = np.where(differ_mask)[0][:n_examples]
            print(f"\nCases where models DIFFER ({len(indices)} shown):")
        else:
            indices = np.random.choice(len(y_test), min(n_examples, len(y_test)), replace=False)
            print(f"\nRandom examples ({len(indices)} shown):")

        print("-" * 80)

        for i, idx in enumerate(indices):
            true_label = y_test[idx]
            std_pred = std_preds[idx]
            dyn_pred = dyn_preds[idx]
            std_conf = std_probs[idx, std_pred]
            dyn_conf = dyn_probs[idx, dyn_pred]

            std_status = "CORRECT" if std_correct[idx] else "WRONG"
            dyn_status = "CORRECT" if dyn_correct[idx] else "WRONG"

            # Decode if tokenizer available
            if self.tokenizer and hasattr(self.tokenizer, 'decode_input'):
                input_text = self.tokenizer.decode_input(X_test[idx])
                true_text = self.tokenizer.decode_output_label(true_label)
                std_text = self.tokenizer.decode_output_label(std_pred)
                dyn_text = self.tokenizer.decode_output_label(dyn_pred)
            else:
                input_text = f"[Token IDs: {X_test[idx][:5]}...]"
                true_text = f"Class {true_label}"
                std_text = f"Class {std_pred}"
                dyn_text = f"Class {dyn_pred}"

            print(f"\nExample {i+1}:")
            print(f"  Input: {input_text[:60]}...")
            print(f"  True:  {true_text}")
            print(f"  Standard: {std_text} ({std_conf:.2%}) - {std_status}")
            print(f"  Dynamic:  {dyn_text} ({dyn_conf:.2%}) - {dyn_status}")

        print("\n" + "=" * 80)

        # Summary
        std_acc = np.mean(std_correct)
        dyn_acc = np.mean(dyn_correct)
        both_correct = np.mean(std_correct & dyn_correct)
        both_wrong = np.mean(~std_correct & ~dyn_correct)
        std_only_correct = np.mean(std_correct & ~dyn_correct)
        dyn_only_correct = np.mean(~std_correct & dyn_correct)

        print("\nPREDICTION AGREEMENT ANALYSIS:")
        print(f"  Both correct:        {both_correct:.2%}")
        print(f"  Both wrong:          {both_wrong:.2%}")
        print(f"  Standard only right: {std_only_correct:.2%}")
        print(f"  Dynamic only right:  {dyn_only_correct:.2%}")
        print("=" * 80 + "\n")

    def generate_comparison_report(
        self,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate markdown comparison report.
        """
        if self.metrics is None:
            return "Run evaluate() first"

        m = self.metrics

        report = f"""# Transformer Comparison Report

## Executive Summary

| Metric | Standard | Dynamic | Winner |
|--------|----------|---------|--------|
| Accuracy | {m.standard_accuracy:.2%} | {m.dynamic_accuracy:.2%} | {'Dynamic' if m.dynamic_accuracy > m.standard_accuracy else 'Standard'} |
| Top-3 Accuracy | {m.standard_top3_accuracy:.2%} | {m.dynamic_top3_accuracy:.2%} | {'Dynamic' if m.dynamic_top3_accuracy > m.standard_top3_accuracy else 'Standard'} |
| Perplexity | {m.standard_perplexity:.2f} | {m.dynamic_perplexity:.2f} | {'Dynamic' if m.dynamic_perplexity < m.standard_perplexity else 'Standard'} |
| Parameters | {m.standard_params:,} | {m.dynamic_final_params:,} | {'Dynamic' if m.dynamic_final_params < m.standard_params else 'Standard'} |
| Training Time | {m.standard_time_seconds:.1f}s | {m.dynamic_time_seconds:.1f}s | {'Dynamic' if m.dynamic_time_seconds < m.standard_time_seconds else 'Standard'} |

## Performance Analysis

### Accuracy Improvement
- Absolute improvement: {(m.dynamic_accuracy - m.standard_accuracy):.2%}
- Relative improvement: {((m.dynamic_accuracy - m.standard_accuracy) / (m.standard_accuracy + 1e-8) * 100):.1f}%

### Parameter Efficiency
- Standard: {m.standard_params_per_accuracy:,.0f} params per accuracy point
- Dynamic: {m.dynamic_params_per_accuracy:,.0f} params per accuracy point
- Efficiency gain: {((m.standard_params_per_accuracy - m.dynamic_params_per_accuracy) / m.standard_params_per_accuracy * 100):.1f}%

## Dynamic Architecture Changes

| Change Type | Added | Removed | Net |
|-------------|-------|---------|-----|
| Layers | {m.dynamic_layers_added} | {m.dynamic_layers_removed} | {m.dynamic_layers_added - m.dynamic_layers_removed} |
| Attention Heads | {m.dynamic_heads_added} | {m.dynamic_heads_removed} | {m.dynamic_heads_added - m.dynamic_heads_removed} |
| FFN Nodes | {m.dynamic_nodes_added} | {m.dynamic_nodes_removed} | {m.dynamic_nodes_added - m.dynamic_nodes_removed} |

## Health Monitoring

- **Cancer Score** (overgrowth): {m.dynamic_final_cancer_score:.2%}
- **Alzheimer Score** (over-pruning): {m.dynamic_final_alzheimer_score:.2%}
- **Overall Health**: {(1 - max(m.dynamic_final_cancer_score, m.dynamic_final_alzheimer_score)):.2%}

## Conclusions

{'The Dynamic Transformer achieved HIGHER accuracy with adaptive architecture.' if m.dynamic_accuracy > m.standard_accuracy else 'The Standard Transformer performed better on this task.'}

{'The Dynamic model is MORE parameter-efficient.' if m.dynamic_params_per_accuracy < m.standard_params_per_accuracy else 'The Standard model is more parameter-efficient.'}

Health scores indicate {'healthy' if max(m.dynamic_final_cancer_score, m.dynamic_final_alzheimer_score) < 0.3 else 'moderate' if max(m.dynamic_final_cancer_score, m.dynamic_final_alzheimer_score) < 0.7 else 'concerning'} architecture adaptation behavior.
"""

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)

        return report
