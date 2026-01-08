"""
Comparison Evaluator for Standard NN vs Dynamic NN

Provides comprehensive comparison metrics, visualizations, and analysis
for emotional classification models.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from .emotional_data_generator import Emotion


@dataclass
class ComparisonMetrics:
    """Comparison metrics between Standard and Dynamic NN."""
    # Standard NN metrics
    standard_accuracy: float
    standard_top3_accuracy: float
    standard_loss: float
    standard_parameters: int
    standard_training_time: float
    standard_class_accuracy: Dict[int, float]

    # Dynamic NN metrics
    dynamic_accuracy: float
    dynamic_top3_accuracy: float
    dynamic_loss: float
    dynamic_parameters: int
    dynamic_training_time: float
    dynamic_class_accuracy: Dict[int, float]

    # Health metrics (Dynamic only)
    cancer_score: float = 0.0
    alzheimer_score: float = 0.0
    depression_ratio: float = 0.0
    excitement_ratio: float = 0.0
    emotional_state: str = "neutral"

    # Differences
    @property
    def accuracy_diff(self) -> float:
        return self.dynamic_accuracy - self.standard_accuracy

    @property
    def parameter_diff(self) -> int:
        return self.dynamic_parameters - self.standard_parameters

    @property
    def time_diff(self) -> float:
        return self.dynamic_training_time - self.standard_training_time


class EmotionalClassifierComparison:
    """
    Comprehensive comparison between Standard and Dynamic Neural Networks
    for emotional classification.
    """

    EMOTION_NAMES = [e.value for e in Emotion]
    EMOTION_COLORS = {
        'happy': '#2ecc71',
        'sad': '#3498db',
        'angry': '#e74c3c',
        'fearful': '#9b59b6',
        'surprised': '#f39c12',
        'disgusted': '#1abc9c',
        'neutral': '#95a5a6'
    }

    def __init__(self, standard_model, dynamic_model, vectorizer):
        """
        Initialize comparison evaluator.

        Args:
            standard_model: Trained StandardNeuralNetwork
            dynamic_model: Trained DynamicNeuralNetwork
            vectorizer: Fitted TextVectorizer
        """
        self.standard = standard_model
        self.dynamic = dynamic_model
        self.vectorizer = vectorizer
        self.metrics: Optional[ComparisonMetrics] = None

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 verbose: bool = True) -> ComparisonMetrics:
        """
        Evaluate both models and compute comparison metrics.

        Args:
            X_test: Test features
            y_test: Test labels
            verbose: Print results

        Returns:
            ComparisonMetrics object
        """
        # Evaluate Standard NN
        std_eval = self.standard.evaluate(X_test, y_test)
        std_history = self.standard.training_history

        # Evaluate Dynamic NN
        dyn_eval = self.dynamic.evaluate(X_test, y_test)
        dyn_history = self.dynamic.training_history
        dyn_health = self.dynamic.health_status()

        self.metrics = ComparisonMetrics(
            # Standard
            standard_accuracy=std_eval['accuracy'],
            standard_top3_accuracy=std_eval['top3_accuracy'],
            standard_loss=std_eval['loss'],
            standard_parameters=self.standard.count_parameters(),
            standard_training_time=std_history.training_time if std_history else 0,
            standard_class_accuracy=std_eval['class_accuracy'],
            # Dynamic
            dynamic_accuracy=dyn_eval['accuracy'],
            dynamic_top3_accuracy=dyn_eval['top3_accuracy'],
            dynamic_loss=dyn_eval['loss'],
            dynamic_parameters=self.dynamic.count_parameters(),
            dynamic_training_time=dyn_history.training_time if dyn_history else 0,
            dynamic_class_accuracy=dyn_eval['class_accuracy'],
            # Health
            cancer_score=dyn_health['cancer_score'],
            alzheimer_score=dyn_health['alzheimer_score'],
            depression_ratio=dyn_health['depression_ratio'],
            excitement_ratio=dyn_health['excitement_ratio'],
            emotional_state=dyn_health['emotional_state']
        )

        if verbose:
            self._print_comparison()

        return self.metrics

    def _print_comparison(self) -> None:
        """Print comparison results."""
        m = self.metrics

        print("\n" + "=" * 70)
        print("EMOTIONAL CLASSIFIER COMPARISON: Standard NN vs Dynamic NN")
        print("=" * 70)

        print("\n--- ACCURACY ---")
        print(f"{'Model':<15} {'Accuracy':<12} {'Top-3':<12} {'Loss':<12}")
        print("-" * 51)
        print(f"{'Standard NN':<15} {m.standard_accuracy:>10.2%} {m.standard_top3_accuracy:>10.2%} {m.standard_loss:>10.4f}")
        print(f"{'Dynamic NN':<15} {m.dynamic_accuracy:>10.2%} {m.dynamic_top3_accuracy:>10.2%} {m.dynamic_loss:>10.4f}")
        print(f"{'Difference':<15} {m.accuracy_diff:>+10.2%}")

        print("\n--- PARAMETERS ---")
        print(f"Standard NN: {m.standard_parameters:,} parameters")
        print(f"Dynamic NN:  {m.dynamic_parameters:,} parameters")
        print(f"Difference:  {m.parameter_diff:+,} ({m.parameter_diff/m.standard_parameters:+.1%})")

        print("\n--- TRAINING TIME ---")
        print(f"Standard NN: {m.standard_training_time:.2f}s")
        print(f"Dynamic NN:  {m.dynamic_training_time:.2f}s")
        print(f"Difference:  {m.time_diff:+.2f}s")

        print("\n--- DYNAMIC NN HEALTH ---")
        print(f"Cancer Score:     {m.cancer_score:.2%}")
        print(f"Alzheimer Score:  {m.alzheimer_score:.2%}")
        print(f"Depression Ratio: {m.depression_ratio:.2%}")
        print(f"Excitement Ratio: {m.excitement_ratio:.2%}")
        print(f"Emotional State:  {m.emotional_state}")

        print("\n--- PER-CLASS ACCURACY ---")
        print(f"{'Emotion':<12} {'Standard':<12} {'Dynamic':<12} {'Winner':<12}")
        print("-" * 48)
        for i, emotion in enumerate(self.EMOTION_NAMES):
            std_acc = m.standard_class_accuracy.get(i, 0)
            dyn_acc = m.dynamic_class_accuracy.get(i, 0)
            winner = "Dynamic" if dyn_acc > std_acc else "Standard" if std_acc > dyn_acc else "Tie"
            print(f"{emotion:<12} {std_acc:>10.2%} {dyn_acc:>10.2%} {winner:<12}")

        print("\n" + "=" * 70)

        # Overall winner
        if m.accuracy_diff > 0.02:
            print("WINNER: Dynamic NN (significantly better accuracy)")
        elif m.accuracy_diff < -0.02:
            print("WINNER: Standard NN (significantly better accuracy)")
        elif m.parameter_diff < 0:
            print("WINNER: Dynamic NN (similar accuracy, fewer parameters)")
        elif m.parameter_diff > 0:
            print("WINNER: Standard NN (similar accuracy, fewer parameters)")
        else:
            print("RESULT: Tie (similar performance)")

        print("=" * 70)

    def plot_training_curves(self, figsize: tuple = (14, 10)) -> None:
        """Plot training curves for both models."""
        std_history = self.standard.training_history
        dyn_history = self.dynamic.training_history

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Cost curves
        ax1 = axes[0, 0]
        ax1.plot(std_history.cost_history, 'b-', label='Standard NN', linewidth=2)
        ax1.plot(dyn_history.cost_history, 'r-', label='Dynamic NN', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Cost')
        ax1.set_title('Training Cost')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy curves
        ax2 = axes[0, 1]
        ax2.plot(std_history.accuracy_history, 'b-', label='Standard NN', linewidth=2)
        ax2.plot(dyn_history.accuracy_history, 'r-', label='Dynamic NN', linewidth=2)
        if std_history.val_accuracy_history:
            ax2.plot(std_history.val_accuracy_history, 'b--', label='Standard Val', linewidth=1, alpha=0.7)
        if dyn_history.val_accuracy_history:
            ax2.plot(dyn_history.val_accuracy_history, 'r--', label='Dynamic Val', linewidth=1, alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Learning rate comparison
        ax3 = axes[1, 0]
        ax3.plot(std_history.learning_rate_history, 'b-', label='Standard NN (decay)', linewidth=2)
        ax3.plot(dyn_history.learning_rate_history, 'r-', label='Dynamic NN (R/P)', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Adaptation')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Efficiency (Dynamic only)
        ax4 = axes[1, 1]
        if hasattr(dyn_history, 'efficiency_history') and dyn_history.efficiency_history:
            ax4.plot(dyn_history.efficiency_history, 'r-', label='Dynamic NN', linewidth=2)
            ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Efficiency')
            ax4.set_title('Training Efficiency (Dynamic NN)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Efficiency data not available',
                     ha='center', va='center', transform=ax4.transAxes)

        plt.suptitle('Training Curves Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_emotional_state(self, figsize: tuple = (14, 5)) -> None:
        """Plot emotional state metrics for Dynamic NN."""
        dyn_history = self.dynamic.training_history

        if not hasattr(dyn_history, 'depression_history') or not dyn_history.depression_history:
            print("Emotional state data not available")
            return

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        epochs = range(len(dyn_history.depression_history))

        # Depression/Excitement ratios
        ax1 = axes[0]
        ax1.plot(epochs, dyn_history.depression_history, 'b-', label='Depression Ratio', linewidth=2)
        ax1.plot(epochs, dyn_history.excitement_history, 'g-', label='Excitement Ratio', linewidth=2)
        ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Extreme Threshold')
        ax1.fill_between(epochs, dyn_history.depression_history, alpha=0.2, color='blue')
        ax1.fill_between(epochs, dyn_history.excitement_history, alpha=0.2, color='green')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Ratio')
        ax1.set_title('Emotional State Over Time')
        ax1.set_ylim(0, 1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Reward/Penalty actions pie chart
        ax2 = axes[1]
        actions = dyn_history.emotional_state_actions
        action_counts = {
            'Reward': sum(1 for a in actions if a == 'reward'),
            'Penalty': sum(1 for a in actions if a == 'penalty'),
            'Neutral': sum(1 for a in actions if a == 'neutral'),
            'Reset': sum(1 for a in actions if 'reset' in a)
        }

        colors = ['#2ecc71', '#e74c3c', '#95a5a6', '#f39c12']
        non_zero = [(k, v) for k, v in action_counts.items() if v > 0]
        if non_zero:
            labels, sizes = zip(*non_zero)
            ax2.pie(sizes, labels=labels, autopct='%1.1f%%',
                    colors=colors[:len(non_zero)], startangle=90)
            ax2.set_title(f'Reward/Penalty Distribution\n({dyn_history.total_rewards} R / {dyn_history.total_penalties} P)')
        else:
            ax2.text(0.5, 0.5, 'No R/P actions recorded',
                     ha='center', va='center', transform=ax2.transAxes)

        plt.suptitle('Dynamic NN Emotional State Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_per_class_accuracy(self, figsize: tuple = (12, 6)) -> None:
        """Plot per-class accuracy comparison."""
        if not self.metrics:
            print("Run evaluate() first")
            return

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(self.EMOTION_NAMES))
        width = 0.35

        std_accs = [self.metrics.standard_class_accuracy.get(i, 0) for i in range(len(self.EMOTION_NAMES))]
        dyn_accs = [self.metrics.dynamic_class_accuracy.get(i, 0) for i in range(len(self.EMOTION_NAMES))]

        bars1 = ax.bar(x - width/2, std_accs, width, label='Standard NN', color='#3498db')
        bars2 = ax.bar(x + width/2, dyn_accs, width, label='Dynamic NN', color='#e74c3c')

        ax.set_xlabel('Emotion')
        ax.set_ylabel('Accuracy')
        ax.set_title('Per-Class Accuracy Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([e.capitalize() for e in self.EMOTION_NAMES], rotation=45)
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.0%}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=8)

        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.0%}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.show()

    def plot_health_scores(self, figsize: tuple = (12, 5)) -> None:
        """Plot health scores for Dynamic NN."""
        dyn_history = self.dynamic.training_history

        if not dyn_history.cancer_score_history:
            print("Health score data not available")
            return

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        epochs = range(len(dyn_history.cancer_score_history))

        # Health scores over time
        ax1 = axes[0]
        ax1.plot(epochs, dyn_history.cancer_score_history, 'r-', label='Cancer Score', linewidth=2)
        ax1.plot(epochs, dyn_history.alzheimer_score_history, 'b-', label='Alzheimer Score', linewidth=2)
        ax1.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Warning Threshold')
        ax1.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Critical Threshold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Score')
        ax1.set_title('Health Scores Over Time')
        ax1.set_ylim(0, 1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Architecture evolution
        ax2 = axes[1]
        if dyn_history.architecture_history:
            layers = [a[0] for a in dyn_history.architecture_history]
            nodes = [a[1] for a in dyn_history.architecture_history]
            arch_epochs = range(len(layers))

            ax2_twin = ax2.twinx()
            ax2.plot(arch_epochs, layers, 'b-', label='Layers', linewidth=2)
            ax2_twin.plot(arch_epochs, nodes, 'g-', label='Nodes', linewidth=2)

            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Layers', color='b')
            ax2_twin.set_ylabel('Nodes', color='g')
            ax2.set_title('Architecture Evolution')
            ax2.tick_params(axis='y', labelcolor='b')
            ax2_twin.tick_params(axis='y', labelcolor='g')

            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax2.text(0.5, 0.5, 'Architecture history not available',
                     ha='center', va='center', transform=ax2.transAxes)

        plt.suptitle('Dynamic NN Health Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_confusion_comparison(self, X_test: np.ndarray, y_test: np.ndarray,
                                   figsize: tuple = (14, 6)) -> None:
        """Plot confusion matrices for both models."""
        std_pred = self.standard.predict(X_test)
        dyn_pred = self.dynamic.predict(X_test)

        n_classes = len(self.EMOTION_NAMES)

        # Compute confusion matrices
        std_cm = np.zeros((n_classes, n_classes), dtype=int)
        dyn_cm = np.zeros((n_classes, n_classes), dtype=int)

        for true, pred_s, pred_d in zip(y_test, std_pred, dyn_pred):
            std_cm[true, pred_s] += 1
            dyn_cm[true, pred_d] += 1

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        for ax, cm, title in [(axes[0], std_cm, 'Standard NN'),
                               (axes[1], dyn_cm, 'Dynamic NN')]:
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.set_title(f'{title} Confusion Matrix')

            # Show colorbar
            plt.colorbar(im, ax=ax)

            # Add labels
            tick_marks = np.arange(n_classes)
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels([e[:4] for e in self.EMOTION_NAMES], rotation=45)
            ax.set_yticklabels([e[:4] for e in self.EMOTION_NAMES])

            # Add text annotations
            thresh = cm.max() / 2
            for i in range(n_classes):
                for j in range(n_classes):
                    ax.text(j, i, format(cm[i, j], 'd'),
                           ha='center', va='center',
                           color='white' if cm[i, j] > thresh else 'black',
                           fontsize=8)

            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')

        plt.suptitle('Confusion Matrix Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def generate_report(self) -> str:
        """Generate a text report of the comparison."""
        if not self.metrics:
            return "Run evaluate() first"

        m = self.metrics

        report = []
        report.append("=" * 70)
        report.append("EMOTIONAL CLASSIFIER COMPARISON REPORT")
        report.append("Standard Neural Network vs Dynamic Neural Network")
        report.append("=" * 70)

        report.append("\n## Summary")
        report.append(f"- Standard NN Accuracy: {m.standard_accuracy:.2%}")
        report.append(f"- Dynamic NN Accuracy: {m.dynamic_accuracy:.2%}")
        report.append(f"- Accuracy Difference: {m.accuracy_diff:+.2%}")

        report.append("\n## Parameters")
        report.append(f"- Standard NN: {m.standard_parameters:,}")
        report.append(f"- Dynamic NN: {m.dynamic_parameters:,}")
        report.append(f"- Difference: {m.parameter_diff:+,} ({m.parameter_diff/m.standard_parameters:+.1%})")

        report.append("\n## Training Time")
        report.append(f"- Standard NN: {m.standard_training_time:.2f}s")
        report.append(f"- Dynamic NN: {m.dynamic_training_time:.2f}s")

        report.append("\n## Dynamic NN Health Status")
        report.append(f"- Cancer Score: {m.cancer_score:.2%}")
        report.append(f"- Alzheimer Score: {m.alzheimer_score:.2%}")
        report.append(f"- Depression Ratio: {m.depression_ratio:.2%}")
        report.append(f"- Excitement Ratio: {m.excitement_ratio:.2%}")
        report.append(f"- Emotional State: {m.emotional_state}")

        report.append("\n## Conclusion")
        if m.accuracy_diff > 0.02:
            report.append("Dynamic NN outperforms Standard NN with significantly better accuracy.")
            if m.parameter_diff < 0:
                report.append("Additionally, Dynamic NN uses fewer parameters (more efficient).")
        elif m.accuracy_diff < -0.02:
            report.append("Standard NN outperforms Dynamic NN with better accuracy.")
        elif m.parameter_diff < 0:
            report.append("Models have similar accuracy, but Dynamic NN is more parameter-efficient.")
        else:
            report.append("Models show similar performance.")

        report.append("\n" + "=" * 70)

        return "\n".join(report)
