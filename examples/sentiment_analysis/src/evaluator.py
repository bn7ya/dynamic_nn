"""
Evaluation module for Sentiment Analysis.
Handles model evaluation, metrics calculation, and result visualization.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .word_banks import LABEL_NAMES


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    accuracy: float
    per_class_accuracy: Dict[str, float]
    confusion_matrix: np.ndarray
    prediction_time: float
    predictions_per_second: float


class SentimentEvaluator:
    """
    Evaluator class for sentiment analysis models.
    Calculates metrics and generates evaluation reports.
    """

    def __init__(self, label_names: Optional[Dict[int, str]] = None):
        """
        Initialize the evaluator.

        Args:
            label_names: Dictionary mapping label IDs to names
        """
        self.label_names = label_names or LABEL_NAMES
        self.num_classes = len(self.label_names)
        self.metrics: Optional[EvaluationMetrics] = None

    def evaluate(
        self,
        predictions: np.ndarray,
        y_true: np.ndarray,
        prediction_time: Optional[float] = None,
        verbose: bool = True
    ) -> EvaluationMetrics:
        """
        Evaluate model predictions.

        Args:
            predictions: Prediction probabilities (n_samples, n_classes)
            y_true: True labels (one-hot encoded)
            prediction_time: Time taken for predictions (optional)
            verbose: Whether to print results

        Returns:
            EvaluationMetrics with evaluation results
        """
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_true, axis=1)

        # Overall accuracy
        accuracy = np.mean(pred_classes == true_classes)

        # Per-class accuracy
        per_class_accuracy = {}
        for label_id, label_name in self.label_names.items():
            mask = true_classes == label_id
            if mask.sum() > 0:
                class_acc = np.mean(pred_classes[mask] == true_classes[mask])
                per_class_accuracy[label_name] = class_acc

        # Confusion matrix
        confusion = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for true, pred in zip(true_classes, pred_classes):
            confusion[true, pred] += 1

        # Prediction speed
        if prediction_time is not None and prediction_time > 0:
            predictions_per_second = len(predictions) / prediction_time
        else:
            predictions_per_second = 0.0

        self.metrics = EvaluationMetrics(
            accuracy=accuracy,
            per_class_accuracy=per_class_accuracy,
            confusion_matrix=confusion,
            prediction_time=prediction_time or 0.0,
            predictions_per_second=predictions_per_second
        )

        if verbose:
            self.print_report()

        return self.metrics

    def evaluate_from_trainer(
        self,
        trainer,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True
    ) -> EvaluationMetrics:
        """
        Evaluate a trainer's model on test data.

        Args:
            trainer: SentimentTrainer instance
            X_test: Test features
            y_test: Test labels (one-hot encoded)
            verbose: Whether to print results

        Returns:
            EvaluationMetrics with evaluation results
        """
        if verbose:
            print(f"\nPredicting on {X_test.shape[0]:,} test samples...")

        start_time = time.time()
        predictions = trainer.predict(X_test)
        prediction_time = time.time() - start_time

        return self.evaluate(
            predictions,
            y_test,
            prediction_time=prediction_time,
            verbose=verbose
        )

    def print_report(self) -> None:
        """Print formatted evaluation report."""
        if self.metrics is None:
            print("No evaluation metrics available.")
            return

        print(f"\nPrediction time: {self.metrics.prediction_time:.2f}s")
        print(f"Predictions per second: {self.metrics.predictions_per_second:,.0f}")
        print(f"\nTest Accuracy: {self.metrics.accuracy:.4f} ({self.metrics.accuracy*100:.2f}%)")

        print("\nPer-class performance:")
        for label_name, acc in self.metrics.per_class_accuracy.items():
            # Count samples for this class
            label_id = [k for k, v in self.label_names.items() if v == label_name][0]
            count = self.metrics.confusion_matrix[label_id].sum()
            print(f"  {label_name}: {acc:.4f} ({count:,} samples)")

        self.print_confusion_matrix()

    def print_confusion_matrix(self) -> None:
        """Print formatted confusion matrix."""
        if self.metrics is None:
            return

        print("\nConfusion Matrix:")
        print("              Predicted")

        # Header
        header = "              " + "  ".join([
            f"{name[:3]:>7}" for name in self.label_names.values()
        ])
        print(header)
        print("Actual")

        # Rows
        for i, name in self.label_names.items():
            row_values = "  ".join([
                f"{self.metrics.confusion_matrix[i, j]:>7,}"
                for j in range(self.num_classes)
            ])
            print(f"  {name:>8}  {row_values}")

    def get_classification_report(self) -> Dict:
        """
        Get detailed classification report.

        Returns:
            Dictionary with classification metrics
        """
        if self.metrics is None:
            return {"error": "No evaluation metrics available"}

        # Calculate precision, recall, F1 for each class
        report = {
            "accuracy": self.metrics.accuracy,
            "classes": {}
        }

        for label_id, label_name in self.label_names.items():
            tp = self.metrics.confusion_matrix[label_id, label_id]
            fp = self.metrics.confusion_matrix[:, label_id].sum() - tp
            fn = self.metrics.confusion_matrix[label_id, :].sum() - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            report["classes"][label_name] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": int(self.metrics.confusion_matrix[label_id, :].sum())
            }

        return report

    def show_prediction_examples(
        self,
        texts: List[str],
        predictions: np.ndarray,
        y_true: np.ndarray,
        n: int = 10
    ) -> None:
        """
        Show example predictions.

        Args:
            texts: List of text samples
            predictions: Prediction probabilities
            y_true: True labels (one-hot encoded)
            n: Number of examples to show
        """
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_true, axis=1)

        print("\nSample predictions:")
        for i in range(min(n, len(texts))):
            confidence = predictions[i, pred_classes[i]]
            correct = "OK" if pred_classes[i] == true_classes[i] else "X"

            text_preview = texts[i][:40] + "..." if len(texts[i]) > 40 else texts[i]

            print(f"  [{correct}] '{text_preview}'")
            print(f"      Predicted: {self.label_names[pred_classes[i]]} ({confidence:.2f})")
            print(f"      Actual: {self.label_names[true_classes[i]]}")
            print()
