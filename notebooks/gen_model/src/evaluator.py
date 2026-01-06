"""
Evaluation module for Text Generation.
Provides comprehensive metrics and visualization for generation quality.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .tokenizer import GenerativeTokenizer, normalize_features
from .trainer import GenerativeTrainer


class GenerativeEvaluator:
    """
    Evaluator for text generation model performance.
    Calculates accuracy, perplexity, and per-pattern metrics.
    """

    def __init__(
        self,
        trainer: GenerativeTrainer,
        tokenizer: GenerativeTokenizer
    ):
        """
        Initialize the evaluator.

        Args:
            trainer: Trained GenerativeTrainer instance
            tokenizer: Fitted GenerativeTokenizer instance
        """
        self.trainer = trainer
        self.tokenizer = tokenizer

    def evaluate(
        self,
        input_texts: List[str],
        output_texts: List[str],
        pattern_types: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate model on test data.

        Args:
            input_texts: List of input texts
            output_texts: List of expected output words
            pattern_types: Optional list of pattern types for per-pattern metrics
            languages: Optional list of languages for per-language metrics
            verbose: Whether to print results

        Returns:
            Dictionary with evaluation metrics
        """
        if verbose:
            print(f"\nEvaluating on {len(input_texts):,} samples...")

        # Convert to features
        X = self.tokenizer.transform_input(input_texts, verbose=False)
        X = normalize_features(X)

        # Get predictions
        probs = self.trainer.predict(X)
        pred_indices = np.argmax(probs, axis=1)

        # Get true indices
        true_indices = self.tokenizer.transform_output(output_texts, verbose=False)

        # Calculate overall metrics
        exact_accuracy = np.mean(pred_indices == true_indices)

        # Top-3 accuracy
        top3_correct = 0
        for i, true_idx in enumerate(true_indices):
            top3_indices = np.argsort(probs[i])[-3:]
            if true_idx in top3_indices:
                top3_correct += 1
        top3_accuracy = top3_correct / len(true_indices)

        # Top-5 accuracy
        top5_correct = 0
        for i, true_idx in enumerate(true_indices):
            top5_indices = np.argsort(probs[i])[-5:]
            if true_idx in top5_indices:
                top5_correct += 1
        top5_accuracy = top5_correct / len(true_indices)

        # Perplexity (using cross-entropy)
        epsilon = 1e-10
        ce_losses = []
        for i, true_idx in enumerate(true_indices):
            if 0 <= true_idx < len(probs[i]):
                ce_losses.append(-np.log(probs[i][true_idx] + epsilon))
        avg_ce = np.mean(ce_losses) if ce_losses else float('inf')
        perplexity = np.exp(avg_ce)

        results = {
            "total_samples": len(input_texts),
            "exact_accuracy": float(exact_accuracy),
            "top3_accuracy": float(top3_accuracy),
            "top5_accuracy": float(top5_accuracy),
            "perplexity": float(perplexity),
            "avg_cross_entropy": float(avg_ce),
        }

        # Per-pattern metrics
        if pattern_types is not None:
            pattern_metrics = self._calculate_per_group_metrics(
                pred_indices, true_indices, pattern_types
            )
            results["per_pattern"] = pattern_metrics

        # Per-language metrics
        if languages is not None:
            language_metrics = self._calculate_per_group_metrics(
                pred_indices, true_indices, languages
            )
            results["per_language"] = language_metrics

        if verbose:
            self._print_results(results)

        return results

    def _calculate_per_group_metrics(
        self,
        pred_indices: np.ndarray,
        true_indices: np.ndarray,
        groups: List[str]
    ) -> Dict[str, Dict]:
        """
        Calculate metrics per group (pattern type or language).

        Args:
            pred_indices: Predicted class indices
            true_indices: True class indices
            groups: List of group labels

        Returns:
            Dictionary with per-group metrics
        """
        group_results = defaultdict(lambda: {"correct": 0, "total": 0})

        for pred, true, group in zip(pred_indices, true_indices, groups):
            group_results[group]["total"] += 1
            if pred == true:
                group_results[group]["correct"] += 1

        metrics = {}
        for group, counts in group_results.items():
            accuracy = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
            metrics[group] = {
                "accuracy": float(accuracy),
                "correct": counts["correct"],
                "total": counts["total"]
            }

        return dict(metrics)

    def _print_results(self, results: Dict) -> None:
        """Print evaluation results in formatted output."""
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)

        print(f"\nOverall Metrics:")
        print(f"  Total Samples:    {results['total_samples']:,}")
        print(f"  Exact Accuracy:   {results['exact_accuracy']:.2%}")
        print(f"  Top-3 Accuracy:   {results['top3_accuracy']:.2%}")
        print(f"  Top-5 Accuracy:   {results['top5_accuracy']:.2%}")
        print(f"  Perplexity:       {results['perplexity']:.2f}")

        if "per_pattern" in results:
            print(f"\nPer-Pattern Accuracy:")
            for pattern, metrics in sorted(results["per_pattern"].items()):
                print(f"  {pattern:15s}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")

        if "per_language" in results:
            print(f"\nPer-Language Accuracy:")
            for lang, metrics in sorted(results["per_language"].items()):
                print(f"  {lang:15s}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")

        print("=" * 50)

    def show_examples(
        self,
        input_texts: List[str],
        output_texts: List[str],
        pattern_types: Optional[List[str]] = None,
        n_examples: int = 10,
        show_errors_only: bool = False
    ) -> None:
        """
        Show prediction examples.

        Args:
            input_texts: List of input texts
            output_texts: List of expected output words
            pattern_types: Optional list of pattern types
            n_examples: Number of examples to show
            show_errors_only: If True, only show incorrect predictions
        """
        # Convert to features
        X = self.tokenizer.transform_input(input_texts, verbose=False)
        X = normalize_features(X)

        # Get predictions
        probs = self.trainer.predict(X)
        pred_indices = np.argmax(probs, axis=1)
        true_indices = self.tokenizer.transform_output(output_texts, verbose=False)

        print("\n" + "=" * 70)
        print("PREDICTION EXAMPLES")
        print("=" * 70)

        shown = 0
        for i in range(len(input_texts)):
            is_correct = pred_indices[i] == true_indices[i]

            if show_errors_only and is_correct:
                continue

            pred_word = self.tokenizer.output_id_to_word.get(pred_indices[i], "<UNK>")
            true_word = output_texts[i]
            confidence = probs[i][pred_indices[i]]

            status = "OK" if is_correct else "XX"
            pattern = pattern_types[i] if pattern_types else "N/A"

            print(f"\n[{status}] Pattern: {pattern}")
            print(f"  Input:      '{input_texts[i]}'")
            print(f"  Expected:   '{true_word}'")
            print(f"  Predicted:  '{pred_word}' ({confidence:.2%})")

            # Show top-3 predictions
            top3_indices = np.argsort(probs[i])[-3:][::-1]
            top3_words = [
                f"{self.tokenizer.output_id_to_word.get(idx, '<UNK>')} ({probs[i][idx]:.2%})"
                for idx in top3_indices
            ]
            print(f"  Top-3:      {', '.join(top3_words)}")

            shown += 1
            if shown >= n_examples:
                break

        print("\n" + "=" * 70)

    def confusion_matrix(
        self,
        input_texts: List[str],
        output_texts: List[str],
        top_n_classes: int = 10
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate confusion matrix for top-n most frequent classes.

        Args:
            input_texts: List of input texts
            output_texts: List of expected output words
            top_n_classes: Number of top classes to include

        Returns:
            Tuple of (confusion_matrix, class_labels)
        """
        from collections import Counter

        # Convert to features and get predictions
        X = self.tokenizer.transform_input(input_texts, verbose=False)
        X = normalize_features(X)
        pred_indices = self.trainer.predict_classes(X)
        true_indices = self.tokenizer.transform_output(output_texts, verbose=False)

        # Find top-n most frequent true classes
        true_counts = Counter(true_indices)
        top_classes = [cls for cls, _ in true_counts.most_common(top_n_classes)]

        # Create confusion matrix
        n_classes = len(top_classes)
        matrix = np.zeros((n_classes, n_classes), dtype=np.int32)

        class_to_idx = {cls: idx for idx, cls in enumerate(top_classes)}

        for true, pred in zip(true_indices, pred_indices):
            if true in class_to_idx and pred in class_to_idx:
                matrix[class_to_idx[true], class_to_idx[pred]] += 1

        # Get class labels
        class_labels = [
            self.tokenizer.output_id_to_word.get(cls, f"<{cls}>")
            for cls in top_classes
        ]

        return matrix, class_labels

    def generation_demo(
        self,
        test_contexts: Optional[List[str]] = None,
        temperature: float = 1.0
    ) -> None:
        """
        Demonstrate generation capabilities with test contexts.

        Args:
            test_contexts: Optional list of test contexts
            temperature: Sampling temperature
        """
        if test_contexts is None:
            test_contexts = [
                # English patterns
                "color red",
                "color blue",
                "one two three",
                "first second third",
                "fruit apple banana",
                "animal cat dog",
                "the big house is",
                "king man woman",
                # Arabic patterns
                "لون احمر",
                "واحد اثنان ثلاثة",
                "فاكهة تفاحة موزة",
                "ملك رجل امرأة",
            ]

        print("\n" + "=" * 60)
        print("GENERATION DEMO")
        print("=" * 60)

        for context in test_contexts:
            word, confidence = self.trainer.generate_next_word(context, temperature)
            print(f"\n  Context:    '{context}'")
            print(f"  Generated:  '{word}' ({confidence:.2%})")

            # Show alternatives
            _, top_preds = self.trainer.generate_with_sampling(context, top_k=3, temperature=temperature)
            alternatives = [f"{w} ({p:.0%})" for w, p in top_preds]
            print(f"  Top-3:      {', '.join(alternatives)}")

        print("\n" + "=" * 60)


def calculate_generation_quality(
    trainer: GenerativeTrainer,
    tokenizer: GenerativeTokenizer,
    test_contexts: List[str],
    expected_outputs: List[str]
) -> Dict:
    """
    Calculate generation quality metrics.

    Args:
        trainer: Trained trainer
        tokenizer: Fitted tokenizer
        test_contexts: Test input contexts
        expected_outputs: Expected output words

    Returns:
        Dictionary with quality metrics
    """
    correct = 0
    total_confidence = 0.0

    for context, expected in zip(test_contexts, expected_outputs):
        word, confidence = trainer.generate_next_word(context)
        if word.lower() == expected.lower():
            correct += 1
        total_confidence += confidence

    return {
        "accuracy": correct / len(test_contexts) if test_contexts else 0,
        "avg_confidence": total_confidence / len(test_contexts) if test_contexts else 0,
        "total_tests": len(test_contexts)
    }
