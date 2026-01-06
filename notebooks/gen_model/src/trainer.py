"""
Training module for Text Generation.
Wraps DynamicNetwork with generation-specific methods.
"""

import sys
import os
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add the pydnn package to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'python')))

from pydnn import DynamicNetwork

from .tokenizer import GenerativeTokenizer, normalize_features, one_hot_encode


@dataclass
class GenerationMetrics:
    """Metrics from training."""
    epochs_completed: int
    final_cost: float
    best_cost: float
    training_time_ms: int
    samples_per_second: float
    cost_history: List[float]
    efficiency_history: List[float]


class GenerativeTrainer:
    """
    Trainer wrapper for text generation using DynamicNetwork.
    Provides convenient methods for training and inference.
    """

    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        seed: int = 42,
        cost_function: str = "CrossEntropy"
    ):
        """
        Initialize the trainer.

        Args:
            input_vocab_size: Size of input vocabulary (BoW dimension)
            output_vocab_size: Size of output vocabulary (number of classes)
            seed: Random seed for reproducibility
            cost_function: Cost function to use
        """
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.seed = seed
        self.cost_function = cost_function

        self.network: Optional[DynamicNetwork] = None
        self.tokenizer: Optional[GenerativeTokenizer] = None
        self.is_trained = False

    def build_network(self) -> DynamicNetwork:
        """
        Build the DynamicNetwork.

        Returns:
            The built network
        """
        self.network = DynamicNetwork(
            input_shape=(self.input_vocab_size,),
            output_size=self.output_vocab_size,
            seed=self.seed,
            cost_function=self.cost_function
        )
        return self.network

    def set_tokenizer(self, tokenizer: GenerativeTokenizer) -> None:
        """
        Set the tokenizer for inference.

        Args:
            tokenizer: Fitted GenerativeTokenizer instance
        """
        self.tokenizer = tokenizer

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        verbose: bool = True
    ) -> GenerationMetrics:
        """
        Train the network.

        Args:
            X_train: Training features (n_samples, input_vocab_size)
            y_train: Training labels (n_samples, output_vocab_size) one-hot encoded
            verbose: Whether to print progress

        Returns:
            GenerationMetrics with training statistics
        """
        if self.network is None:
            self.build_network()

        if verbose:
            print(f"\nTraining on {len(X_train):,} samples...")
            print(f"  Input shape: {X_train.shape}")
            print(f"  Output shape: {y_train.shape}")

        start_time = time.time()

        result = self.network.fit(X_train, y_train, verbose=verbose)

        elapsed_ms = int((time.time() - start_time) * 1000)
        samples_per_sec = len(X_train) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0

        self.is_trained = True

        return GenerationMetrics(
            epochs_completed=result.epochs_completed,
            final_cost=result.final_cost,
            best_cost=result.best_cost,
            training_time_ms=elapsed_ms,
            samples_per_second=samples_per_sec,
            cost_history=result.cost_history,
            efficiency_history=result.efficiency_history
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Get predictions (probability distributions).

        Args:
            X: Input features (n_samples, input_vocab_size)

        Returns:
            Probability distributions (n_samples, output_vocab_size)
        """
        if self.network is None:
            raise ValueError("Network not built. Call build_network() first.")

        return self.network.predict(X)

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """
        Get predicted class indices.

        Args:
            X: Input features (n_samples, input_vocab_size)

        Returns:
            Predicted class indices (n_samples,)
        """
        probs = self.predict(X)
        return np.argmax(probs, axis=1)

    def generate_next_word(
        self,
        context: str,
        temperature: float = 1.0
    ) -> Tuple[str, float]:
        """
        Generate the next word given a context.

        Args:
            context: Input context text
            temperature: Sampling temperature (1.0 = no change, <1.0 = more deterministic)

        Returns:
            Tuple of (predicted_word, confidence)
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Call set_tokenizer() first.")

        # Convert context to features
        X = self.tokenizer.transform_input([context], verbose=False)
        X = normalize_features(X)

        # Get predictions
        probs = self.predict(X)[0]

        if temperature != 1.0:
            # Apply temperature scaling
            probs = np.power(probs, 1.0 / temperature)
            probs = probs / probs.sum()

        # Get top prediction
        predicted_idx = np.argmax(probs)
        predicted_word = self.tokenizer.output_id_to_word.get(predicted_idx, "<UNK>")
        confidence = float(probs[predicted_idx])

        return predicted_word, confidence

    def generate_sequence(
        self,
        seed_text: str,
        num_words: int = 5,
        temperature: float = 1.0,
        min_confidence: float = 0.1
    ) -> List[Tuple[str, float]]:
        """
        Generate a sequence of words autoregressively.

        Args:
            seed_text: Initial context text
            num_words: Number of words to generate
            temperature: Sampling temperature
            min_confidence: Stop if confidence drops below this

        Returns:
            List of (word, confidence) tuples
        """
        generated = []
        context = seed_text

        for _ in range(num_words):
            word, confidence = self.generate_next_word(context, temperature)
            generated.append((word, confidence))

            if confidence < min_confidence:
                break

            # Update context with new word (sliding window)
            context_words = context.split()[-4:] + [word]
            context = ' '.join(context_words)

        return generated

    def generate_with_sampling(
        self,
        context: str,
        top_k: int = 5,
        temperature: float = 1.0
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """
        Generate next word with sampling from top-k predictions.

        Args:
            context: Input context text
            top_k: Number of top candidates to sample from
            temperature: Sampling temperature

        Returns:
            Tuple of (sampled_word, top_k_predictions)
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Call set_tokenizer() first.")

        # Convert context to features
        X = self.tokenizer.transform_input([context], verbose=False)
        X = normalize_features(X)

        # Get predictions
        probs = self.predict(X)[0]

        # Apply temperature
        if temperature != 1.0:
            probs = np.power(probs, 1.0 / temperature)
            probs = probs / probs.sum()

        # Get top-k
        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_predictions = [
            (self.tokenizer.output_id_to_word.get(idx, "<UNK>"), float(probs[idx]))
            for idx in top_indices
        ]

        # Sample from top-k
        top_probs = np.array([probs[idx] for idx in top_indices])
        top_probs = top_probs / top_probs.sum()  # Renormalize
        sampled_idx = np.random.choice(len(top_indices), p=top_probs)
        sampled_word = self.tokenizer.output_id_to_word.get(top_indices[sampled_idx], "<UNK>")

        return sampled_word, top_predictions

    def get_health_status(self) -> Dict:
        """
        Get network health status.

        Returns:
            Dictionary with health metrics
        """
        if self.network is None:
            return {"error": "Network not built"}

        health = self.network.health_status()
        return {
            "state": str(health.state),
            "cancer_score": health.cancer_score,
            "alzheimer_score": health.alzheimer_score,
            "overall_health": health.overall_health,
            "diagnosis": health.diagnosis,
            "current_layers": health.current_layers,
            "current_nodes": health.current_nodes,
        }

    def save(self, path: str, name: str = "gen_model") -> None:
        """
        Save the trained model.

        Args:
            path: Directory path to save to
            name: Model name prefix
        """
        if self.network is None:
            raise ValueError("Network not built")

        self.network.save(path, name)

    @classmethod
    def load(cls, path: str) -> 'GenerativeTrainer':
        """
        Load a trained model.

        Args:
            path: Path to model JSON file

        Returns:
            GenerativeTrainer with loaded network
        """
        network = DynamicNetwork.load(path)

        trainer = cls(
            input_vocab_size=network.input_shape[0],
            output_vocab_size=network.output_size,
            seed=network.seed,
            cost_function=network.cost_function
        )
        trainer.network = network
        trainer.is_trained = True

        return trainer
