"""
Tokenization module for Text Generation.
Provides text tokenization with separate input/output vocabularies.
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Optional


class GenerativeTokenizer:
    """
    Tokenizer for text generation tasks.
    Uses Bag-of-Words for input context and class encoding for output words.
    Maintains separate vocabularies for input and output.
    """

    def __init__(self, max_vocab_size: Optional[int] = None):
        """
        Initialize the tokenizer.

        Args:
            max_vocab_size: Maximum vocabulary size for input (optional)
        """
        # Input vocabulary (for context)
        self.input_word_to_id: Dict[str, int] = {}
        self.input_id_to_word: Dict[int, str] = {}
        self.input_word_counts: Dict[str, int] = {}
        self.input_vocab_size: int = 0

        # Output vocabulary (for predictions)
        self.output_word_to_id: Dict[str, int] = {}
        self.output_id_to_word: Dict[int, str] = {}
        self.output_vocab_size: int = 0

        self.max_vocab_size = max_vocab_size
        self._fitted = False

    def _tokenize(self, text: str) -> List[str]:
        """
        Split text into tokens (words).

        Args:
            text: Input text string

        Returns:
            List of token strings
        """
        text = text.lower()
        # Remove punctuation but keep Arabic characters
        for char in '.,!?;:"\'()[]{}':
            text = text.replace(char, ' ')
        words = text.split()
        return words

    def fit(
        self,
        input_texts: List[str],
        output_texts: List[str],
        verbose: bool = True
    ) -> 'GenerativeTokenizer':
        """
        Build vocabularies from input and output texts.

        Args:
            input_texts: List of input text strings
            output_texts: List of output text strings (single words)
            verbose: Whether to print progress

        Returns:
            Self for method chaining
        """
        if verbose:
            print(f"  Building vocabularies from {len(input_texts):,} samples...")

        start_time = time.time()

        # Build input vocabulary from context texts
        self.input_word_counts = {}
        for text in input_texts:
            words = self._tokenize(text)
            for word in words:
                self.input_word_counts[word] = self.input_word_counts.get(word, 0) + 1

        # Also include output words in input vocabulary (for autoregressive generation)
        for word in output_texts:
            word_lower = word.lower().strip()
            self.input_word_counts[word_lower] = self.input_word_counts.get(word_lower, 0) + 1

        # Sort by frequency and limit if needed
        sorted_words = sorted(self.input_word_counts.items(), key=lambda x: -x[1])

        if self.max_vocab_size:
            sorted_words = sorted_words[:self.max_vocab_size - 1]

        # Create input word-to-id mapping (reserve 0 for unknown/padding)
        self.input_word_to_id = {}
        self.input_id_to_word = {}

        for idx, (word, count) in enumerate(sorted_words, start=1):
            self.input_word_to_id[word] = idx
            self.input_id_to_word[idx] = word

        self.input_vocab_size = len(self.input_word_to_id) + 1  # +1 for padding

        # Build output vocabulary (unique output words)
        unique_outputs = sorted(set(word.lower().strip() for word in output_texts))

        self.output_word_to_id = {}
        self.output_id_to_word = {}

        for idx, word in enumerate(unique_outputs):
            self.output_word_to_id[word] = idx
            self.output_id_to_word[idx] = word

        self.output_vocab_size = len(self.output_word_to_id)

        self._fitted = True

        elapsed = time.time() - start_time

        if verbose:
            print(f"  Vocabularies built in {elapsed:.2f}s")
            print(f"  Input vocabulary size: {self.input_vocab_size:,}")
            print(f"  Output vocabulary size: {self.output_vocab_size:,}")
            print(f"  Unique input words: {len(self.input_word_counts):,}")

        return self

    def transform_input(
        self,
        texts: List[str],
        verbose: bool = True,
        batch_log_interval: int = 10000
    ) -> np.ndarray:
        """
        Convert input texts to bag-of-words representation.

        Args:
            texts: List of text strings
            verbose: Whether to print progress
            batch_log_interval: How often to log progress

        Returns:
            Bag-of-words feature matrix (n_samples, input_vocab_size)
        """
        if not self._fitted:
            raise ValueError("Tokenizer must be fitted before transform")

        if verbose:
            print(f"  Converting {len(texts):,} texts to bag-of-words...")

        start_time = time.time()

        bow = np.zeros((len(texts), self.input_vocab_size), dtype=np.float32)

        for i, text in enumerate(texts):
            words = self._tokenize(text)
            for word in words:
                word_id = self.input_word_to_id.get(word, 0)
                bow[i, word_id] = 1.0  # Binary presence

            if verbose and (i + 1) % batch_log_interval == 0:
                print(f"    Processed {i + 1:,} / {len(texts):,} texts...")

        elapsed = time.time() - start_time

        if verbose:
            print(f"  Conversion completed in {elapsed:.2f}s")
            print(f"  Feature matrix shape: {bow.shape}")
            print(f"  Feature matrix size: {bow.nbytes / 1024 / 1024:.1f} MB")

        return bow

    def transform_output(
        self,
        output_words: List[str],
        verbose: bool = True
    ) -> np.ndarray:
        """
        Convert output words to class indices.

        Args:
            output_words: List of output word strings
            verbose: Whether to print progress

        Returns:
            Array of class indices
        """
        if not self._fitted:
            raise ValueError("Tokenizer must be fitted before transform")

        if verbose:
            print(f"  Converting {len(output_words):,} output words to indices...")

        indices = np.array([
            self.output_word_to_id.get(word.lower().strip(), 0)
            for word in output_words
        ], dtype=np.int32)

        if verbose:
            print(f"  Output indices shape: {indices.shape}")

        return indices

    def decode_output(self, indices: np.ndarray) -> List[str]:
        """
        Convert class indices back to words.

        Args:
            indices: Array of class indices

        Returns:
            List of output words
        """
        return [self.output_id_to_word.get(int(idx), "<UNK>") for idx in indices]

    def decode_probabilities(
        self,
        probs: np.ndarray,
        top_k: int = 5
    ) -> List[List[Tuple[str, float]]]:
        """
        Decode probability distributions to top-k word predictions.

        Args:
            probs: Probability matrix (n_samples, output_vocab_size)
            top_k: Number of top predictions to return

        Returns:
            List of lists of (word, probability) tuples
        """
        results = []
        for prob_dist in probs:
            top_indices = np.argsort(prob_dist)[-top_k:][::-1]
            predictions = [
                (self.output_id_to_word.get(idx, "<UNK>"), float(prob_dist[idx]))
                for idx in top_indices
            ]
            results.append(predictions)
        return results

    def get_vocab_summary(self, top_n: int = 10) -> Dict:
        """
        Get vocabulary summary statistics.

        Args:
            top_n: Number of top words to include

        Returns:
            Dictionary with vocabulary statistics
        """
        if not self._fitted:
            return {"error": "Tokenizer not fitted"}

        sorted_input = sorted(self.input_word_counts.items(), key=lambda x: -x[1])

        return {
            "input_vocab_size": self.input_vocab_size,
            "output_vocab_size": self.output_vocab_size,
            "unique_input_words": len(self.input_word_counts),
            "top_input_words": sorted_input[:top_n],
            "sample_input_mappings": list(self.input_word_to_id.items())[:top_n],
            "sample_output_mappings": list(self.output_word_to_id.items())[:top_n],
        }


def normalize_features(X: np.ndarray) -> np.ndarray:
    """
    Normalize features using min-max normalization.

    Args:
        X: Feature matrix

    Returns:
        Normalized feature matrix
    """
    X_min = X.min()
    X_max = X.max()

    if X_max - X_min > 0:
        return (X - X_min) / (X_max - X_min)
    return X


def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert labels to one-hot encoding.

    Args:
        labels: Array of integer labels
        num_classes: Number of classes

    Returns:
        One-hot encoded labels matrix
    """
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
    for i, label in enumerate(labels):
        if 0 <= label < num_classes:
            one_hot[i, label] = 1.0
    return one_hot
