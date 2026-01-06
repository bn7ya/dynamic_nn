"""
Tokenization module for Sentiment Analysis.
Provides text tokenization with label encoding and bag-of-words features.
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Optional


class SimpleTokenizer:
    """
    Simple tokenizer using split-based approach with label encoding.
    Maps each unique word to an integer ID.
    Supports both sequence output and bag-of-words output.
    """

    def __init__(self, max_vocab_size: Optional[int] = None):
        """
        Initialize the tokenizer.

        Args:
            max_vocab_size: Maximum vocabulary size (optional, limits to most frequent words)
        """
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.word_counts: Dict[str, int] = {}
        self.vocab_size: int = 0
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
        # Remove punctuation
        for char in '.,!?;:"\'()[]{}':
            text = text.replace(char, ' ')
        words = text.split()
        return words

    def fit(self, texts: List[str], verbose: bool = True) -> 'SimpleTokenizer':
        """
        Build vocabulary from texts.

        Args:
            texts: List of text strings
            verbose: Whether to print progress

        Returns:
            Self for method chaining
        """
        if verbose:
            print(f"  Building vocabulary from {len(texts):,} texts...")

        start_time = time.time()

        # Count word frequencies
        self.word_counts = {}
        for text in texts:
            words = self._tokenize(text)
            for word in words:
                self.word_counts[word] = self.word_counts.get(word, 0) + 1

        # Sort by frequency and limit vocab size if needed
        sorted_words = sorted(self.word_counts.items(), key=lambda x: -x[1])

        if self.max_vocab_size:
            sorted_words = sorted_words[:self.max_vocab_size - 1]

        # Create word-to-id mapping (label encoding)
        # Reserve 0 for padding/unknown
        self.word_to_id = {}
        self.id_to_word = {}

        for idx, (word, count) in enumerate(sorted_words, start=1):
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word

        self.vocab_size = len(self.word_to_id) + 1  # +1 for padding token
        self._fitted = True

        elapsed = time.time() - start_time

        if verbose:
            print(f"  Vocabulary built in {elapsed:.2f}s")
            print(f"  Vocabulary size: {self.vocab_size:,}")
            print(f"  Unique words found: {len(self.word_counts):,}")

        return self

    def transform_sequence(
        self,
        texts: List[str],
        max_length: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Convert texts to sequences of token IDs.

        Args:
            texts: List of text strings
            max_length: Maximum sequence length (optional)

        Returns:
            Tuple of (padded sequences array, max_length used)
        """
        if not self._fitted:
            raise ValueError("Tokenizer must be fitted before transform")

        sequences = []
        for text in texts:
            words = self._tokenize(text)
            sequence = [self.word_to_id.get(word, 0) for word in words]
            sequences.append(sequence)

        if max_length is None:
            max_length = max(len(seq) for seq in sequences)

        padded = np.zeros((len(sequences), max_length), dtype=np.float32)
        for i, seq in enumerate(sequences):
            length = min(len(seq), max_length)
            padded[i, :length] = seq[:length]

        return padded, max_length

    def transform_bow(
        self,
        texts: List[str],
        verbose: bool = True,
        batch_log_interval: int = 10000
    ) -> np.ndarray:
        """
        Convert texts to bag-of-words representation.

        Args:
            texts: List of text strings
            verbose: Whether to print progress
            batch_log_interval: How often to log progress

        Returns:
            Bag-of-words feature matrix
        """
        if not self._fitted:
            raise ValueError("Tokenizer must be fitted before transform")

        if verbose:
            print(f"  Converting {len(texts):,} texts to bag-of-words...")

        start_time = time.time()

        bow = np.zeros((len(texts), self.vocab_size), dtype=np.float32)

        for i, text in enumerate(texts):
            words = self._tokenize(text)
            for word in words:
                word_id = self.word_to_id.get(word, 0)
                bow[i, word_id] = 1.0  # Binary presence

            if verbose and (i + 1) % batch_log_interval == 0:
                print(f"    Processed {i + 1:,} / {len(texts):,} texts...")

        elapsed = time.time() - start_time

        if verbose:
            print(f"  Conversion completed in {elapsed:.2f}s")
            print(f"  Feature matrix shape: {bow.shape}")
            print(f"  Feature matrix size: {bow.nbytes / 1024 / 1024:.1f} MB")

        return bow

    def fit_transform_bow(
        self,
        texts: List[str],
        verbose: bool = True
    ) -> np.ndarray:
        """
        Fit and transform to bag-of-words in one step.

        Args:
            texts: List of text strings
            verbose: Whether to print progress

        Returns:
            Bag-of-words feature matrix
        """
        self.fit(texts, verbose=verbose)
        return self.transform_bow(texts, verbose=verbose)

    def fit_transform_sequence(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        verbose: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Fit and transform to sequences in one step.

        Args:
            texts: List of text strings
            max_length: Maximum sequence length (optional)
            verbose: Whether to print progress

        Returns:
            Tuple of (padded sequences array, max_length used)
        """
        self.fit(texts, verbose=verbose)
        return self.transform_sequence(texts, max_length)

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

        sorted_words = sorted(self.word_counts.items(), key=lambda x: -x[1])

        return {
            "vocab_size": self.vocab_size,
            "unique_words": len(self.word_counts),
            "top_words": sorted_words[:top_n],
            "sample_mappings": list(self.word_to_id.items())[:top_n]
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


def one_hot_encode(labels: List[int], num_classes: int) -> np.ndarray:
    """
    Convert labels to one-hot encoding.

    Args:
        labels: List of integer labels
        num_classes: Number of classes

    Returns:
        One-hot encoded labels matrix
    """
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
    for i, label in enumerate(labels):
        one_hot[i, label] = 1.0
    return one_hot
