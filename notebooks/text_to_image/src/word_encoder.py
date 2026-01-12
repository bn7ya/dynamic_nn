"""Word encoding module for text-to-image generation."""

import json
import numpy as np
from typing import List, Union


class WordEncoder:
    """One-hot encoder for pattern vocabulary."""

    def __init__(self, vocabulary: List[str]):
        """Initialize encoder with vocabulary.

        Args:
            vocabulary: List of words to encode
        """
        self.vocabulary = vocabulary
        self.vocab_size = len(vocabulary)
        self.word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
        self.idx_to_word = {idx: word for idx, word in enumerate(vocabulary)}

    def encode(self, idx_or_word: Union[int, str]) -> np.ndarray:
        """Encode a single word or index to one-hot vector.

        Args:
            idx_or_word: Either a word string or integer index

        Returns:
            One-hot encoded vector of shape (vocab_size,)
        """
        if isinstance(idx_or_word, str):
            if idx_or_word not in self.word_to_idx:
                raise ValueError(f"Unknown word: {idx_or_word}. Must be one of {self.vocabulary}")
            idx = self.word_to_idx[idx_or_word]
        else:
            idx = int(idx_or_word)

        one_hot = np.zeros(self.vocab_size, dtype=np.float32)
        one_hot[idx] = 1.0
        return one_hot

    def encode_batch(self, indices: np.ndarray) -> np.ndarray:
        """Encode batch of indices to one-hot matrix.

        Args:
            indices: Array of integer indices

        Returns:
            One-hot encoded matrix of shape (batch_size, vocab_size)
        """
        batch_size = len(indices)
        one_hot = np.zeros((batch_size, self.vocab_size), dtype=np.float32)
        for i, idx in enumerate(indices):
            one_hot[i, int(idx)] = 1.0
        return one_hot

    def decode(self, one_hot: np.ndarray) -> str:
        """Decode one-hot vector to word.

        Args:
            one_hot: One-hot encoded vector

        Returns:
            Decoded word string
        """
        idx = int(np.argmax(one_hot))
        return self.idx_to_word[idx]

    def decode_batch(self, one_hot_batch: np.ndarray) -> List[str]:
        """Decode batch of one-hot vectors to words.

        Args:
            one_hot_batch: One-hot encoded matrix

        Returns:
            List of decoded word strings
        """
        indices = np.argmax(one_hot_batch, axis=1)
        return [self.idx_to_word[int(idx)] for idx in indices]

    def save(self, filepath: str) -> None:
        """Save encoder to JSON file.

        Args:
            filepath: Path to save the encoder
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({'vocabulary': self.vocabulary}, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'WordEncoder':
        """Load encoder from JSON file.

        Args:
            filepath: Path to load the encoder from

        Returns:
            Loaded WordEncoder instance
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(vocabulary=data['vocabulary'])

    def __repr__(self) -> str:
        return f"WordEncoder(vocab_size={self.vocab_size}, vocabulary={self.vocabulary})"
