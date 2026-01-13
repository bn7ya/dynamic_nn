"""
Simple tokenizer for text processing.

Implements a basic word-level tokenizer with special tokens,
following the pattern from notebooks/gen_model/src/tokenizer.py.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
import re


class SimpleTokenizer:
    """
    Simple word-level tokenizer with special tokens.

    Features:
    - Word-level tokenization with basic preprocessing
    - Special tokens: PAD, UNK, BOS, EOS
    - Vocabulary building from corpus
    - Encode/decode methods
    """

    # Special tokens
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    BOS_TOKEN = '<BOS>'
    EOS_TOKEN = '<EOS>'

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    def __init__(self, vocab_size: int = 10000, lowercase: bool = True):
        self.vocab_size = vocab_size
        self.lowercase = lowercase

        # Initialize with special tokens
        self.token_to_id: Dict[str, int] = {
            self.PAD_TOKEN: self.PAD_ID,
            self.UNK_TOKEN: self.UNK_ID,
            self.BOS_TOKEN: self.BOS_ID,
            self.EOS_TOKEN: self.EOS_ID,
        }
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.token_to_id.items()}

        self.vocab_built = False

    def build_vocab(self, texts: List[str], min_freq: int = 2) -> None:
        """
        Build vocabulary from a list of texts.

        Args:
            texts: List of text strings
            min_freq: Minimum frequency for a token to be included
        """
        # Count token frequencies
        token_counts = Counter()

        for text in texts:
            tokens = self._tokenize(text)
            token_counts.update(tokens)

        # Sort by frequency and add to vocabulary
        sorted_tokens = sorted(token_counts.items(), key=lambda x: -x[1])

        # Reserve space for special tokens
        available_slots = self.vocab_size - len(self.token_to_id)

        for token, count in sorted_tokens[:available_slots]:
            if count >= min_freq and token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

        self.vocab_built = True
        print(f"Vocabulary built: {len(self.token_to_id)} tokens")

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if self.lowercase:
            text = text.lower()

        # Basic cleaning
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace

        # Split into words
        tokens = text.strip().split()

        return tokens

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
        padding: bool = True
    ) -> np.ndarray:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            max_length: Maximum sequence length (pad/truncate to this)
            add_special_tokens: Whether to add BOS/EOS tokens
            padding: Whether to pad to max_length

        Returns:
            Array of token IDs
        """
        tokens = self._tokenize(text)

        # Convert to IDs
        ids = []

        if add_special_tokens:
            ids.append(self.BOS_ID)

        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                ids.append(self.UNK_ID)

        if add_special_tokens:
            ids.append(self.EOS_ID)

        # Truncate if needed
        if max_length is not None:
            ids = ids[:max_length]

            # Pad if needed
            if padding and len(ids) < max_length:
                ids = ids + [self.PAD_ID] * (max_length - len(ids))

        return np.array(ids, dtype=np.int64)

    def decode(self, ids: np.ndarray, skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            ids: Array of token IDs
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text
        """
        tokens = []

        special_ids = {self.PAD_ID, self.UNK_ID, self.BOS_ID, self.EOS_ID}

        for id in ids:
            id = int(id)

            if skip_special_tokens and id in special_ids:
                if id == self.EOS_ID:
                    break  # Stop at EOS
                continue

            if id in self.id_to_token:
                tokens.append(self.id_to_token[id])
            else:
                tokens.append(self.UNK_TOKEN)

        return ' '.join(tokens)

    def batch_encode(
        self,
        texts: List[str],
        max_length: int,
        add_special_tokens: bool = True
    ) -> np.ndarray:
        """
        Encode a batch of texts.

        Args:
            texts: List of text strings
            max_length: Maximum sequence length

        Returns:
            2D array of token IDs (batch_size, max_length)
        """
        encoded = []
        for text in texts:
            ids = self.encode(text, max_length, add_special_tokens, padding=True)
            encoded.append(ids)

        return np.stack(encoded)

    def batch_decode(self, batch_ids: np.ndarray, skip_special_tokens: bool = True) -> List[str]:
        """Decode a batch of token IDs."""
        return [self.decode(ids, skip_special_tokens) for ids in batch_ids]

    @property
    def actual_vocab_size(self) -> int:
        """Get the actual vocabulary size (may be less than max)."""
        return len(self.token_to_id)

    def save(self, path: str) -> None:
        """Save tokenizer vocabulary to file."""
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab_size': self.vocab_size,
                'lowercase': self.lowercase,
                'token_to_id': self.token_to_id,
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'SimpleTokenizer':
        """Load tokenizer from file."""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tokenizer = cls(
            vocab_size=data['vocab_size'],
            lowercase=data['lowercase']
        )
        tokenizer.token_to_id = data['token_to_id']
        tokenizer.id_to_token = {int(v): k for k, v in tokenizer.token_to_id.items()}
        tokenizer.vocab_built = True

        return tokenizer


class CharTokenizer:
    """
    Character-level tokenizer for simple experiments.

    Useful for smaller-scale testing without needing a large vocabulary.
    """

    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    BOS_TOKEN = '<BOS>'
    EOS_TOKEN = '<EOS>'

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase

        # Special tokens
        self.token_to_id: Dict[str, int] = {
            self.PAD_TOKEN: self.PAD_ID,
            self.UNK_TOKEN: self.UNK_ID,
            self.BOS_TOKEN: self.BOS_ID,
            self.EOS_TOKEN: self.EOS_ID,
        }
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.token_to_id.items()}

    def build_vocab(self, texts: List[str]) -> None:
        """Build character vocabulary from texts."""
        chars = set()
        for text in texts:
            if self.lowercase:
                text = text.lower()
            chars.update(text)

        for char in sorted(chars):
            if char not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[char] = idx
                self.id_to_token[idx] = char

        print(f"Character vocabulary built: {len(self.token_to_id)} characters")

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
        padding: bool = True
    ) -> np.ndarray:
        """Encode text to character IDs."""
        if self.lowercase:
            text = text.lower()

        ids = []
        if add_special_tokens:
            ids.append(self.BOS_ID)

        for char in text:
            ids.append(self.token_to_id.get(char, self.UNK_ID))

        if add_special_tokens:
            ids.append(self.EOS_ID)

        if max_length is not None:
            ids = ids[:max_length]
            if padding and len(ids) < max_length:
                ids = ids + [self.PAD_ID] * (max_length - len(ids))

        return np.array(ids, dtype=np.int64)

    def decode(self, ids: np.ndarray, skip_special_tokens: bool = True) -> str:
        """Decode character IDs to text."""
        chars = []
        special_ids = {self.PAD_ID, self.UNK_ID, self.BOS_ID, self.EOS_ID}

        for id in ids:
            id = int(id)
            if skip_special_tokens and id in special_ids:
                if id == self.EOS_ID:
                    break
                continue
            chars.append(self.id_to_token.get(id, '?'))

        return ''.join(chars)

    @property
    def actual_vocab_size(self) -> int:
        return len(self.token_to_id)
