"""
Arabic Tokenizer for Transformer Models

Provides word-level tokenization for Arabic text with:
- Separate input and output vocabularies
- Special tokens: [PAD], [UNK], [BOS], [EOS]
- Sequence padding and truncation
- Support for both Arabic and mixed (code-switching) text
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import re


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer"""
    max_seq_len: int = 32
    min_freq: int = 1
    pad_token: str = "[PAD]"
    unk_token: str = "[UNK]"
    bos_token: str = "[BOS]"
    eos_token: str = "[EOS]"


class ArabicTokenizer:
    """
    Word-level tokenizer for Arabic conversational text.

    Features:
    - Builds vocabulary from training data
    - Handles Arabic text with proper word segmentation
    - Supports mixed Arabic-English (code-switching)
    - Provides encoding/decoding functionality
    """

    SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()

        # Input vocabulary (questions/prompts)
        self.input_vocab: Dict[str, int] = {}
        self.input_vocab_inv: Dict[int, str] = {}

        # Output vocabulary (responses)
        self.output_vocab: Dict[str, int] = {}
        self.output_vocab_inv: Dict[int, str] = {}

        self._initialized = False

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        Handles Arabic and mixed Arabic-English text.
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Split on whitespace and punctuation, keeping Arabic characters together
        # This regex splits on spaces and common punctuation while preserving words
        tokens = re.findall(r'[\u0600-\u06FF]+|[a-zA-Z]+|[0-9]+|[^\s\u0600-\u06FFa-zA-Z0-9]', text)

        # Filter empty tokens and single punctuation (except meaningful ones)
        tokens = [t for t in tokens if t.strip() and (len(t) > 1 or t.isalnum() or t in '؟?!.،,')]

        return tokens

    def _build_vocab(
        self,
        texts: List[str],
        min_freq: int = 1
    ) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build vocabulary from list of texts"""
        # Count word frequencies
        word_freq = {}
        for text in texts:
            for token in self._tokenize(text):
                word_freq[token] = word_freq.get(token, 0) + 1

        # Start with special tokens
        vocab = {token: idx for idx, token in enumerate(self.SPECIAL_TOKENS)}

        # Add words meeting minimum frequency
        idx = len(self.SPECIAL_TOKENS)
        for word, freq in sorted(word_freq.items(), key=lambda x: -x[1]):
            if freq >= min_freq and word not in vocab:
                vocab[word] = idx
                idx += 1

        vocab_inv = {idx: word for word, idx in vocab.items()}

        return vocab, vocab_inv

    def fit(
        self,
        input_texts: List[str],
        output_texts: List[str],
        min_freq: int = None
    ) -> 'ArabicTokenizer':
        """
        Build vocabularies from training data.

        Args:
            input_texts: List of input texts (questions/prompts)
            output_texts: List of output texts (responses)
            min_freq: Minimum word frequency (default: from config)

        Returns:
            self for method chaining
        """
        if min_freq is None:
            min_freq = self.config.min_freq

        self.input_vocab, self.input_vocab_inv = self._build_vocab(input_texts, min_freq)
        self.output_vocab, self.output_vocab_inv = self._build_vocab(output_texts, min_freq)

        self._initialized = True

        return self

    def encode_input(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True
    ) -> np.ndarray:
        """
        Encode input text to token IDs.

        Args:
            text: Input text to encode
            add_special_tokens: Add [BOS] and [EOS] tokens
            max_length: Maximum sequence length (default: from config)
            padding: Pad to max_length

        Returns:
            numpy array of token IDs
        """
        if not self._initialized:
            raise ValueError("Tokenizer not initialized. Call fit() first.")

        if max_length is None:
            max_length = self.config.max_seq_len

        tokens = self._tokenize(text)

        # Add special tokens
        if add_special_tokens:
            tokens = [self.config.bos_token] + tokens + [self.config.eos_token]

        # Convert to IDs
        pad_id = self.input_vocab[self.config.pad_token]
        unk_id = self.input_vocab[self.config.unk_token]

        ids = [self.input_vocab.get(t, unk_id) for t in tokens]

        # Truncate if needed
        if len(ids) > max_length:
            ids = ids[:max_length]

        # Pad if needed
        if padding and len(ids) < max_length:
            ids = ids + [pad_id] * (max_length - len(ids))

        return np.array(ids, dtype=np.int32)

    def encode_output(
        self,
        text: str,
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
        padding: bool = True
    ) -> np.ndarray:
        """
        Encode output text to token IDs.

        Args:
            text: Output text to encode
            add_special_tokens: Add [BOS] and [EOS] tokens
            max_length: Maximum sequence length (default: from config)
            padding: Pad to max_length

        Returns:
            numpy array of token IDs
        """
        if not self._initialized:
            raise ValueError("Tokenizer not initialized. Call fit() first.")

        if max_length is None:
            max_length = self.config.max_seq_len

        tokens = self._tokenize(text)

        if add_special_tokens:
            tokens = [self.config.bos_token] + tokens + [self.config.eos_token]

        pad_id = self.output_vocab[self.config.pad_token]
        unk_id = self.output_vocab[self.config.unk_token]

        ids = [self.output_vocab.get(t, unk_id) for t in tokens]

        if len(ids) > max_length:
            ids = ids[:max_length]

        if padding and len(ids) < max_length:
            ids = ids + [pad_id] * (max_length - len(ids))

        return np.array(ids, dtype=np.int32)

    def encode_output_label(self, text: str) -> int:
        """
        Encode output text to single class label (first token after tokenization).
        Used for classification tasks where output is a single response.

        Args:
            text: Output text

        Returns:
            Single integer class ID
        """
        if not self._initialized:
            raise ValueError("Tokenizer not initialized. Call fit() first.")

        tokens = self._tokenize(text)

        if not tokens:
            return self.output_vocab[self.config.unk_token]

        # Return ID of first token (or full text if single word)
        full_text = ' '.join(tokens)
        if full_text in self.output_vocab:
            return self.output_vocab[full_text]

        # Try first token
        return self.output_vocab.get(tokens[0], self.output_vocab[self.config.unk_token])

    def batch_encode_inputs(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True
    ) -> np.ndarray:
        """Encode batch of input texts"""
        encoded = [
            self.encode_input(t, add_special_tokens, max_length, padding)
            for t in texts
        ]
        return np.stack(encoded)

    def batch_encode_outputs(
        self,
        texts: List[str],
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
        padding: bool = True
    ) -> np.ndarray:
        """Encode batch of output texts"""
        encoded = [
            self.encode_output(t, add_special_tokens, max_length, padding)
            for t in texts
        ]
        return np.stack(encoded)

    def build_output_label_mapping(self, texts: List[str]) -> None:
        """
        Build response-to-id mapping from training data.
        Must be called once with training outputs before encoding labels.
        """
        response_to_id = {}
        for text in texts:
            if text not in response_to_id:
                response_to_id[text] = len(response_to_id)

        self._response_to_id = response_to_id
        self._id_to_response = {v: k for k, v in response_to_id.items()}

    def batch_encode_output_labels(self, texts: List[str]) -> np.ndarray:
        """
        Encode batch of outputs as class labels.
        Call build_output_label_mapping() first with training data.
        """
        # If mapping doesn't exist, build it (backward compatibility)
        if not hasattr(self, '_response_to_id') or self._response_to_id is None:
            self.build_output_label_mapping(texts)

        # Use UNK (0) for unknown responses
        unk_id = 0
        labels = []
        for text in texts:
            if text in self._response_to_id:
                labels.append(self._response_to_id[text])
            else:
                # For val/test data with unseen responses, use closest match or UNK
                labels.append(unk_id)

        return np.array(labels, dtype=np.int32)

    def decode_input(
        self,
        ids: Union[np.ndarray, List[int]],
        skip_special_tokens: bool = True
    ) -> str:
        """Decode token IDs to input text"""
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()

        tokens = []
        for idx in ids:
            token = self.input_vocab_inv.get(idx, self.config.unk_token)
            if skip_special_tokens and token in self.SPECIAL_TOKENS:
                continue
            tokens.append(token)

        return ' '.join(tokens)

    def decode_output(
        self,
        ids: Union[np.ndarray, List[int]],
        skip_special_tokens: bool = True
    ) -> str:
        """Decode token IDs to output text"""
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()

        tokens = []
        for idx in ids:
            token = self.output_vocab_inv.get(idx, self.config.unk_token)
            if skip_special_tokens and token in self.SPECIAL_TOKENS:
                continue
            tokens.append(token)

        return ' '.join(tokens)

    def decode_output_label(self, class_id: int) -> str:
        """Decode class ID to response text"""
        if hasattr(self, '_id_to_response'):
            return self._id_to_response.get(class_id, "[UNK]")
        return self.output_vocab_inv.get(class_id, "[UNK]")

    @property
    def input_vocab_size(self) -> int:
        """Size of input vocabulary"""
        return len(self.input_vocab)

    @property
    def output_vocab_size(self) -> int:
        """Size of output vocabulary"""
        return len(self.output_vocab)

    @property
    def num_classes(self) -> int:
        """Number of output classes (for classification)"""
        if hasattr(self, '_response_to_id'):
            return len(self._response_to_id)
        return len(self.output_vocab)

    @property
    def pad_token_id(self) -> int:
        """ID of padding token"""
        return self.input_vocab[self.config.pad_token]

    @property
    def unk_token_id(self) -> int:
        """ID of unknown token"""
        return self.input_vocab[self.config.unk_token]

    @property
    def bos_token_id(self) -> int:
        """ID of beginning-of-sequence token"""
        return self.input_vocab[self.config.bos_token]

    @property
    def eos_token_id(self) -> int:
        """ID of end-of-sequence token"""
        return self.input_vocab[self.config.eos_token]

    def get_summary(self) -> Dict:
        """Get tokenizer summary statistics"""
        return {
            "input_vocab_size": self.input_vocab_size,
            "output_vocab_size": self.output_vocab_size,
            "num_classes": self.num_classes if hasattr(self, '_response_to_id') else self.output_vocab_size,
            "max_seq_len": self.config.max_seq_len,
            "special_tokens": self.SPECIAL_TOKENS,
        }

    def save(self, filepath: str) -> None:
        """Save tokenizer to file"""
        import json

        data = {
            "config": {
                "max_seq_len": self.config.max_seq_len,
                "min_freq": self.config.min_freq,
                "pad_token": self.config.pad_token,
                "unk_token": self.config.unk_token,
                "bos_token": self.config.bos_token,
                "eos_token": self.config.eos_token,
            },
            "input_vocab": self.input_vocab,
            "output_vocab": self.output_vocab,
        }

        if hasattr(self, '_response_to_id'):
            data["response_to_id"] = self._response_to_id

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'ArabicTokenizer':
        """Load tokenizer from file"""
        import json

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        config = TokenizerConfig(**data["config"])
        tokenizer = cls(config)

        tokenizer.input_vocab = data["input_vocab"]
        tokenizer.input_vocab_inv = {int(v): k for k, v in data["input_vocab"].items()}

        tokenizer.output_vocab = data["output_vocab"]
        tokenizer.output_vocab_inv = {int(v): k for k, v in data["output_vocab"].items()}

        if "response_to_id" in data:
            tokenizer._response_to_id = data["response_to_id"]
            tokenizer._id_to_response = {int(v): k for k, v in data["response_to_id"].items()}

        tokenizer._initialized = True

        return tokenizer


def create_tokenizer_from_samples(
    samples,  # List[ConversationSample]
    max_seq_len: int = 32,
    min_freq: int = 1
) -> ArabicTokenizer:
    """
    Convenience function to create tokenizer from ConversationSample list.

    Args:
        samples: List of ConversationSample objects
        max_seq_len: Maximum sequence length
        min_freq: Minimum word frequency for vocabulary

    Returns:
        Fitted ArabicTokenizer
    """
    config = TokenizerConfig(max_seq_len=max_seq_len, min_freq=min_freq)
    tokenizer = ArabicTokenizer(config)

    input_texts = [s.input_text for s in samples]
    output_texts = [s.output_text for s in samples]

    tokenizer.fit(input_texts, output_texts)

    return tokenizer
