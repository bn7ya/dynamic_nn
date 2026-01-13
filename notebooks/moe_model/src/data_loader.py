"""
Data loader for SlimPajama and synthetic text data.

Provides data loading utilities for training the MoE language model.
"""

import numpy as np
from typing import Tuple, List, Optional, Generator

try:
    from .tokenizer import SimpleTokenizer, CharTokenizer
except ImportError:
    from tokenizer import SimpleTokenizer, CharTokenizer


class SlimPajamaLoader:
    """
    Data loader for SlimPajama subset.

    Uses HuggingFace datasets for efficient streaming.
    Falls back to synthetic data if HuggingFace is unavailable.
    """

    def __init__(
        self,
        subset_size: int = 10000,
        max_seq_len: int = 128,
        tokenizer: Optional[SimpleTokenizer] = None,
        seed: int = 42
    ):
        self.subset_size = subset_size
        self.max_seq_len = max_seq_len
        self.seed = seed

        # Initialize tokenizer
        if tokenizer is None:
            self.tokenizer = SimpleTokenizer(vocab_size=10000)
        else:
            self.tokenizer = tokenizer

        self._hf_available = self._check_hf()

    def _check_hf(self) -> bool:
        """Check if HuggingFace datasets is available."""
        try:
            from datasets import load_dataset
            return True
        except ImportError:
            print("HuggingFace datasets not available. Will use synthetic data.")
            return False

    def load_dataset(
        self,
        build_vocab: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load SlimPajama subset for next-token prediction.

        Args:
            build_vocab: Whether to build tokenizer vocabulary

        Returns:
            X: Input token IDs (samples, seq_len-1)
            y: Target token IDs (samples, seq_len-1) - shifted by 1
        """
        if self._hf_available:
            try:
                return self._load_slimpajama(build_vocab)
            except Exception as e:
                print(f"Failed to load SlimPajama: {e}")
                print("Falling back to synthetic data.")

        return self._generate_synthetic_data(build_vocab)

    def _load_slimpajama(self, build_vocab: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from SlimPajama."""
        from datasets import load_dataset

        print(f"Loading SlimPajama subset ({self.subset_size} samples)...")

        # Load subset with streaming
        dataset = load_dataset(
            "cerebras/SlimPajama-627B",
            split="train",
            streaming=True
        )

        # Collect texts
        texts = []
        for i, sample in enumerate(dataset):
            if i >= self.subset_size:
                break
            text = sample.get('text', '')
            if len(text) > 50:  # Skip very short samples
                texts.append(text)

        print(f"Collected {len(texts)} text samples")

        # Build vocabulary
        if build_vocab and not self.tokenizer.vocab_built:
            self.tokenizer.build_vocab(texts, min_freq=2)

        # Tokenize and create training pairs
        return self._create_training_pairs(texts)

    def _generate_synthetic_data(self, build_vocab: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic text data for testing."""
        print("Generating synthetic training data...")

        np.random.seed(self.seed)

        # Simple templates for generating text
        templates = [
            "The {adj} {noun} {verb} the {adj2} {noun2}.",
            "In the {adj} {noun}, there was a {adj2} {noun2}.",
            "Once upon a time, a {adj} {noun} met a {adj2} {noun2}.",
            "The {noun} was very {adj} and the {noun2} was {adj2}.",
            "Every {noun} knows that {adj} things are {adj2}.",
            "When the {adj} {noun} arrived, the {noun2} was {adj2}.",
            "A {adj} {noun} walked into a {adj2} {noun2}.",
            "The story of the {adj} {noun} and the {adj2} {noun2}.",
        ]

        adjectives = [
            "big", "small", "fast", "slow", "bright", "dark", "happy", "sad",
            "old", "new", "good", "bad", "hot", "cold", "soft", "hard",
            "tall", "short", "wide", "narrow", "deep", "shallow", "loud", "quiet",
        ]

        nouns = [
            "cat", "dog", "bird", "fish", "tree", "house", "car", "book",
            "sun", "moon", "star", "river", "mountain", "ocean", "forest", "city",
            "child", "person", "world", "time", "way", "day", "man", "woman",
        ]

        verbs = [
            "saw", "found", "helped", "loved", "watched", "followed", "chased", "caught",
        ]

        # Generate texts
        texts = []
        for _ in range(self.subset_size):
            template = np.random.choice(templates)
            text = template.format(
                adj=np.random.choice(adjectives),
                adj2=np.random.choice(adjectives),
                noun=np.random.choice(nouns),
                noun2=np.random.choice(nouns),
                verb=np.random.choice(verbs),
            )
            # Repeat to make longer sequences
            text = " ".join([text] * np.random.randint(1, 4))
            texts.append(text)

        print(f"Generated {len(texts)} synthetic samples")

        # Build vocabulary
        if build_vocab and not self.tokenizer.vocab_built:
            self.tokenizer.build_vocab(texts, min_freq=1)

        return self._create_training_pairs(texts)

    def _create_training_pairs(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Create input-target pairs for next-token prediction."""
        X_list = []
        y_list = []

        for text in texts:
            # Encode text
            tokens = self.tokenizer.encode(
                text,
                max_length=self.max_seq_len,
                add_special_tokens=True,
                padding=True
            )

            if len(tokens) >= 2:
                # Next-token prediction: input[:-1] -> target[1:]
                x = tokens[:-1]
                y = tokens[1:]

                X_list.append(x)
                y_list.append(y)

        X = np.stack(X_list)
        y = np.stack(y_list)

        print(f"Created {len(X)} training pairs, sequence length: {X.shape[1]}")

        return X, y

    def get_batches(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool = True
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate batches from data.

        Args:
            X: Input data
            y: Target data
            batch_size: Batch size
            shuffle: Whether to shuffle data

        Yields:
            Tuples of (X_batch, y_batch)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            yield X[batch_indices], y[batch_indices]

    def train_val_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_ratio: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and validation sets."""
        n_samples = len(X)
        n_val = int(n_samples * val_ratio)

        # Shuffle indices
        indices = np.arange(n_samples)
        np.random.seed(self.seed)
        np.random.shuffle(indices)

        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        print(f"Train: {len(X_train)}, Validation: {len(X_val)}")

        return X_train, y_train, X_val, y_val


class TextDataGenerator:
    """
    Generator for various text patterns.

    Useful for testing and debugging without external data.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def generate_counting_data(
        self,
        n_samples: int = 1000,
        max_num: int = 100
    ) -> List[str]:
        """Generate counting sequences (1, 2, 3, ...)."""
        texts = []
        for _ in range(n_samples):
            start = np.random.randint(1, max_num - 10)
            length = np.random.randint(5, 15)
            sequence = list(range(start, start + length))
            text = " ".join(map(str, sequence))
            texts.append(text)
        return texts

    def generate_repetition_data(
        self,
        n_samples: int = 1000,
        vocab: Optional[List[str]] = None
    ) -> List[str]:
        """Generate repetition patterns (word word word)."""
        if vocab is None:
            vocab = ["hello", "world", "the", "a", "is", "was", "and", "or"]

        texts = []
        for _ in range(n_samples):
            word = np.random.choice(vocab)
            repeats = np.random.randint(2, 8)
            text = " ".join([word] * repeats)
            texts.append(text)
        return texts

    def generate_sentence_data(
        self,
        n_samples: int = 1000
    ) -> List[str]:
        """Generate simple sentences with patterns."""
        subjects = ["the cat", "a dog", "the bird", "a fish", "the man", "a woman"]
        verbs = ["runs", "walks", "jumps", "sleeps", "eats", "drinks"]
        objects = ["quickly", "slowly", "happily", "sadly", "loudly", "quietly"]

        texts = []
        for _ in range(n_samples):
            n_sentences = np.random.randint(1, 5)
            sentences = []
            for _ in range(n_sentences):
                subj = np.random.choice(subjects)
                verb = np.random.choice(verbs)
                obj = np.random.choice(objects)
                sentences.append(f"{subj} {verb} {obj}")
            text = ". ".join(sentences) + "."
            texts.append(text)
        return texts
