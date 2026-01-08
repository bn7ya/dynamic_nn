"""
Text Vectorizer for Emotional Classification

Converts text samples to numerical feature vectors using multiple techniques:
- Bag of Words (BoW)
- TF-IDF
- Character n-grams
- Simple word embeddings (learned)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from collections import Counter
import re


@dataclass
class VectorizerConfig:
    """Configuration for text vectorizer."""
    max_features: int = 1000  # Maximum vocabulary size
    min_freq: int = 2  # Minimum word frequency
    max_seq_len: int = 50  # Maximum sequence length
    use_tfidf: bool = True  # Use TF-IDF weighting
    use_ngrams: bool = True  # Include character n-grams
    ngram_range: tuple = (2, 4)  # Character n-gram range
    normalize: bool = True  # Normalize vectors


class TextVectorizer:
    """
    Converts text to numerical feature vectors.

    Supports:
    - Word-level features (BoW/TF-IDF)
    - Character-level n-grams
    - Combined feature vectors
    """

    def __init__(self, config: Optional[VectorizerConfig] = None):
        """Initialize vectorizer with config."""
        self.config = config or VectorizerConfig()
        self.vocabulary: Dict[str, int] = {}
        self.idf_weights: Dict[str, float] = {}
        self.ngram_vocab: Dict[str, int] = {}
        self._fitted = False

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple tokenization - split on whitespace and punctuation
        text = text.lower()
        # Handle Arabic text
        tokens = re.findall(r'[\w\u0600-\u06FF]+', text)
        return tokens

    def _extract_ngrams(self, text: str) -> List[str]:
        """Extract character n-grams from text."""
        text = text.lower()
        ngrams = []
        for n in range(self.config.ngram_range[0], self.config.ngram_range[1] + 1):
            for i in range(len(text) - n + 1):
                ngrams.append(text[i:i+n])
        return ngrams

    def fit(self, texts: List[str]) -> 'TextVectorizer':
        """
        Fit vectorizer on corpus of texts.

        Args:
            texts: List of text strings to fit on

        Returns:
            Self for chaining
        """
        # Count word frequencies
        word_counts = Counter()
        doc_counts = Counter()  # Document frequency for IDF
        ngram_counts = Counter()

        for text in texts:
            tokens = self._tokenize(text)
            word_counts.update(tokens)
            doc_counts.update(set(tokens))  # Count each word once per doc

            if self.config.use_ngrams:
                ngrams = self._extract_ngrams(text)
                ngram_counts.update(ngrams)

        # Build word vocabulary (filter by frequency)
        filtered_words = [
            word for word, count in word_counts.most_common()
            if count >= self.config.min_freq
        ][:self.config.max_features]

        self.vocabulary = {word: idx for idx, word in enumerate(filtered_words)}

        # Compute IDF weights
        n_docs = len(texts)
        for word in self.vocabulary:
            df = doc_counts.get(word, 1)
            self.idf_weights[word] = np.log((n_docs + 1) / (df + 1)) + 1

        # Build n-gram vocabulary
        if self.config.use_ngrams:
            # Use remaining feature slots for n-grams
            ngram_slots = max(0, self.config.max_features - len(self.vocabulary))
            filtered_ngrams = [
                ng for ng, count in ngram_counts.most_common()
                if count >= self.config.min_freq
            ][:ngram_slots]

            self.ngram_vocab = {
                ng: idx + len(self.vocabulary)
                for idx, ng in enumerate(filtered_ngrams)
            }

        self._fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to feature vectors.

        Args:
            texts: List of text strings to transform

        Returns:
            Feature matrix of shape (n_texts, n_features)
        """
        if not self._fitted:
            raise RuntimeError("Vectorizer must be fitted before transform")

        n_features = len(self.vocabulary) + len(self.ngram_vocab)
        features = np.zeros((len(texts), n_features), dtype=np.float32)

        for i, text in enumerate(texts):
            # Word features
            tokens = self._tokenize(text)
            token_counts = Counter(tokens)

            for token, count in token_counts.items():
                if token in self.vocabulary:
                    idx = self.vocabulary[token]
                    if self.config.use_tfidf:
                        # TF-IDF: term frequency * inverse document frequency
                        tf = count / (len(tokens) + 1)
                        idf = self.idf_weights.get(token, 1.0)
                        features[i, idx] = tf * idf
                    else:
                        # Simple count/BoW
                        features[i, idx] = count

            # N-gram features
            if self.config.use_ngrams:
                ngrams = self._extract_ngrams(text)
                ngram_counts = Counter(ngrams)

                for ngram, count in ngram_counts.items():
                    if ngram in self.ngram_vocab:
                        idx = self.ngram_vocab[ngram]
                        features[i, idx] = count / (len(ngrams) + 1)  # Normalized

            # Normalize vector
            if self.config.normalize:
                norm = np.linalg.norm(features[i])
                if norm > 0:
                    features[i] /= norm

        return features

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(texts).transform(texts)

    @property
    def n_features(self) -> int:
        """Get number of features."""
        return len(self.vocabulary) + len(self.ngram_vocab)

    def get_summary(self) -> Dict:
        """Get vectorizer summary."""
        return {
            'vocab_size': len(self.vocabulary),
            'ngram_vocab_size': len(self.ngram_vocab),
            'total_features': self.n_features,
            'use_tfidf': self.config.use_tfidf,
            'use_ngrams': self.config.use_ngrams,
            'fitted': self._fitted,
        }

    def save(self, path: str) -> None:
        """Save vectorizer to file."""
        import json

        data = {
            'config': {
                'max_features': self.config.max_features,
                'min_freq': self.config.min_freq,
                'max_seq_len': self.config.max_seq_len,
                'use_tfidf': self.config.use_tfidf,
                'use_ngrams': self.config.use_ngrams,
                'ngram_range': list(self.config.ngram_range),
                'normalize': self.config.normalize,
            },
            'vocabulary': self.vocabulary,
            'idf_weights': self.idf_weights,
            'ngram_vocab': self.ngram_vocab,
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'TextVectorizer':
        """Load vectorizer from file."""
        import json

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        config = VectorizerConfig(
            max_features=data['config']['max_features'],
            min_freq=data['config']['min_freq'],
            max_seq_len=data['config']['max_seq_len'],
            use_tfidf=data['config']['use_tfidf'],
            use_ngrams=data['config']['use_ngrams'],
            ngram_range=tuple(data['config']['ngram_range']),
            normalize=data['config']['normalize'],
        )

        vectorizer = cls(config)
        vectorizer.vocabulary = data['vocabulary']
        vectorizer.idf_weights = data['idf_weights']
        vectorizer.ngram_vocab = data['ngram_vocab']
        vectorizer._fitted = True

        return vectorizer


def create_vectorizer(
    texts: List[str],
    max_features: int = 1000,
    use_tfidf: bool = True,
    use_ngrams: bool = True
) -> TextVectorizer:
    """
    Convenience function to create and fit a vectorizer.

    Args:
        texts: Training texts
        max_features: Maximum vocabulary size
        use_tfidf: Use TF-IDF weighting
        use_ngrams: Include character n-grams

    Returns:
        Fitted TextVectorizer
    """
    config = VectorizerConfig(
        max_features=max_features,
        use_tfidf=use_tfidf,
        use_ngrams=use_ngrams,
    )

    vectorizer = TextVectorizer(config)
    vectorizer.fit(texts)
    return vectorizer
