"""
Lightweight tokenizers — useful for end-to-end demos and tests without
pulling in ``tokenizers`` / ``sentencepiece``.

- ``WhitespaceTokenizer``: word-level vocab built from a corpus.
- ``CharTokenizer``: byte-level fallback (handy for tiny demos / RoPE tests).
- ``BPETokenizer``: a small from-scratch byte-pair-encoding implementation
  (training + encode + decode). Not optimised — meant for educational
  use and small corpora.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


@dataclass
class TokenizerOutput:
    input_ids: np.ndarray
    attention_mask: np.ndarray


class _Base:
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"

    def __init__(self) -> None:
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.eos_token]

    def _add_token(self, tok: str) -> int:
        if tok not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[tok] = idx
            self.id_to_token[idx] = tok
        return self.token_to_id[tok]

    def _add_specials(self) -> None:
        for s in SPECIAL_TOKENS:
            self._add_token(s)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        raise NotImplementedError

    def __call__(self, texts, max_length: Optional[int] = None,
                 padding: bool = True, truncation: bool = True,
                 add_special_tokens: bool = True) -> TokenizerOutput:
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]
        if truncation and max_length is not None:
            encoded = [e[:max_length] for e in encoded]
        if padding:
            target = max_length if max_length is not None else max(len(e) for e in encoded)
            pad = self.pad_token_id
            input_ids = np.full((len(encoded), target), pad, dtype=np.int64)
            attn = np.zeros((len(encoded), target), dtype=np.int64)
            for i, e in enumerate(encoded):
                input_ids[i, :len(e)] = e
                attn[i, :len(e)] = 1
        else:
            # Ragged — return list-of-lists wrapped in object array for completeness.
            input_ids = np.array(encoded, dtype=object)
            attn = np.array([[1] * len(e) for e in encoded], dtype=object)
        return TokenizerOutput(input_ids=input_ids, attention_mask=attn)


class WhitespaceTokenizer(_Base):
    """Word-level tokenizer: split on whitespace, vocab from corpus."""

    def __init__(self) -> None:
        super().__init__()
        self._add_specials()

    def fit(self, corpus: Sequence[str], min_freq: int = 1,
            max_vocab: Optional[int] = None) -> "WhitespaceTokenizer":
        c: Counter = Counter()
        for line in corpus:
            c.update(line.split())
        items = [(t, n) for t, n in c.items() if n >= min_freq]
        items.sort(key=lambda x: (-x[1], x[0]))
        if max_vocab is not None:
            items = items[: max_vocab - len(self.token_to_id)]
        for t, _ in items:
            self._add_token(t)
        return self

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids: List[int] = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
        for tok in text.split():
            ids.append(self.token_to_id.get(tok, self.unk_token_id))
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        out = []
        for i in ids:
            tok = self.id_to_token.get(int(i), self.unk_token)
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            out.append(tok)
        return " ".join(out)


class CharTokenizer(_Base):
    """One token per Unicode character. Useful for tiny demos."""

    def __init__(self) -> None:
        super().__init__()
        self._add_specials()

    def fit(self, corpus: Sequence[str]) -> "CharTokenizer":
        chars = sorted(set("".join(corpus)))
        for ch in chars:
            self._add_token(ch)
        return self

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
        for ch in text:
            ids.append(self.token_to_id.get(ch, self.unk_token_id))
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        out = []
        for i in ids:
            tok = self.id_to_token.get(int(i), self.unk_token)
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            out.append(tok)
        return "".join(out)


# ============================================================
# Tiny BPE
# ============================================================


class BPETokenizer(_Base):
    """Minimal byte-pair encoding tokenizer.

    Trains on a list of strings; ``num_merges`` controls vocab growth.
    Tokens are stored as space-separated symbol sequences ending in ``</w>``.
    """

    END_OF_WORD = "</w>"

    def __init__(self) -> None:
        super().__init__()
        self.merges: List[tuple] = []
        self._add_specials()

    def fit(self, corpus: Sequence[str], num_merges: int = 1000,
            min_freq: int = 1) -> "BPETokenizer":
        # Word frequencies as tuples of symbols.
        word_freqs: Dict[tuple, int] = defaultdict(int)
        for line in corpus:
            for w in line.split():
                if not w:
                    continue
                symbols = tuple(list(w) + [self.END_OF_WORD])
                word_freqs[symbols] += 1

        # Seed vocab with single chars + end-of-word marker.
        for symbols in word_freqs:
            for s in symbols:
                self._add_token(s)

        for _ in range(num_merges):
            pairs: Dict[tuple, int] = defaultdict(int)
            for word, freq in word_freqs.items():
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i + 1])] += freq
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            if pairs[best] < min_freq:
                break
            self.merges.append(best)
            merged_token = best[0] + best[1]
            self._add_token(merged_token)
            new_word_freqs: Dict[tuple, int] = {}
            for word, freq in word_freqs.items():
                new_word: List[str] = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                        new_word.append(merged_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word_freqs[tuple(new_word)] = freq
            word_freqs = new_word_freqs
        return self

    def _bpe_word(self, word: str) -> List[str]:
        symbols = list(word) + [self.END_OF_WORD]
        # Apply merges in order.
        for a, b in self.merges:
            i = 0
            new_symbols: List[str] = []
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(a + b)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids: List[int] = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
        for w in text.split():
            for sym in self._bpe_word(w):
                ids.append(self.token_to_id.get(sym, self.unk_token_id))
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        symbols = []
        for i in ids:
            tok = self.id_to_token.get(int(i), self.unk_token)
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            symbols.append(tok)
        text = "".join(symbols)
        # Convert end-of-word markers to spaces.
        return text.replace(self.END_OF_WORD, " ").strip()
