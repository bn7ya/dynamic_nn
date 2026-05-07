"""
moe_utils.py — helpers shared across the small-MoE-LLM training notebook.

Provides:
  TokenPacker        — buffers token streams into (B, seq_len+1) numpy batches
  save_checkpoint    — serialise model + optimizer + training state to disk
  load_checkpoint    — restore everything from a previous save
  make_text_stream   — generator that wraps a text corpus into token-ID arrays
  smooth             — rolling-average helper for loss plots
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Iterator, List, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# TokenPacker
# ---------------------------------------------------------------------------

class TokenPacker:
    """Buffers incoming token-ID arrays and drains fixed-size (B, T+1) windows.

    Documents flow in as 1-D int64 arrays (already BOS/EOS-wrapped).
    When the internal buffer holds at least batch_size * (seq_len+1) tokens,
    drain_batch() slices out one batch without copying the rest.
    """

    def __init__(self, seq_len: int, pad_id: int = 0):
        self.seq_len = seq_len
        self.pad_id = pad_id
        self._buf: List[int] = []

    def feed(self, token_ids: np.ndarray) -> None:
        self._buf.extend(token_ids.tolist())

    def has_batch(self, batch_size: int) -> bool:
        return len(self._buf) >= batch_size * (self.seq_len + 1)

    def drain_batch(self, batch_size: int) -> np.ndarray:
        needed = batch_size * (self.seq_len + 1)
        chunk = np.array(self._buf[:needed], dtype=np.int64)
        self._buf = self._buf[needed:]
        return chunk.reshape(batch_size, self.seq_len + 1)

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, step: int, loss_history: List[float],
                    cfg, ckpt_dir: Union[str, pathlib.Path], tag: str = 'latest') -> None:
    """Persist model weights, optimizer moments, and training metadata.

    Creates three files under ckpt_dir:
      weights_{tag}.npz   — named parameter arrays
      optim_{tag}.npz     — AdamW moment arrays + step counter
      meta_{tag}.json     — TransformerConfig + step + loss_history
    """
    path = pathlib.Path(ckpt_dir)
    path.mkdir(parents=True, exist_ok=True)

    # 1. Model weights — keyed by stable named_parameters() names
    weights = {name: param.data for name, param in model.named_parameters()}
    np.savez(path / f'weights_{tag}.npz', **weights)

    # 2. Optimizer state
    opt_arrays: dict = {'t': np.array(optimizer.t), 'lr': np.array(optimizer.lr)}
    for i, (m, v) in enumerate(zip(optimizer._m, optimizer._v)):
        opt_arrays[f'm_{i}'] = m
        opt_arrays[f'v_{i}'] = v
    np.savez(path / f'optim_{tag}.npz', **opt_arrays)

    # 3. Metadata
    meta = {
        'step': step,
        'loss_history': loss_history,
        'config': dataclasses.asdict(cfg),
    }
    with open(path / f'meta_{tag}.json', 'w') as fh:
        json.dump(meta, fh, indent=2)


def load_checkpoint(ckpt_dir: Union[str, pathlib.Path], tag: str = 'latest'):
    """Restore model, optimizer, step, loss_history, and cfg from disk.

    Returns (model, optimizer, step, loss_history, cfg).
    Requires pydnn to be importable (sys.path already patched by the notebook).
    """
    import sys
    # Lazy import to avoid hard dep at module-import time
    from pydnn.transformer import DynamicTransformer, TransformerConfig
    from pydnn.transformer.training import AdamW

    path = pathlib.Path(ckpt_dir)

    with open(path / f'meta_{tag}.json') as fh:
        meta = json.load(fh)

    cfg = TransformerConfig(**meta['config'])
    model = DynamicTransformer(cfg)

    # Restore weights by name
    weights_npz = np.load(path / f'weights_{tag}.npz')
    param_dict = dict(model.named_parameters())
    for name, param in param_dict.items():
        if name in weights_npz:
            param.data[...] = weights_npz[name]

    # Rebuild optimizer
    params = list(model.parameters())
    optimizer = AdamW(params, lr=float(meta.get('lr', 3e-4)),
                      betas=(0.9, 0.95), weight_decay=0.01)
    opt_npz = np.load(path / f'optim_{tag}.npz')
    optimizer.t = int(opt_npz['t'])
    optimizer.lr = float(opt_npz['lr'])
    optimizer._m = [opt_npz[f'm_{i}'] for i in range(len(params))]
    optimizer._v = [opt_npz[f'v_{i}'] for i in range(len(params))]

    return model, optimizer, meta['step'], meta['loss_history'], cfg


# ---------------------------------------------------------------------------
# Text stream
# ---------------------------------------------------------------------------

def make_text_stream(
    raw_texts: List[str],
    hf_tokenizer=None,
    fallback_tokenizer=None,
    repeat: bool = True,
) -> Iterator[np.ndarray]:
    """Yield one int64 token-ID array per document, cycling forever if repeat=True.

    hf_tokenizer   — a HuggingFace `tokenizers.Tokenizer` instance (preferred)
    fallback_tokenizer — a pydnn BPETokenizer (used when hf_tokenizer is None)
    """
    idx = 0
    n = len(raw_texts)
    while True:
        text = raw_texts[idx % n]
        idx += 1
        if hf_tokenizer is not None:
            enc = hf_tokenizer.encode(text)
            yield np.array(enc.ids, dtype=np.int64)
        elif fallback_tokenizer is not None:
            ids = fallback_tokenizer.encode(text, add_special_tokens=True)
            yield np.array(ids, dtype=np.int64)
        else:
            raise ValueError('Provide either hf_tokenizer or fallback_tokenizer')
        if not repeat and idx >= n:
            break


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def smooth(arr: List[float], window: int) -> np.ndarray:
    """Rolling average with reflection padding to preserve length."""
    if window <= 1 or len(arr) < window:
        return np.asarray(arr, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')
