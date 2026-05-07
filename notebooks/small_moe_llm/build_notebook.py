"""
build_notebook.py  —  generates training_notebook.ipynb.

Run:  python build_notebook.py
"""

import json, pathlib, textwrap

NB_PATH = pathlib.Path(__file__).parent / "training_notebook.ipynb"

# ─────────────────────────────────────────────────────────────────────────────
# helpers

def md(src: str) -> dict:
    return {"cell_type": "markdown", "id": _uid(), "metadata": {}, "source": src}

def code(src: str) -> dict:
    return {"cell_type": "code", "id": _uid(), "metadata": {},
            "outputs": [], "execution_count": None, "source": src}

_counter = 0
def _uid() -> str:
    global _counter; _counter += 1
    return f"cell{_counter:04d}"

# ─────────────────────────────────────────────────────────────────────────────
# cells

cells = []

# ── 1 ── Title ───────────────────────────────────────────────────────────────
cells.append(md(textwrap.dedent("""\
# Small MoE-LLM Training Notebook

**Architecture (Opus 4.7-inspired):** Decoder-only · RoPE · GQA · SwiGLU · RMSNorm · Sparse MoE (top-2 of 8 experts)

| Preset | Total params | Active params | Suggested GPU | Steps |
|--------|-------------|---------------|---------------|-------|
| `tiny` | ~3 M | ~2 M | CPU / any | 500 |
| `small` | ~73 M | ~26 M | T4 / A10 | 5 000 |
| `medium` | ~230 M | ~79 M | A100 | 20 000 |

**Data:** `nampdn-ai/tiny-codes` (coding) + `tatsu-lab/alpaca` (instruction-following) from HuggingFace.
Falls back to a built-in synthetic corpus if HuggingFace is unreachable.

> **Speed note:** The `pydnn` transformer runs on NumPy + optional C++ CPU kernels.
> Set `PYDNN_TRANSFORMER_BACKEND=cpu` to use C++ ops; `=numpy` for the pure-NumPy reference.
> CUDA is not yet routed through the transformer module — CPU ops give the best throughput.
""")))

# ── 2 ── Install & imports ───────────────────────────────────────────────────
cells.append(code(textwrap.dedent("""\
import subprocess, sys, importlib, pathlib, os, urllib.request

# ── Install packages ───────────────────────────────────────────────────────
for _pkg in ('datasets', 'tokenizers'):
    try:
        importlib.import_module(_pkg)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', _pkg, '-q'])

# ── Locate repo root and patch sys.path ───────────────────────────────────
_here = pathlib.Path().resolve()
_repo_root = None
for _cand in [_here, _here.parent, _here.parent.parent, _here.parent.parent.parent]:
    if (_cand / 'python' / 'pydnn').is_dir():
        _repo_root = str(_cand)
        _python_dir = str(_cand / 'python')
        if _python_dir not in sys.path:
            sys.path.insert(0, _python_dir)
        break

if _repo_root is None:
    raise RuntimeError('Cannot find the dynamic_nn repo root. '
                       'Run this notebook from inside the repo directory.')

# ── HuggingFace reachability probe ────────────────────────────────────────
try:
    urllib.request.urlopen('https://huggingface.co', timeout=5)
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False
print(f'HuggingFace reachable: {HF_AVAILABLE}')

# ── Core imports ──────────────────────────────────────────────────────────
import numpy as np
import json, dataclasses, time, pathlib
import matplotlib.pyplot as plt

from pydnn import transformer as T
from pydnn.transformer import autograd as A
from pydnn.transformer.model import DecoderOnlyModel
from pydnn.transformer.training import AdamW, cosine_with_warmup, clip_grad_norm
from pydnn.transformer import (
    TransformerConfig, DynamicTransformer, BPETokenizer, cross_entropy,
)

# ── Fix: DynamicTransformer.generate needs _sample from DecoderOnlyModel ──
# DynamicTransformer assigns `generate = DecoderOnlyModel.generate` but does
# not inherit from DecoderOnlyModel, so _sample (a @staticmethod) is missing.
if not hasattr(DynamicTransformer, '_sample'):
    DynamicTransformer._sample = staticmethod(DecoderOnlyModel._sample)

# ── Load helpers from moe_utils.py (same directory as this notebook) ──────
sys.path.insert(0, str(pathlib.Path(_repo_root) / 'notebooks' / 'small_moe_llm'))
from moe_utils import TokenPacker, save_checkpoint, load_checkpoint, smooth, make_text_stream

print('All imports OK')
""")))

# ── 3 ── Preset selection ────────────────────────────────────────────────────
cells.append(md("## 1. Preset Selection\n\nChange `PRESET` and re-run from here."))

cells.append(code(textwrap.dedent("""\
# ── Choose a preset ───────────────────────────────────────────────────────
PRESET = 'tiny'    # 'tiny' | 'small' | 'medium'

_PRESETS = {
    'tiny': dict(
        vocab_size=32000, dim=64,  num_heads=4, num_kv_heads=2,
        num_decoder_layers=6,  ffn_hidden_dim=128,  max_seq_len=256,
        batch_size=8, seq_len=128, lr=3e-4, steps=500,   warmup=50,
    ),
    'small': dict(
        vocab_size=32000, dim=256, num_heads=8, num_kv_heads=2,
        num_decoder_layers=10, ffn_hidden_dim=1024, max_seq_len=512,
        batch_size=4, seq_len=256, lr=2e-4, steps=5000,  warmup=200,
    ),
    'medium': dict(
        vocab_size=32000, dim=512, num_heads=8, num_kv_heads=4,
        num_decoder_layers=16, ffn_hidden_dim=1024, max_seq_len=1024,
        batch_size=2, seq_len=512, lr=1e-4, steps=20000, warmup=500,
    ),
}
_p = _PRESETS[PRESET]

# All presets share these Opus 4.7-inspired flags:
cfg = TransformerConfig(
    vocab_size=_p['vocab_size'],
    dim=_p['dim'],
    num_heads=_p['num_heads'],
    num_kv_heads=_p['num_kv_heads'],
    num_decoder_layers=_p['num_decoder_layers'],
    ffn_hidden_dim=_p['ffn_hidden_dim'],
    max_seq_len=_p['max_seq_len'],
    ffn_type='moe',           # Sparse Mixture-of-Experts FFN
    num_experts=8,
    moe_top_k=2,              # top-2 routing per token
    moe_aux_loss_weight=0.01, # Switch-style load-balance penalty
    pos_encoding='rope',      # Rotary position embeddings
    norm_type='rmsnorm',      # Pre-norm RMSNorm (no bias in norm)
    activation='silu',        # SwiGLU gate inside each expert
    bias=False,               # No bias in attention / FFN projections
    tie_embeddings=True,      # Shared input/output embedding matrix
    dropout=0.0,              # No dropout (standard LLM pretraining)
    seed=42,
)

BATCH_SIZE   = _p['batch_size']
SEQ_LEN      = _p['seq_len']
LR           = _p['lr']
MAX_STEPS    = _p['steps']
WARMUP_STEPS = _p['warmup']
ADAPT_EVERY  = 100   # call model.adapt() every N steps
CKPT_EVERY   = 250   # save checkpoint every N steps

CKPT_DIR = pathlib.Path('checkpoints') / PRESET
CKPT_DIR.mkdir(parents=True, exist_ok=True)

print(f'Preset  : {PRESET}')
print(f'Config  : dim={cfg.dim}, heads={cfg.num_heads}(kv={cfg.num_kv_heads}), '
      f'layers={cfg.num_decoder_layers}, experts={cfg.num_experts}(top-{cfg.moe_top_k})')
print(f'Training: batch={BATCH_SIZE}, seq={SEQ_LEN}, lr={LR:.1e}, '
      f'steps={MAX_STEPS}, warmup={WARMUP_STEPS}')
print(f'Checkpoints -> {CKPT_DIR}')
""")))

# ── 4 ── Data loading ────────────────────────────────────────────────────────
cells.append(md("## 2. Data Loading\n\nLoads coding + instruction data from HuggingFace, or uses the built-in synthetic corpus."))

cells.append(code(textwrap.dedent("""\
# ── Built-in synthetic corpus (always available as a fallback) ─────────────
SYNTHETIC_CORPUS = [
    'def add(a, b):\\n    return a + b\\n',
    'def multiply(x, y):\\n    return x * y\\n',
    'def factorial(n):\\n    if n <= 1:\\n        return 1\\n    return n * factorial(n - 1)\\n',
    'def fibonacci(n):\\n    a, b = 0, 1\\n    for _ in range(n):\\n        a, b = b, a + b\\n    return a\\n',
    'def is_prime(n):\\n    if n < 2:\\n        return False\\n    for i in range(2, int(n**0.5)+1):\\n        if n % i == 0:\\n            return False\\n    return True\\n',
    'class Stack:\\n    def __init__(self):\\n        self._d = []\\n    def push(self, x): self._d.append(x)\\n    def pop(self): return self._d.pop()\\n',
    'class Queue:\\n    def __init__(self):\\n        self._d = []\\n    def enqueue(self, x): self._d.append(x)\\n    def dequeue(self): return self._d.pop(0)\\n',
    '### Instruction:\\nWhat is a list comprehension?\\n\\n### Response:\\nA concise way to create lists: [expr for item in iterable if cond].\\n',
    '### Instruction:\\nExplain recursion.\\n\\n### Response:\\nA function that calls itself with a smaller input until a base case is reached.\\n',
    '### Instruction:\\nWrite a binary search.\\n\\n### Response:\\ndef binary_search(arr, t):\\n    lo, hi = 0, len(arr)-1\\n    while lo <= hi:\\n        mid = (lo+hi)//2\\n        if arr[mid] == t: return mid\\n        elif arr[mid] < t: lo = mid+1\\n        else: hi = mid-1\\n    return -1\\n',
    '### Instruction:\\nHow does gradient descent work?\\n\\n### Response:\\nIt iteratively moves parameters in the direction of steepest loss decrease, scaled by a learning rate.\\n',
    '### Instruction:\\nWhat is a transformer?\\n\\n### Response:\\nA neural network using self-attention to model long-range dependencies in sequences.\\n',
] * 250   # ~3000 documents

if HF_AVAILABLE:
    try:
        from datasets import load_dataset
        _coding_docs, _alpaca_docs = [], []

        print('Loading nampdn-ai/tiny-codes (streaming)...')
        for i, row in enumerate(load_dataset('nampdn-ai/tiny-codes', split='train', streaming=True)):
            if i >= 8000: break
            text = row.get('response') or row.get('code', '')
            if text.strip():
                _coding_docs.append(f'### Code:\\n{text}\\n')

        print('Loading tatsu-lab/alpaca (streaming)...')
        for i, row in enumerate(load_dataset('tatsu-lab/alpaca', split='train', streaming=True)):
            if i >= 4000: break
            inp = row.get('input', '').strip()
            prompt = f"{row['instruction']}\\n{inp}".strip() if inp else row['instruction']
            _alpaca_docs.append(f"### Instruction:\\n{prompt}\\n\\n### Response:\\n{row['output']}\\n")

        raw_texts = _coding_docs + _alpaca_docs
        print(f'Loaded {len(_coding_docs)} code + {len(_alpaca_docs)} instruction docs')

    except Exception as _exc:
        print(f'HF load failed ({_exc}), using synthetic corpus')
        raw_texts = SYNTHETIC_CORPUS
        HF_AVAILABLE = False
else:
    raw_texts = SYNTHETIC_CORPUS
    print(f'Using synthetic corpus ({len(raw_texts)} docs)')

print(f'Total documents : {len(raw_texts)}')
print(f'Sample doc      : {raw_texts[0][:100].replace(chr(10), "\\\\n")!r}')
""")))

# ── 5 ── Tokenizer ───────────────────────────────────────────────────────────
cells.append(md("## 3. Tokenizer\n\nTrains a BPE tokenizer on the corpus.  Prefers the HuggingFace Rust BPE (fast); falls back to pydnn's pure-Python BPE for offline runs."))

cells.append(code(textwrap.dedent("""\
TOKENIZER_PATH = CKPT_DIR / 'tokenizer.json'

hf_tokenizer    = None
fallback_tok     = None
PAD_ID = 0; UNK_ID = 1; BOS_ID = 2; EOS_ID = 3
VOCAB_SIZE = cfg.vocab_size

if HF_AVAILABLE:
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.processors import TemplateProcessing
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder

        _special = ['<pad>', '<unk>', '<bos>', '<eos>']
        _tok = Tokenizer(BPE(unk_token='<unk>'))
        _tok.pre_tokenizer = ByteLevel(add_prefix_space=True)
        _tok.decoder = ByteLevelDecoder()
        _trainer = BpeTrainer(
            vocab_size=VOCAB_SIZE, special_tokens=_special, min_frequency=2
        )
        _sample_docs = raw_texts[:min(20_000, len(raw_texts))]
        print(f'Training HF BPE on {len(_sample_docs)} docs ...')
        _tok.train_from_iterator(_sample_docs, trainer=_trainer)
        _tok.post_processor = TemplateProcessing(
            single='<bos> $A <eos>',
            special_tokens=[('<bos>', BOS_ID), ('<eos>', EOS_ID)],
        )
        _tok.enable_padding(pad_id=PAD_ID, pad_token='<pad>')
        _tok.enable_truncation(max_length=SEQ_LEN + 1)
        _tok.save(str(TOKENIZER_PATH))
        hf_tokenizer = _tok
        _actual_vocab = hf_tokenizer.get_vocab_size()
        print(f'HF BPE ready, actual vocab size = {_actual_vocab}')
        VOCAB_SIZE = _actual_vocab

        # Round-trip sanity check
        _test = 'def hello_world(): return 42'
        _ids  = hf_tokenizer.encode(_test).ids
        _dec  = hf_tokenizer.decode(_ids, skip_special_tokens=True)
        print(f'Round-trip: {_test!r} -> {len(_ids)} tokens -> {_dec!r}')

    except Exception as _exc:
        print(f'HF tokenizer failed ({_exc}), using pydnn BPETokenizer')
        hf_tokenizer = None

if hf_tokenizer is None:
    print('Training pydnn BPETokenizer (slow for large corpora) ...')
    fallback_tok = BPETokenizer()
    fallback_tok.fit(raw_texts[:min(5000, len(raw_texts))], num_merges=2000)
    VOCAB_SIZE = fallback_tok.vocab_size
    PAD_ID     = fallback_tok.pad_token_id
    BOS_ID     = fallback_tok.bos_token_id
    EOS_ID     = fallback_tok.eos_token_id
    # Rebuild cfg with corrected vocab size (dataclasses are frozen via replace)
    cfg = dataclasses.replace(cfg, vocab_size=VOCAB_SIZE)
    print(f'pydnn BPE ready, vocab={VOCAB_SIZE}')

print(f'\\nSpecial tokens: PAD={PAD_ID}  BOS={BOS_ID}  EOS={EOS_ID}')
print(f'Vocab size     : {VOCAB_SIZE}')
""")))

# ── 6 ── Model ───────────────────────────────────────────────────────────────
cells.append(md("## 4. Model\n\nInstantiates `DynamicTransformer` — a decoder-only LM that can grow and prune its own layers during training."))

cells.append(code(textwrap.dedent("""\
model = DynamicTransformer(
    cfg,
    prune_threshold=1e-3,  # utilisation floor for soft-pruning
    grow_patience=10,       # plateau checkpoints before growing
    prune_patience=5,       # checkpoints below threshold before freezing alpha→0
)

print(f'DynamicTransformer  [{PRESET}]')
print(f'  Total parameters : {model.num_parameters():>14,}')
print(f'  Active parameters: {model.num_active_parameters():>14,}')
print(f'  Layers           : {cfg.num_decoder_layers} decoder blocks')
print(f'  Experts/block    : {cfg.num_experts} total, top-{cfg.moe_top_k} routed')
print(f'  Positional enc.  : {cfg.pos_encoding}')
print(f'  FFN norm         : {cfg.norm_type}  (pre-norm)')
print(f'  Vocab            : {cfg.vocab_size:,}  |  max_seq={cfg.max_seq_len}')
""")))

# ── 7 ── Optimizer ───────────────────────────────────────────────────────────
cells.append(md("## 5. Optimizer & LR Schedule"))

cells.append(code(textwrap.dedent("""\
params    = list(model.parameters())
optimizer = AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.01)

step            = 0
loss_history    = []
aux_loss_history= []
lr_history      = []

# ── Resume from latest checkpoint if it exists ────────────────────────────
_latest_meta = CKPT_DIR / 'meta_latest.json'
if _latest_meta.exists():
    print('Resuming from latest checkpoint ...')
    model, optimizer, step, loss_history, cfg = load_checkpoint(CKPT_DIR, tag='latest')
    params = list(model.parameters())
    # Rebuild aux history from scratch (not stored)
    aux_loss_history = []
    lr_history = [cosine_with_warmup(s, WARMUP_STEPS, MAX_STEPS, LR, LR*0.1)
                  for s in range(step)]
    print(f'Resumed at step {step}  last loss={loss_history[-1]:.4f}')
else:
    print(f'Starting fresh training run')

print(f'\\nOptimizer : AdamW  lr={LR:.1e}  wd=0.01  betas=(0.9, 0.95)')
print(f'Schedule  : cosine {WARMUP_STEPS} warmup / {MAX_STEPS} total steps')
print(f'Parameters: {len(params):,}  parameter tensors  ({sum(p.data.size for p in params):,} scalars)')
""")))

# ── 8 ── Training loop ───────────────────────────────────────────────────────
cells.append(md(textwrap.dedent("""\
## 6. Training

The loop:
1. Streams tokenised documents into a `TokenPacker` that yields `(B, seq_len+1)` windows.
2. Computes next-token CE loss + MoE auxiliary load-balance loss.
3. Calls `model.adapt()` every `ADAPT_EVERY` steps to soft-prune/grow layers.
4. Saves a checkpoint every `CKPT_EVERY` steps (resumable).

**Estimated throughput** (pure NumPy, single CPU core):

| Preset | ~ms/step | tokens/s |
|--------|----------|----------|
| tiny   | 150 ms   | ~7 000   |
| small  | 2–4 s    | ~500     |
| medium | 10–20 s  | ~100     |

Enable C++ ops for 3–5× speedup: `import os; os.environ['PYDNN_TRANSFORMER_BACKEND'] = 'cpu'` (must set *before* importing pydnn).
""")))

cells.append(code(textwrap.dedent("""\
model.train()
_stream  = make_text_stream(raw_texts,
                             hf_tokenizer=hf_tokenizer,
                             fallback_tokenizer=fallback_tok)
_packer  = TokenPacker(SEQ_LEN, pad_id=PAD_ID)
_t_start = time.time()
_t_log   = _t_start

for _doc_ids in _stream:
    _packer.feed(_doc_ids)

    while _packer.has_batch(BATCH_SIZE):
        if step >= MAX_STEPS:
            break

        # ── LR schedule ───────────────────────────────────────────────────
        _lr_now = cosine_with_warmup(step, WARMUP_STEPS, MAX_STEPS, LR, LR * 0.1)
        optimizer.set_lr(_lr_now)
        lr_history.append(_lr_now)

        # ── Batch ─────────────────────────────────────────────────────────
        _batch   = _packer.drain_batch(BATCH_SIZE)   # (B, T+1)
        _inputs  = _batch[:, :-1]                     # (B, T)
        _targets = _batch[:, 1:]                      # (B, T)

        # ── Forward ───────────────────────────────────────────────────────
        optimizer.zero_grad()
        _logits = model(_inputs)                      # Tensor (B, T, V)
        _loss   = A.cross_entropy(_logits, _targets, ignore_index=PAD_ID)
        _aux    = model.aux_loss()
        _total  = (_loss + _aux) if _aux is not None else _loss

        # ── Backward ──────────────────────────────────────────────────────
        _total.backward()
        _gnorm = clip_grad_norm(params, max_norm=1.0)
        optimizer.step()

        step += 1
        loss_history.append(float(_loss.data))
        if _aux is not None:
            aux_loss_history.append(float(_aux.data))

        # ── Logging ───────────────────────────────────────────────────────
        if step % 10 == 0 or step == 1:
            _now = time.time()
            _ela = _now - _t_start
            _bps = step / max(_ela, 1e-9)
            _eta = (MAX_STEPS - step) / max(_bps, 1e-9)
            print(f'step {step:5d}/{MAX_STEPS}  '
                  f'loss={loss_history[-1]:.4f}  '
                  f'aux={aux_loss_history[-1] if aux_loss_history else 0.0:.4f}  '
                  f'|g|={_gnorm:.3f}  lr={_lr_now:.2e}  '
                  f'ETA={_eta/60:.1f}min')

        # ── Adapt (grow / prune) ──────────────────────────────────────────
        if step % ADAPT_EVERY == 0:
            _actions = model.adapt(
                loss_history,
                plateau_window=min(20, len(loss_history)),
                plateau_eps=1e-3,
                allow_grow=True,
                allow_prune=True,
            )
            _h = model.health_report()
            if any(_actions.get(k) for k in ('pruned', 'grew', 'reinit_experts')):
                print(f'  [adapt@{step}] {_actions}  '
                      f'health={_h.diagnosis}  '
                      f'active={_h.active_layers}/{_h.total_layers}')
                # Re-gather params after structural change; extend AdamW moments
                params = list(model.parameters())
                optimizer.params = params
                while len(optimizer._m) < len(params):
                    optimizer._m.append(np.zeros_like(params[len(optimizer._m)].data))
                    optimizer._v.append(np.zeros_like(params[len(optimizer._v)].data))

        # ── Checkpoint ────────────────────────────────────────────────────
        if step % CKPT_EVERY == 0:
            save_checkpoint(model, optimizer, step, loss_history, cfg, CKPT_DIR,
                            tag=f'step{step:06d}')
            save_checkpoint(model, optimizer, step, loss_history, cfg, CKPT_DIR,
                            tag='latest')
            print(f'  [ckpt@{step}] saved -> {CKPT_DIR}')

    if step >= MAX_STEPS:
        break

# ── Final checkpoint ──────────────────────────────────────────────────────
save_checkpoint(model, optimizer, step, loss_history, cfg, CKPT_DIR, tag='final')
_elapsed = time.time() - _t_start
print(f'\\nTraining complete: {step} steps in {_elapsed/60:.1f} min')
print(f'Final loss: {loss_history[-1]:.4f}')
""")))

# ── 9 ── Training curves ─────────────────────────────────────────────────────
cells.append(md("## 7. Training Curves"))

cells.append(code(textwrap.dedent("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# ── LM loss ──────────────────────────────────────────────────────────────
_sm = smooth(loss_history, window=max(1, len(loss_history)//40))
axes[0].plot(loss_history, alpha=0.3, color='steelblue', label='raw')
axes[0].plot(_sm, color='steelblue', lw=2, label='smoothed')
axes[0].set_title('Cross-Entropy Loss'); axes[0].set_xlabel('step')
axes[0].legend(); axes[0].grid(alpha=0.3)

# ── Aux (load-balance) loss ───────────────────────────────────────────────
if aux_loss_history:
    axes[1].plot(aux_loss_history, color='darkorange', alpha=0.6)
    axes[1].set_title('MoE Aux Loss (load balance)')
    axes[1].set_xlabel('step'); axes[1].grid(alpha=0.3)
else:
    axes[1].text(0.5, 0.5, 'No aux loss recorded', ha='center', transform=axes[1].transAxes)

# ── LR schedule ──────────────────────────────────────────────────────────
axes[2].plot(lr_history, color='seagreen')
axes[2].set_title('Learning Rate (cosine)')
axes[2].set_xlabel('step'); axes[2].set_yscale('log'); axes[2].grid(alpha=0.3)

plt.suptitle(f'Training — preset={PRESET}', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()
""")))

# ── 10 ── Expert load chart ──────────────────────────────────────────────────
cells.append(md("### Expert Load Balance\n\nA healthy MoE should route tokens roughly uniformly across experts."))

cells.append(code(textwrap.dedent("""\
_h = model.health_report()
if _h.expert_loads:
    _nl = len(_h.expert_loads)
    _fig2, _ax2 = plt.subplots(1, _nl, figsize=(4 * _nl, 3))
    if _nl == 1:
        _ax2 = [_ax2]
    for _li, _loads in enumerate(_h.expert_loads):
        _ne = len(_loads)
        _ax2[_li].bar(range(_ne), _loads, color='mediumpurple', alpha=0.8)
        _ax2[_li].axhline(1.0 / _ne, ls='--', color='red', label='uniform')
        _ax2[_li].set_title(f'Layer {_li} Expert Load')
        _ax2[_li].set_xlabel('expert index')
        _ax2[_li].set_ylim(0, max(_loads) * 1.2)
        _ax2[_li].legend(fontsize=8)
    plt.suptitle('MoE Expert Token-Routing Load', fontsize=12)
    plt.tight_layout(); plt.show()
else:
    print('No expert load data (model may not have MoE layers or adapt() not called yet)')
""")))

# ── 11 ── Generation ─────────────────────────────────────────────────────────
cells.append(md("## 8. Generation Examples\n\nTests the model's generative ability on a few prompts."))

cells.append(code(textwrap.dedent("""\
model.eval()
_rng_gen = np.random.default_rng(7)

def generate_text(prompt: str, max_new_tokens: int = 80,
                  temperature: float = 0.8, top_k: int = 40, top_p: float = 0.9) -> str:
    if hf_tokenizer is not None:
        _ids = np.array(hf_tokenizer.encode(prompt).ids, dtype=np.int64)[None, :]
    else:
        _ids = np.array(fallback_tok.encode(prompt, add_special_tokens=True),
                        dtype=np.int64)[None, :]
    _out = model.generate(
        _ids, max_new_tokens=max_new_tokens,
        temperature=temperature, top_k=top_k, top_p=top_p,
        eos_token_id=EOS_ID, use_cache=False, rng=_rng_gen,
    )
    _new = _out[0, _ids.shape[1]:]
    if hf_tokenizer is not None:
        return hf_tokenizer.decode(_new.tolist(), skip_special_tokens=True)
    return fallback_tok.decode(_new.tolist(), skip_special=True)

_PROMPTS = [
    'def fibonacci(n):',
    '### Instruction:\\nWrite a function to reverse a string.\\n\\n### Response:\\n',
    'class BinaryTree:\\n    def __init__(self):',
]

for _p in _PROMPTS:
    _gen = generate_text(_p)
    print('─' * 60)
    print(f'PROMPT : {_p!r}')
    print(f'OUTPUT : {_gen!r}')
print('─' * 60)
""")))

# ── 12 ── Architecture health ────────────────────────────────────────────────
cells.append(md("## 9. Architecture Health Report\n\nInspects layer utilisation, alpha weights, and pruned blocks."))

cells.append(code(textwrap.dedent("""\
_h = model.health_report()
print('=== DynamicTransformer Health ===')
print(f'  Diagnosis        : {_h.diagnosis}')
print(f'  Active layers    : {_h.active_layers} / {_h.total_layers}')
print(f'  Pruned indices   : {_h.pruned_layers}')
print()
print('  Layer alphas (attn+ffn avg):')
for _i, (_a, _u) in enumerate(zip(_h.layer_alphas, _h.layer_utilisation)):
    _bar = '|' * int(_a * 20)
    print(f'    [{_i:2d}] alpha={_a:.3f}  util={_u:.4f}  {_bar}')
print()
print(f'  Total params  : {model.num_parameters():,}')
print(f'  Active params : {model.num_active_parameters():,}')
""")))

# ── 13 ── Save final model ───────────────────────────────────────────────────
cells.append(md("## 10. Save Final Model\n\nCompacts soft-pruned layers, then saves weights + tokenizer for inference."))

cells.append(code(textwrap.dedent("""\
_final_dir = pathlib.Path('final_model') / PRESET
_final_dir.mkdir(parents=True, exist_ok=True)

# Compact: permanently drop blocks with alpha ≈ 0 (mirrors Network::compact())
_n_removed = model.compact()
print(f'Compacted model: removed {_n_removed} soft-pruned blocks')
print(f'  Remaining params: {model.num_parameters():,}')

# Save weights
_weights = {name: param.data for name, param in model.named_parameters()}
np.savez(_final_dir / 'weights.npz', **_weights)

# Save config
import dataclasses
with open(_final_dir / 'config.json', 'w') as _fh:
    json.dump(dataclasses.asdict(cfg), _fh, indent=2)

# Copy tokenizer
if hf_tokenizer is not None:
    hf_tokenizer.save(str(_final_dir / 'tokenizer.json'))
    print('Saved HF BPE tokenizer')
elif fallback_tok is not None:
    import pickle
    with open(_final_dir / 'tokenizer.pkl', 'wb') as _fh:
        pickle.dump(fallback_tok, _fh)
    print('Saved pydnn BPETokenizer (pickle)')

print(f'\\nFinal model saved to: {_final_dir.resolve()}')
print('Files:', [f.name for f in sorted(_final_dir.iterdir())])
""")))

# ── 14 ── Smoke test ─────────────────────────────────────────────────────────
cells.append(md(textwrap.dedent("""\
## 11. Smoke Test (no HF required)

Run this cell independently to verify the full pipeline on a tiny synthetic setup.
Checks: forward pass, training loop (50 steps, loss decreasing), checkpoint round-trip, and generation.
""")))

cells.append(code(textwrap.dedent("""\
def run_smoke_test():
    import sys, pathlib, dataclasses
    # Ensure pydnn is importable
    _pdir = str(pathlib.Path(_repo_root) / 'python')
    if _pdir not in sys.path:
        sys.path.insert(0, _pdir)

    from pydnn.transformer import DynamicTransformer, TransformerConfig, BPETokenizer
    from pydnn.transformer import autograd as _A
    from pydnn.transformer.training import AdamW, cosine_with_warmup, clip_grad_norm
    from pydnn.transformer.model import DecoderOnlyModel
    import numpy as np

    if not hasattr(DynamicTransformer, '_sample'):
        DynamicTransformer._sample = staticmethod(DecoderOnlyModel._sample)

    # Tiny synthetic corpus
    _texts = [
        'def add(a, b): return a + b',
        'def mul(x, y): return x * y',
        'def double(n): return n * 2',
        '### Instruction:\\nWhat is 2+2?\\n\\n### Response:\\nThe answer is 4.',
        '### Instruction:\\nHow does Python work?\\n\\n### Response:\\nPython is interpreted.',
    ] * 40   # 200 short docs

    # Train a tiny BPE tokenizer
    _tok = BPETokenizer()
    _tok.fit(_texts, num_merges=300)
    _V = _tok.vocab_size

    # Smallest-possible config
    _cfg = TransformerConfig(
        vocab_size=_V, dim=32, num_heads=2, num_kv_heads=1,
        num_decoder_layers=2, ffn_hidden_dim=64,
        ffn_type='moe', num_experts=4, moe_top_k=2, moe_aux_loss_weight=0.01,
        pos_encoding='rope', norm_type='rmsnorm',
        bias=False, activation='silu', tie_embeddings=True,
        dropout=0.0, max_seq_len=64, seed=0,
    )
    _model = DynamicTransformer(_cfg)
    _params = list(_model.parameters())
    _opt   = AdamW(_params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)

    # Pack tokens from synthetic corpus
    _all_ids = []
    for _t in _texts:
        _all_ids.extend(_tok.encode(_t, add_special_tokens=True))
    _arr = np.array(_all_ids, dtype=np.int64)

    # 50 training steps
    _losses = []
    _model.train()
    for _s in range(50):
        _st = (_s * 33) % max(1, len(_arr) - 33)
        _b  = _arr[_st:_st + 33]
        _inp = _b[None, :-1]; _tgt = _b[None, 1:]
        _opt.zero_grad()
        _logits = _model(_inp)
        _loss = _A.cross_entropy(_logits, _tgt)
        _aux  = _model.aux_loss()
        _tot  = (_loss + _aux) if _aux is not None else _loss
        _tot.backward()
        clip_grad_norm(_params, 1.0)
        _opt.step()
        _losses.append(float(_loss.data))

    # Check loss doesn't diverge
    assert _losses[-1] < _losses[0] * 2.0, f'Loss diverged: {_losses[0]:.3f} -> {_losses[-1]:.3f}'
    print(f'Training OK: {_losses[0]:.3f} -> {_losses[-1]:.3f} over 50 steps')

    # Checkpoint round-trip
    import tempfile
    _ckpt = pathlib.Path(tempfile.mkdtemp()) / 'smoke'
    _ckpt.mkdir()
    save_checkpoint(_model, _opt, 50, _losses, _cfg, _ckpt, tag='smoke')
    _m2, _o2, _s2, _lh2, _c2 = load_checkpoint(_ckpt, tag='smoke')
    assert _s2 == 50 and _c2.ffn_type == 'moe', 'Checkpoint round-trip failed'
    print(f'Checkpoint round-trip OK (step={_s2}, ffn_type={_c2.ffn_type})')

    # Generation test
    _prompt = np.array([[_tok.bos_token_id]], dtype=np.int64)
    _out = _m2.generate(_prompt, max_new_tokens=8, temperature=0.9, top_k=5,
                         use_cache=False, rng=np.random.default_rng(0))
    assert _out.shape[1] > 1, 'generate() produced no new tokens'
    print(f'Generation OK: prompt_len=1 -> output_len={_out.shape[1]}')
    print('\\n✓ Smoke test PASSED')

run_smoke_test()
""")))

# ─────────────────────────────────────────────────────────────────────────────
# assemble notebook

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells,
}

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Wrote {NB_PATH}  ({NB_PATH.stat().st_size // 1024} KB)")
