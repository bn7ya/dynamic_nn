"""Parity tests for the C++ transformer backend.

For each op exposed by ``pydnn.transformer._backend`` we compare the
``numpy`` reference against the ``cpu`` (and, when present, ``cuda``)
backend. We then run the ``DynamicTransformer`` grow/prune cycle under
the C++ backend to confirm the dynamic-NN integration still works —
the kernels back the existing dynamic transformer, they don't replace
it.
"""

from __future__ import annotations

import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np

from pydnn.transformer import _backend as B
from pydnn.transformer import autograd as A
from pydnn.transformer.layers import LayerNorm, RMSNorm


RTOL = 1e-4
ATOL = 1e-5


def _close(a, b, name):
    a = np.asarray(a)
    b = np.asarray(b)
    if not np.allclose(a, b, rtol=RTOL, atol=ATOL):
        diff = np.abs(a - b).max()
        raise AssertionError(f"{name}: max abs diff {diff:.3g}")


def _with_backend(name, fn):
    prev = B.current_backend()
    B.set_backend(name)
    try:
        return fn()
    finally:
        B.set_backend(prev)


def test_matmul():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
    b = rng.standard_normal((2, 3, 5, 6)).astype(np.float32)
    c_ref = _with_backend("numpy", lambda: B.matmul(a, b))
    c_cpp = _with_backend("cpu", lambda: B.matmul(a, b))
    _close(c_ref, c_cpp, "matmul fwd")

    # Broadcasted leading dim
    a2 = rng.standard_normal((1, 3, 4, 5)).astype(np.float32)
    b2 = rng.standard_normal((4, 1, 5, 6)).astype(np.float32)
    c_ref = _with_backend("numpy", lambda: B.matmul(a2, b2))
    c_cpp = _with_backend("cpu", lambda: B.matmul(a2, b2))
    _close(c_ref, c_cpp, "matmul broadcast")

    # transpose flags
    c_ref = _with_backend("numpy",
                          lambda: B.matmul(a, b.swapaxes(-1, -2), transpose_b=True))
    c_cpp = _with_backend("cpu",
                          lambda: B.matmul(a, b.swapaxes(-1, -2), transpose_b=True))
    _close(c_ref, c_cpp, "matmul transpose_b")


def test_softmax_and_backward():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 7, 11)).astype(np.float32)
    y_ref = _with_backend("numpy", lambda: B.softmax(x))
    y_cpp = _with_backend("cpu", lambda: B.softmax(x))
    _close(y_ref, y_cpp, "softmax fwd")
    dy = rng.standard_normal(y_cpp.shape).astype(np.float32)
    dx_ref = _with_backend("numpy", lambda: B.softmax_backward(y_ref, dy))
    dx_cpp = _with_backend("cpu", lambda: B.softmax_backward(y_cpp, dy))
    _close(dx_ref, dx_cpp, "softmax bwd")


def test_gelu_silu():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 8)).astype(np.float32)
    dy = rng.standard_normal((4, 8)).astype(np.float32)
    for name, fwd, bwd in [("gelu", B.gelu, B.gelu_backward),
                            ("silu", B.silu, B.silu_backward)]:
        y_ref = _with_backend("numpy", lambda: fwd(x))
        y_cpp = _with_backend("cpu", lambda: fwd(x))
        _close(y_ref, y_cpp, f"{name} fwd")
        dx_ref = _with_backend("numpy", lambda: bwd(x, dy))
        dx_cpp = _with_backend("cpu", lambda: bwd(x, dy))
        _close(dx_ref, dx_cpp, f"{name} bwd")


def test_layernorm():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 5, 8)).astype(np.float32)
    g = rng.standard_normal((8,)).astype(np.float32)
    b = rng.standard_normal((8,)).astype(np.float32)
    eps = 1e-5
    y_ref, m_ref, isd_ref = _with_backend("numpy",
        lambda: B.layernorm_forward(x, g, b, eps))
    y_cpp, m_cpp, isd_cpp = _with_backend("cpu",
        lambda: B.layernorm_forward(x, g, b, eps))
    _close(y_ref, y_cpp, "ln fwd y")
    _close(m_ref, m_cpp, "ln fwd mean")
    _close(isd_ref, isd_cpp, "ln fwd inv_std")
    dy = rng.standard_normal(y_cpp.shape).astype(np.float32)
    dx_ref, gw_ref, gb_ref = _with_backend("numpy",
        lambda: B.layernorm_backward(x, g, m_cpp, isd_cpp, dy))
    dx_cpp, gw_cpp, gb_cpp = _with_backend("cpu",
        lambda: B.layernorm_backward(x, g, m_cpp, isd_cpp, dy))
    _close(dx_ref, dx_cpp, "ln bwd dx")
    _close(gw_ref, gw_cpp, "ln bwd dgamma")
    _close(gb_ref, gb_cpp, "ln bwd dbeta")


def test_rmsnorm():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, 5, 8)).astype(np.float32)
    g = rng.standard_normal((8,)).astype(np.float32)
    eps = 1e-6
    y_ref, inv_ref = _with_backend("numpy", lambda: B.rmsnorm_forward(x, g, eps))
    y_cpp, inv_cpp = _with_backend("cpu", lambda: B.rmsnorm_forward(x, g, eps))
    _close(y_ref, y_cpp, "rms fwd y")
    _close(inv_ref, inv_cpp, "rms fwd inv_rms")
    dy = rng.standard_normal(y_cpp.shape).astype(np.float32)
    dx_ref, gw_ref = _with_backend("numpy", lambda: B.rmsnorm_backward(x, g, inv_cpp, dy))
    dx_cpp, gw_cpp = _with_backend("cpu", lambda: B.rmsnorm_backward(x, g, inv_cpp, dy))
    _close(dx_ref, dx_cpp, "rms bwd dx")
    _close(gw_ref, gw_cpp, "rms bwd dgamma")


def test_embedding():
    rng = np.random.default_rng(5)
    weight = rng.standard_normal((20, 6)).astype(np.float32)
    ids = rng.integers(0, 20, size=(3, 7)).astype(np.int64)
    y_ref = _with_backend("numpy", lambda: B.embedding_forward(weight, ids))
    y_cpp = _with_backend("cpu", lambda: B.embedding_forward(weight, ids))
    _close(y_ref, y_cpp, "emb fwd")
    dy = rng.standard_normal(y_cpp.shape).astype(np.float32)
    dw_ref = _with_backend("numpy", lambda: B.embedding_backward(dy, ids, 20, 6))
    dw_cpp = _with_backend("cpu", lambda: B.embedding_backward(dy, ids, 20, 6))
    _close(dw_ref, dw_cpp, "emb bwd")


def test_xent():
    rng = np.random.default_rng(6)
    logits = rng.standard_normal((4, 9, 12)).astype(np.float32)
    targets = rng.integers(0, 12, size=(4, 9)).astype(np.int64)
    targets[0, 0] = -100
    loss_ref, lp_ref, n_ref = _with_backend("numpy",
        lambda: B.xent_forward(logits, targets, -100))
    loss_cpp, lp_cpp, n_cpp = _with_backend("cpu",
        lambda: B.xent_forward(logits, targets, -100))
    assert n_ref == n_cpp
    _close(loss_ref, loss_cpp, "xent loss")
    _close(lp_ref, lp_cpp, "xent log_probs")
    g_ref = _with_backend("numpy",
        lambda: B.xent_backward(lp_cpp, targets, 1.0, n_cpp, -100))
    g_cpp = _with_backend("cpu",
        lambda: B.xent_backward(lp_cpp, targets, 1.0, n_cpp, -100))
    _close(g_ref, g_cpp, "xent dlogits")


def test_autograd_end_to_end_parity():
    """Forward+backward of a small decoder block produces identical grads
    on numpy and cpu backends (up to fp32 noise).

    We build the model once, snapshot its parameters and a fixed input,
    then re-run forward+backward under each backend. Module init uses
    a fresh ``np.random.default_rng()`` per call, so we have to share
    one model — not rebuild it — to get a clean A/B comparison.
    """
    from pydnn import transformer as T

    cfg = T.TransformerConfig(
        vocab_size=32, dim=16, num_heads=2,
        num_decoder_layers=1, ffn_type="swiglu",
        pos_encoding="rope", norm_type="rmsnorm",
        max_seq_len=24, ffn_hidden_dim=24, dropout=0.0,
    )
    model = T.DecoderOnlyModel(cfg)
    rng = np.random.default_rng(123)
    ids = rng.integers(0, cfg.vocab_size, size=(2, 8))
    # Snapshot params so each run starts identical.
    snapshot = [(p, p.data.copy()) for p in model.parameters()]

    def run(backend):
        B.set_backend(backend)
        for p, data in snapshot:
            p.data[...] = data
            p.grad = None
        logits = model(ids)
        loss = T.cross_entropy(logits, ids)
        loss.backward()
        return float(loss.data), [p.grad.copy() for p in model.parameters()
                                  if p.grad is not None]

    loss_np, grads_np = run("numpy")
    loss_cpp, grads_cpp = run("cpu")
    _close(loss_np, loss_cpp, "e2e loss")
    assert len(grads_np) == len(grads_cpp)
    for i, (a, b) in enumerate(zip(grads_np, grads_cpp)):
        _close(a, b, f"e2e param[{i}] grad")


def test_dynamic_transformer_grow_prune():
    """The C++ kernels must back the *existing* DynamicTransformer
    (grow/prune), not replace it. Run a minimal adapt cycle."""
    from pydnn import transformer as T
    from pydnn import DynamicTransformer

    B.set_backend("cpu")
    cfg = T.TransformerConfig(
        vocab_size=24, dim=16, num_heads=2,
        num_decoder_layers=2, ffn_type="swiglu",
        pos_encoding="rope", norm_type="rmsnorm",
        max_seq_len=16, ffn_hidden_dim=24, dropout=0.0,
    )
    model = DynamicTransformer(cfg)
    blocks_initial = len(model.blocks)
    rng = np.random.default_rng(7)
    ids = rng.integers(0, cfg.vocab_size, size=(2, 8))

    # One forward / backward step to make sure ops route through C++
    logits = model(ids)
    loss = T.cross_entropy(logits, ids)
    loss.backward()
    assert np.isfinite(loss.data)

    # Force a topology mutation (grow) by simulating a loss plateau.
    flat_history = [1.0] * 20
    model._plateau_count = model.grow_patience  # short-circuit the patience timer
    actions = model.adapt(loss_history=flat_history)
    assert actions["grew"] is True
    assert len(model.blocks) == blocks_initial + 1

    # Soft-deactivate a block and confirm forward still runs (active mask).
    model.blocks[0].active = False
    logits2 = model(ids)
    assert logits2.shape == logits.shape

    # End-of-training compaction physically drops the inactive block.
    removed = model.compact()
    assert removed == 1
    assert all(blk.active for blk in model.blocks)


if __name__ == "__main__":
    test_matmul();              print("matmul OK")
    test_softmax_and_backward();print("softmax OK")
    test_gelu_silu();           print("gelu/silu OK")
    test_layernorm();           print("layernorm OK")
    test_rmsnorm();             print("rmsnorm OK")
    test_embedding();           print("embedding OK")
    test_xent();                print("xent OK")
    test_autograd_end_to_end_parity(); print("e2e parity OK")
    test_dynamic_transformer_grow_prune(); print("DynamicTransformer OK")
    print("\nAll backend parity tests passed.")
