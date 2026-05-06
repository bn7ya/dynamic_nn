#pragma once

/**
 * Transformer-specific tensor ops on plain float buffers.
 *
 * These are the C++ kernels behind the `pydnn.transformer` Python
 * package. They take raw row-major float pointers + shape information
 * (no Tensor wrapper) so the same signatures work for both
 * `core::Tensor<float>::data()` and `numpy.float32` buffers passed
 * through pybind11.
 *
 * Numerical conventions match the NumPy reference in
 * `python/pydnn/transformer/autograd.py`:
 *
 * - Row-major (C-order) storage.
 * - Backward kernels *accumulate* into the gradient buffers (caller is
 *   responsible for zeroing them or starting from existing grads).
 * - LayerNorm uses `(x - mean) / sqrt(var + eps)`; RMSNorm uses
 *   `x / sqrt(mean(x^2) + eps)`. Both apply `gamma * y + bias`.
 * - GELU is the tanh-approximation form (matches the Python `gelu`).
 * - Embedding backward is a scatter-add into a (V, D) gradient table.
 * - Cross-entropy uses `ignore_index` semantics (default -100).
 */

#include <cstddef>
#include <cstdint>

namespace dnn {
namespace transformer {

// ============================================================
// Batched matmul over leading dims.
//   A: (..., M, K)
//   B: (..., K, N)
//   C: (..., M, N)
// where the leading "..." dims are broadcast like numpy: each side may
// be size 1 along any leading axis. ``leading_dims`` is the broadcast
// shape (length ``leading_rank``); ``a_strides`` / ``b_strides`` are
// per-axis batch strides in *elements* (0 means broadcast).
// ============================================================
void matmul_batched_cpu(const float* a, const float* b, float* c,
                        const int64_t* leading_dims, int leading_rank,
                        const int64_t* a_batch_strides,
                        const int64_t* b_batch_strides,
                        int64_t M, int64_t K, int64_t N,
                        bool transpose_a = false,
                        bool transpose_b = false);

// ============================================================
// Softmax along the last axis. ``rows`` is the product of all leading
// dims; ``cols`` is the last-axis length. Numerically stable.
// ============================================================
void softmax_lastdim_cpu(const float* x, float* y,
                        int64_t rows, int64_t cols);

// Backward through softmax: dx = y * (dy - sum(dy * y, axis=-1, keepdims))
void softmax_lastdim_backward_cpu(const float* y, const float* dy, float* dx,
                                  int64_t rows, int64_t cols);

// ============================================================
// LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
// ``rows`` = product of leading dims, ``cols`` = feature dim.
// ``gamma``, ``beta`` are length-``cols`` 1-D arrays. ``mean`` and
// ``inv_std`` (length ``rows``) are saved for backward.
// ============================================================
void layernorm_forward_cpu(const float* x, const float* gamma, const float* beta,
                           float* y, float* mean_out, float* inv_std_out,
                           int64_t rows, int64_t cols, float eps);

void layernorm_backward_cpu(const float* x, const float* gamma,
                            const float* mean, const float* inv_std,
                            const float* dy, float* dx,
                            float* dgamma, float* dbeta,
                            int64_t rows, int64_t cols);

// ============================================================
// RMSNorm: y = gamma * x / sqrt(mean(x^2) + eps)
// ============================================================
void rmsnorm_forward_cpu(const float* x, const float* gamma,
                         float* y, float* inv_rms_out,
                         int64_t rows, int64_t cols, float eps);

void rmsnorm_backward_cpu(const float* x, const float* gamma,
                          const float* inv_rms,
                          const float* dy, float* dx, float* dgamma,
                          int64_t rows, int64_t cols);

// ============================================================
// Activations (elementwise)
// ============================================================
void gelu_forward_cpu(const float* x, float* y, int64_t n);
void gelu_backward_cpu(const float* x, const float* dy, float* dx, int64_t n);

void silu_forward_cpu(const float* x, float* y, int64_t n);
void silu_backward_cpu(const float* x, const float* dy, float* dx, int64_t n);

// ============================================================
// Embedding lookup: y[i] = weight[ids[i]]
//   weight: (V, D); ids: (N,) int64; y: (N, D)
// Backward scatter-adds dy into dweight (does not zero dweight first).
// ============================================================
void embedding_forward_cpu(const float* weight, const int64_t* ids,
                           float* y, int64_t n_ids, int64_t dim, int64_t vocab);

void embedding_backward_cpu(const float* dy, const int64_t* ids,
                            float* dweight, int64_t n_ids, int64_t dim,
                            int64_t vocab);

// ============================================================
// Fused softmax + cross-entropy (mean over non-ignored positions).
//   logits: (N, V) row-major
//   targets: (N,) int64; positions equal to ignore_index are skipped
// Returns the scalar loss; ``log_probs_out`` is filled with log-softmax
// per row (length N*V) so backward can finish in O(N*V) without redoing
// the softmax. ``valid_count`` returns the divisor used in the mean.
// ============================================================
float softmax_xent_forward_cpu(const float* logits, const int64_t* targets,
                               float* log_probs_out, int64_t* valid_count_out,
                               int64_t n, int64_t v, int64_t ignore_index);

// dlogits = (softmax - onehot) * (grad_loss / valid_count), zeroed at
// ignored positions.
void softmax_xent_backward_cpu(const float* log_probs, const int64_t* targets,
                               float grad_loss, int64_t valid_count,
                               float* dlogits,
                               int64_t n, int64_t v, int64_t ignore_index);

// ============================================================
// CUDA equivalents (declarations only; defined in src/cuda/transformer_*.cu).
// Stubs in transformer_ops.cpp throw when CUDA is not built so callers
// don't need to ifdef.
// ============================================================
bool cuda_transformer_available() noexcept;

void matmul_batched_cuda(const float* a, const float* b, float* c,
                         const int64_t* leading_dims, int leading_rank,
                         const int64_t* a_batch_strides,
                         const int64_t* b_batch_strides,
                         int64_t M, int64_t K, int64_t N,
                         bool transpose_a, bool transpose_b);

void softmax_lastdim_cuda(const float* x, float* y, int64_t rows, int64_t cols);
void softmax_lastdim_backward_cuda(const float* y, const float* dy, float* dx,
                                   int64_t rows, int64_t cols);

void layernorm_forward_cuda(const float* x, const float* gamma, const float* beta,
                            float* y, float* mean_out, float* inv_std_out,
                            int64_t rows, int64_t cols, float eps);
void layernorm_backward_cuda(const float* x, const float* gamma,
                             const float* mean, const float* inv_std,
                             const float* dy, float* dx,
                             float* dgamma, float* dbeta,
                             int64_t rows, int64_t cols);

void rmsnorm_forward_cuda(const float* x, const float* gamma,
                          float* y, float* inv_rms_out,
                          int64_t rows, int64_t cols, float eps);
void rmsnorm_backward_cuda(const float* x, const float* gamma,
                           const float* inv_rms,
                           const float* dy, float* dx, float* dgamma,
                           int64_t rows, int64_t cols);

void gelu_forward_cuda(const float* x, float* y, int64_t n);
void gelu_backward_cuda(const float* x, const float* dy, float* dx, int64_t n);
void silu_forward_cuda(const float* x, float* y, int64_t n);
void silu_backward_cuda(const float* x, const float* dy, float* dx, int64_t n);

void embedding_forward_cuda(const float* weight, const int64_t* ids,
                            float* y, int64_t n_ids, int64_t dim, int64_t vocab);
void embedding_backward_cuda(const float* dy, const int64_t* ids,
                             float* dweight, int64_t n_ids, int64_t dim,
                             int64_t vocab);

float softmax_xent_forward_cuda(const float* logits, const int64_t* targets,
                                float* log_probs_out, int64_t* valid_count_out,
                                int64_t n, int64_t v, int64_t ignore_index);
void softmax_xent_backward_cuda(const float* log_probs, const int64_t* targets,
                                float grad_loss, int64_t valid_count,
                                float* dlogits,
                                int64_t n, int64_t v, int64_t ignore_index);

}  // namespace transformer
}  // namespace dnn
