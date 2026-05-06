/**
 * CPU implementations of transformer ops.
 *
 * All kernels are templated on float, OpenMP-parallelised over the
 * outer dimension when available. Numerics match the NumPy reference
 * in `python/pydnn/transformer/autograd.py`.
 */

#include "dnn/transformer/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#define DNN_TX_PARALLEL_FOR _Pragma("omp parallel for")
#else
#define DNN_TX_PARALLEL_FOR
#endif

namespace dnn {
namespace transformer {

namespace {

constexpr float kGeluC = 0.7978845608028654f;   // sqrt(2/pi)
constexpr float kGeluA = 0.044715f;

// Resolve the per-batch base offset for a leading-multi-index using the
// provided batch strides (already in elements). Strides of 0 broadcast.
inline int64_t resolve_offset(const int64_t* dims, const int64_t* strides,
                              int rank, int64_t flat) {
    int64_t off = 0;
    for (int axis = rank - 1; axis >= 0; --axis) {
        int64_t d = dims[axis];
        int64_t i = (d == 0) ? 0 : (flat % d);
        off += i * strides[axis];
        flat = (d == 0) ? flat : (flat / d);
    }
    return off;
}

inline int64_t product(const int64_t* dims, int rank) {
    int64_t p = 1;
    for (int i = 0; i < rank; ++i) p *= dims[i];
    return p;
}

}  // namespace

// ============================================================
// Batched matmul
// ============================================================

void matmul_batched_cpu(const float* a, const float* b, float* c,
                        const int64_t* leading_dims, int leading_rank,
                        const int64_t* a_batch_strides,
                        const int64_t* b_batch_strides,
                        int64_t M, int64_t K, int64_t N,
                        bool transpose_a, bool transpose_b) {
    const int64_t batch = (leading_rank == 0) ? 1 : product(leading_dims, leading_rank);
    const int64_t c_mat = M * N;

    DNN_TX_PARALLEL_FOR
    for (int64_t bi = 0; bi < batch; ++bi) {
        const int64_t a_off = (leading_rank == 0) ? 0
            : resolve_offset(leading_dims, a_batch_strides, leading_rank, bi);
        const int64_t b_off = (leading_rank == 0) ? 0
            : resolve_offset(leading_dims, b_batch_strides, leading_rank, bi);
        const float* A = a + a_off;
        const float* B = b + b_off;
        float* C = c + bi * c_mat;

        // C = op(A) @ op(B), naive triple-loop (good enough for typical
        // transformer head_dim ~ 32-128; serious workloads should use
        // the CUDA path).
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                float acc = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    float av = transpose_a ? A[k * M + i] : A[i * K + k];
                    float bv = transpose_b ? B[j * K + k] : B[k * N + j];
                    acc += av * bv;
                }
                C[i * N + j] = acc;
            }
        }
    }
}

// ============================================================
// Softmax (last-axis)
// ============================================================

void softmax_lastdim_cpu(const float* x, float* y,
                        int64_t rows, int64_t cols) {
    DNN_TX_PARALLEL_FOR
    for (int64_t r = 0; r < rows; ++r) {
        const float* xr = x + r * cols;
        float* yr = y + r * cols;
        float m = -std::numeric_limits<float>::infinity();
        for (int64_t c = 0; c < cols; ++c) m = std::max(m, xr[c]);
        float s = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            float e = std::exp(xr[c] - m);
            yr[c] = e;
            s += e;
        }
        float inv = 1.0f / s;
        for (int64_t c = 0; c < cols; ++c) yr[c] *= inv;
    }
}

void softmax_lastdim_backward_cpu(const float* y, const float* dy, float* dx,
                                  int64_t rows, int64_t cols) {
    DNN_TX_PARALLEL_FOR
    for (int64_t r = 0; r < rows; ++r) {
        const float* yr = y + r * cols;
        const float* dyr = dy + r * cols;
        float* dxr = dx + r * cols;
        float dot = 0.0f;
        for (int64_t c = 0; c < cols; ++c) dot += dyr[c] * yr[c];
        for (int64_t c = 0; c < cols; ++c) dxr[c] = yr[c] * (dyr[c] - dot);
    }
}

// ============================================================
// LayerNorm
// ============================================================

void layernorm_forward_cpu(const float* x, const float* gamma, const float* beta,
                           float* y, float* mean_out, float* inv_std_out,
                           int64_t rows, int64_t cols, float eps) {
    DNN_TX_PARALLEL_FOR
    for (int64_t r = 0; r < rows; ++r) {
        const float* xr = x + r * cols;
        float* yr = y + r * cols;
        float m = 0.0f;
        for (int64_t c = 0; c < cols; ++c) m += xr[c];
        m /= float(cols);
        float v = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            float d = xr[c] - m;
            v += d * d;
        }
        v /= float(cols);
        float inv = 1.0f / std::sqrt(v + eps);
        if (mean_out) mean_out[r] = m;
        if (inv_std_out) inv_std_out[r] = inv;
        for (int64_t c = 0; c < cols; ++c) {
            float n = (xr[c] - m) * inv;
            yr[c] = n * gamma[c] + beta[c];
        }
    }
}

void layernorm_backward_cpu(const float* x, const float* gamma,
                            const float* mean, const float* inv_std,
                            const float* dy, float* dx,
                            float* dgamma, float* dbeta,
                            int64_t rows, int64_t cols) {
    // dgamma / dbeta are summed across rows; build per-thread buffers if
    // we ever care, for now zero-and-accumulate serially (the bulk of
    // the cost is per-row dx).
    if (dgamma) std::memset(dgamma, 0, sizeof(float) * cols);
    if (dbeta)  std::memset(dbeta,  0, sizeof(float) * cols);

    // Per-row dx in parallel; serialise the param-grad reduction.
    std::vector<double> dgamma_acc(cols, 0.0), dbeta_acc(cols, 0.0);

    for (int64_t r = 0; r < rows; ++r) {
        const float* xr = x + r * cols;
        const float* dyr = dy + r * cols;
        float* dxr = dx + r * cols;
        const float m = mean[r];
        const float inv = inv_std[r];

        // Pass 1: param grads + helper sums.
        float sum_g = 0.0f, sum_gn = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            float n = (xr[c] - m) * inv;
            float g = dyr[c] * gamma[c];
            sum_g  += g;
            sum_gn += g * n;
            dgamma_acc[c] += dyr[c] * n;
            dbeta_acc[c]  += dyr[c];
        }
        const float mg  = sum_g  / float(cols);
        const float mgn = sum_gn / float(cols);
        // Pass 2: dx.
        for (int64_t c = 0; c < cols; ++c) {
            float n = (xr[c] - m) * inv;
            float g = dyr[c] * gamma[c];
            dxr[c] = (g - mg - n * mgn) * inv;
        }
    }
    if (dgamma) for (int64_t c = 0; c < cols; ++c) dgamma[c] = float(dgamma_acc[c]);
    if (dbeta)  for (int64_t c = 0; c < cols; ++c) dbeta[c]  = float(dbeta_acc[c]);
}

// ============================================================
// RMSNorm
// ============================================================

void rmsnorm_forward_cpu(const float* x, const float* gamma,
                         float* y, float* inv_rms_out,
                         int64_t rows, int64_t cols, float eps) {
    DNN_TX_PARALLEL_FOR
    for (int64_t r = 0; r < rows; ++r) {
        const float* xr = x + r * cols;
        float* yr = y + r * cols;
        float s = 0.0f;
        for (int64_t c = 0; c < cols; ++c) s += xr[c] * xr[c];
        float inv = 1.0f / std::sqrt(s / float(cols) + eps);
        if (inv_rms_out) inv_rms_out[r] = inv;
        for (int64_t c = 0; c < cols; ++c) yr[c] = xr[c] * inv * gamma[c];
    }
}

void rmsnorm_backward_cpu(const float* x, const float* gamma,
                          const float* inv_rms,
                          const float* dy, float* dx, float* dgamma,
                          int64_t rows, int64_t cols) {
    if (dgamma) std::memset(dgamma, 0, sizeof(float) * cols);
    std::vector<double> dgamma_acc(cols, 0.0);

    for (int64_t r = 0; r < rows; ++r) {
        const float* xr = x + r * cols;
        const float* dyr = dy + r * cols;
        float* dxr = dx + r * cols;
        const float inv = inv_rms[r];
        const float inv2 = inv * inv;

        float dot_gx = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            float g = dyr[c] * gamma[c];
            dot_gx += g * xr[c];
            dgamma_acc[c] += dyr[c] * xr[c] * inv;
        }
        const float coef = dot_gx * inv2 / float(cols);
        for (int64_t c = 0; c < cols; ++c) {
            float g = dyr[c] * gamma[c];
            dxr[c] = (g - xr[c] * coef) * inv;
        }
    }
    if (dgamma) for (int64_t c = 0; c < cols; ++c) dgamma[c] = float(dgamma_acc[c]);
}

// ============================================================
// Activations
// ============================================================

void gelu_forward_cpu(const float* x, float* y, int64_t n) {
    DNN_TX_PARALLEL_FOR
    for (int64_t i = 0; i < n; ++i) {
        float v = x[i];
        float t = std::tanh(kGeluC * (v + kGeluA * v * v * v));
        y[i] = 0.5f * v * (1.0f + t);
    }
}

void gelu_backward_cpu(const float* x, const float* dy, float* dx, int64_t n) {
    DNN_TX_PARALLEL_FOR
    for (int64_t i = 0; i < n; ++i) {
        float v = x[i];
        float inner = kGeluC * (v + kGeluA * v * v * v);
        float t = std::tanh(inner);
        float sech2 = 1.0f - t * t;
        float d_inner = kGeluC * (1.0f + 3.0f * kGeluA * v * v);
        float dv = 0.5f * (1.0f + t) + 0.5f * v * sech2 * d_inner;
        dx[i] = dy[i] * dv;
    }
}

void silu_forward_cpu(const float* x, float* y, int64_t n) {
    DNN_TX_PARALLEL_FOR
    for (int64_t i = 0; i < n; ++i) {
        float v = x[i];
        float s = 1.0f / (1.0f + std::exp(-v));
        y[i] = v * s;
    }
}

void silu_backward_cpu(const float* x, const float* dy, float* dx, int64_t n) {
    DNN_TX_PARALLEL_FOR
    for (int64_t i = 0; i < n; ++i) {
        float v = x[i];
        float s = 1.0f / (1.0f + std::exp(-v));
        dx[i] = dy[i] * (s + v * s * (1.0f - s));
    }
}

// ============================================================
// Embedding
// ============================================================

void embedding_forward_cpu(const float* weight, const int64_t* ids,
                           float* y, int64_t n_ids, int64_t dim, int64_t vocab) {
    DNN_TX_PARALLEL_FOR
    for (int64_t i = 0; i < n_ids; ++i) {
        int64_t id = ids[i];
        if (id < 0 || id >= vocab) {
            // Match numpy semantics: out-of-range raises in Python; here
            // we zero-fill defensively rather than read garbage.
            std::memset(y + i * dim, 0, sizeof(float) * dim);
            continue;
        }
        std::memcpy(y + i * dim, weight + id * dim, sizeof(float) * dim);
    }
}

void embedding_backward_cpu(const float* dy, const int64_t* ids,
                            float* dweight, int64_t n_ids, int64_t dim,
                            int64_t vocab) {
    // Note: scatter-add must be deterministic for testability, so we do
    // it serially. Token counts are typically small relative to dim*V.
    for (int64_t i = 0; i < n_ids; ++i) {
        int64_t id = ids[i];
        if (id < 0 || id >= vocab) continue;
        const float* src = dy + i * dim;
        float* dst = dweight + id * dim;
        for (int64_t d = 0; d < dim; ++d) dst[d] += src[d];
    }
}

// ============================================================
// Fused softmax + cross-entropy
// ============================================================

float softmax_xent_forward_cpu(const float* logits, const int64_t* targets,
                               float* log_probs_out, int64_t* valid_count_out,
                               int64_t n, int64_t v, int64_t ignore_index) {
    double total = 0.0;
    int64_t valid = 0;
    for (int64_t r = 0; r < n; ++r) {
        const float* row = logits + r * v;
        float* lp_row = log_probs_out + r * v;
        float m = -std::numeric_limits<float>::infinity();
        for (int64_t c = 0; c < v; ++c) m = std::max(m, row[c]);
        double s = 0.0;
        for (int64_t c = 0; c < v; ++c) s += std::exp(row[c] - m);
        float log_z = float(std::log(s)) + m;
        for (int64_t c = 0; c < v; ++c) lp_row[c] = row[c] - log_z;

        int64_t t = targets[r];
        if (t == ignore_index) continue;
        if (t < 0 || t >= v) continue;
        total += -lp_row[t];
        valid += 1;
    }
    if (valid_count_out) *valid_count_out = valid;
    int64_t denom = (valid > 0) ? valid : 1;
    return float(total / double(denom));
}

void softmax_xent_backward_cpu(const float* log_probs, const int64_t* targets,
                               float grad_loss, int64_t valid_count,
                               float* dlogits,
                               int64_t n, int64_t v, int64_t ignore_index) {
    const int64_t denom = (valid_count > 0) ? valid_count : 1;
    const float scale = grad_loss / float(denom);

    DNN_TX_PARALLEL_FOR
    for (int64_t r = 0; r < n; ++r) {
        const float* lp_row = log_probs + r * v;
        float* dl_row = dlogits + r * v;
        int64_t t = targets[r];
        bool valid = (t != ignore_index) && (t >= 0) && (t < v);
        if (!valid) {
            std::memset(dl_row, 0, sizeof(float) * v);
            continue;
        }
        for (int64_t c = 0; c < v; ++c) {
            float p = std::exp(lp_row[c]);
            dl_row[c] = scale * p;
        }
        dl_row[t] -= scale;
    }
}

}  // namespace transformer
}  // namespace dnn
