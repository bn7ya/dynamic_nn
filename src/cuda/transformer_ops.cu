/**
 * CUDA implementations of transformer ops.
 *
 * Same numerics as the CPU reference in ``src/transformer/cpu_ops.cpp``.
 * Buffers passed in are *device* pointers — the pybind layer copies
 * host numpy arrays in/out of cuda::CudaTensor staging buffers.
 *
 * Single .cu file (rather than one per op) keeps the build matrix small
 * and lets us share helpers like ``cdiv`` and the cublas handle.
 */

#ifdef DNN_ENABLE_CUDA

#include "dnn/transformer/ops.hpp"
#include "dnn/cuda/cuda_common.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace dnn {
namespace transformer {

namespace {

constexpr int kThreads = 256;

inline int cdiv(int64_t a, int64_t b) { return int((a + b - 1) / b); }

// One cublas handle per process (matches src/cuda/cuda_blas.cu).
cublasHandle_t& cublas_handle() {
    static cublasHandle_t h = nullptr;
    static bool init = false;
    if (!init) {
        cublasCreate(&h);
        cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
        init = true;
    }
    return h;
}

}  // namespace

bool cuda_transformer_available() noexcept {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

// ============================================================
// Batched matmul. Row-major in / row-major out, via the cuBLAS
// operand-swap trick (compute C^T = B^T @ A^T to recover row-major).
// Broadcasted leading dims: we materialise the per-batch source
// pointers up-front (cheap; len = batch).
// ============================================================
void matmul_batched_cuda(const float* a, const float* b, float* c,
                         const int64_t* leading_dims, int leading_rank,
                         const int64_t* a_batch_strides,
                         const int64_t* b_batch_strides,
                         int64_t M, int64_t K, int64_t N,
                         bool transpose_a, bool transpose_b) {
    int64_t batch = 1;
    for (int i = 0; i < leading_rank; ++i) batch *= leading_dims[i];
    if (batch <= 0) return;

    const float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle = cublas_handle();

    // For a typical attention layout the strides are uniform so we
    // can use SgemmStridedBatched directly. When any stride is 0
    // (broadcast) we fall back to per-batch Sgemm.
    bool uniform = true;
    int64_t a_stride = 0, b_stride = 0;
    if (leading_rank > 0) {
        a_stride = a_batch_strides[0];
        b_stride = b_batch_strides[0];
    }
    for (int i = 1; i < leading_rank && uniform; ++i) {
        // Each non-trivial leading axis must contribute a constant stride
        // pattern. This holds when the leading dims have *no* broadcast.
        if (a_batch_strides[i] == 0 || b_batch_strides[i] == 0) {
            uniform = false;
        }
    }

    auto opA = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto opB = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    // Row-major emulation: compute C = A @ B as
    //   C^T = B^T @ A^T  in column-major,
    // i.e. cublasSgemm(handle, opB, opA, N, M, K, ..., B, ldB, A, ldA, ..., C, ldC)
    // Leading dims (row-major) are the *innermost* dim:
    //   ldA = K (or M if transpose_a),
    //   ldB = N (or K if transpose_b),
    //   ldC = N.
    int ldA = transpose_a ? int(M) : int(K);
    int ldB = transpose_b ? int(K) : int(N);
    int ldC = int(N);

    if (uniform && leading_rank > 0) {
        cublasSgemmStridedBatched(
            handle, opB, opA,
            int(N), int(M), int(K),
            &alpha,
            b, ldB, b_stride,
            a, ldA, a_stride,
            &beta,
            c, ldC, M * N,
            int(batch));
        return;
    }

    // Fallback: walk batch index by index, resolving offsets via the
    // strides (handles broadcasting).
    for (int64_t bi = 0; bi < batch; ++bi) {
        int64_t flat = bi;
        int64_t a_off = 0, b_off = 0;
        for (int axis = leading_rank - 1; axis >= 0; --axis) {
            int64_t d = leading_dims[axis];
            int64_t i = (d == 0) ? 0 : (flat % d);
            a_off += i * a_batch_strides[axis];
            b_off += i * b_batch_strides[axis];
            flat = (d == 0) ? flat : (flat / d);
        }
        cublasSgemm(handle, opB, opA,
                    int(N), int(M), int(K),
                    &alpha,
                    b + b_off, ldB,
                    a + a_off, ldA,
                    &beta,
                    c + bi * M * N, ldC);
    }
}

// ============================================================
// Softmax (last axis) — block per row, stable 3-pass.
// ============================================================

__global__ void softmax_lastdim_kernel(const float* __restrict__ x,
                                       float* __restrict__ y,
                                       int64_t cols) {
    extern __shared__ float smem[];
    int64_t row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + row * cols;
    float* yr = y + row * cols;

    // Pass 1: max
    float m = -INFINITY;
    for (int64_t c = tid; c < cols; c += blockDim.x) m = fmaxf(m, xr[c]);
    smem[tid] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float row_max = smem[0];

    // Pass 2: sum(exp)
    float local_sum = 0.0f;
    for (int64_t c = tid; c < cols; c += blockDim.x) {
        float e = __expf(xr[c] - row_max);
        yr[c] = e;
        local_sum += e;
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float row_sum = smem[0];
    float inv = 1.0f / row_sum;

    // Pass 3: normalise
    for (int64_t c = tid; c < cols; c += blockDim.x) yr[c] *= inv;
}

void softmax_lastdim_cuda(const float* x, float* y, int64_t rows, int64_t cols) {
    if (rows <= 0 || cols <= 0) return;
    int threads = 256;
    while (threads > cols && threads > 32) threads >>= 1;
    softmax_lastdim_kernel<<<int(rows), threads, threads * sizeof(float)>>>(x, y, cols);
}

__global__ void softmax_lastdim_bwd_kernel(const float* __restrict__ y,
                                           const float* __restrict__ dy,
                                           float* __restrict__ dx,
                                           int64_t cols) {
    extern __shared__ float smem[];
    int64_t row = blockIdx.x;
    int tid = threadIdx.x;
    const float* yr = y + row * cols;
    const float* dyr = dy + row * cols;
    float* dxr = dx + row * cols;

    float local = 0.0f;
    for (int64_t c = tid; c < cols; c += blockDim.x) local += yr[c] * dyr[c];
    smem[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float dot = smem[0];
    for (int64_t c = tid; c < cols; c += blockDim.x) {
        dxr[c] = yr[c] * (dyr[c] - dot);
    }
}

void softmax_lastdim_backward_cuda(const float* y, const float* dy, float* dx,
                                   int64_t rows, int64_t cols) {
    if (rows <= 0 || cols <= 0) return;
    int threads = 256;
    while (threads > cols && threads > 32) threads >>= 1;
    softmax_lastdim_bwd_kernel<<<int(rows), threads, threads * sizeof(float)>>>(
        y, dy, dx, cols);
}

// ============================================================
// LayerNorm
// ============================================================

__global__ void layernorm_fwd_kernel(const float* __restrict__ x,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ beta,
                                     float* __restrict__ y,
                                     float* __restrict__ mean_out,
                                     float* __restrict__ inv_std_out,
                                     int64_t cols, float eps) {
    extern __shared__ float smem[];
    float* sum = smem;
    float* sqsum = smem + blockDim.x;
    int64_t row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + row * cols;
    float* yr = y + row * cols;

    float ls = 0.0f, lq = 0.0f;
    for (int64_t c = tid; c < cols; c += blockDim.x) {
        float v = xr[c];
        ls += v;
        lq += v * v;
    }
    sum[tid] = ls;
    sqsum[tid] = lq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum[tid] += sum[tid + s];
            sqsum[tid] += sqsum[tid + s];
        }
        __syncthreads();
    }
    float m = sum[0] / float(cols);
    float v = sqsum[0] / float(cols) - m * m;
    if (v < 0.0f) v = 0.0f;
    float inv = rsqrtf(v + eps);
    if (tid == 0 && mean_out) mean_out[row] = m;
    if (tid == 0 && inv_std_out) inv_std_out[row] = inv;
    for (int64_t c = tid; c < cols; c += blockDim.x) {
        float n = (xr[c] - m) * inv;
        yr[c] = n * gamma[c] + beta[c];
    }
}

void layernorm_forward_cuda(const float* x, const float* gamma, const float* beta,
                            float* y, float* mean_out, float* inv_std_out,
                            int64_t rows, int64_t cols, float eps) {
    if (rows <= 0 || cols <= 0) return;
    int threads = 256;
    while (threads > cols && threads > 32) threads >>= 1;
    layernorm_fwd_kernel<<<int(rows), threads, 2 * threads * sizeof(float)>>>(
        x, gamma, beta, y, mean_out, inv_std_out, cols, eps);
}

__global__ void layernorm_bwd_kernel(const float* __restrict__ x,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ mean,
                                     const float* __restrict__ inv_std,
                                     const float* __restrict__ dy,
                                     float* __restrict__ dx,
                                     float* __restrict__ dgamma_per_row,
                                     float* __restrict__ dbeta_per_row,
                                     int64_t cols) {
    extern __shared__ float smem[];
    float* sg = smem;
    float* sgn = smem + blockDim.x;
    int64_t row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + row * cols;
    const float* dyr = dy + row * cols;
    float* dxr = dx + row * cols;
    const float m = mean[row];
    const float inv = inv_std[row];

    float l_g = 0.0f, l_gn = 0.0f;
    for (int64_t c = tid; c < cols; c += blockDim.x) {
        float n = (xr[c] - m) * inv;
        float g = dyr[c] * gamma[c];
        l_g  += g;
        l_gn += g * n;
    }
    sg[tid]  = l_g;
    sgn[tid] = l_gn;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sg[tid]  += sg[tid + s];
            sgn[tid] += sgn[tid + s];
        }
        __syncthreads();
    }
    float mg  = sg[0]  / float(cols);
    float mgn = sgn[0] / float(cols);

    for (int64_t c = tid; c < cols; c += blockDim.x) {
        float n = (xr[c] - m) * inv;
        float g = dyr[c] * gamma[c];
        dxr[c] = (g - mg - n * mgn) * inv;
        dgamma_per_row[row * cols + c] = dyr[c] * n;
        dbeta_per_row[row * cols + c]  = dyr[c];
    }
}

__global__ void column_sum_kernel(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  int64_t rows, int64_t cols) {
    int64_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;
    float s = 0.0f;
    for (int64_t r = 0; r < rows; ++r) s += in[r * cols + c];
    out[c] = s;
}

void layernorm_backward_cuda(const float* x, const float* gamma,
                             const float* mean, const float* inv_std,
                             const float* dy, float* dx,
                             float* dgamma, float* dbeta,
                             int64_t rows, int64_t cols) {
    if (rows <= 0 || cols <= 0) return;
    // Per-row dgamma/dbeta scratch then reduce across rows.
    float* dg_rows = nullptr;
    float* db_rows = nullptr;
    cudaMalloc(&dg_rows, sizeof(float) * rows * cols);
    cudaMalloc(&db_rows, sizeof(float) * rows * cols);

    int threads = 256;
    while (threads > cols && threads > 32) threads >>= 1;
    layernorm_bwd_kernel<<<int(rows), threads, 2 * threads * sizeof(float)>>>(
        x, gamma, mean, inv_std, dy, dx, dg_rows, db_rows, cols);

    if (dgamma) {
        column_sum_kernel<<<cdiv(cols, kThreads), kThreads>>>(
            dg_rows, dgamma, rows, cols);
    }
    if (dbeta) {
        column_sum_kernel<<<cdiv(cols, kThreads), kThreads>>>(
            db_rows, dbeta, rows, cols);
    }
    cudaFree(dg_rows);
    cudaFree(db_rows);
}

// ============================================================
// RMSNorm
// ============================================================

__global__ void rmsnorm_fwd_kernel(const float* __restrict__ x,
                                   const float* __restrict__ gamma,
                                   float* __restrict__ y,
                                   float* __restrict__ inv_rms_out,
                                   int64_t cols, float eps) {
    extern __shared__ float smem[];
    int64_t row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + row * cols;
    float* yr = y + row * cols;

    float ls = 0.0f;
    for (int64_t c = tid; c < cols; c += blockDim.x) {
        float v = xr[c];
        ls += v * v;
    }
    smem[tid] = ls;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv = rsqrtf(smem[0] / float(cols) + eps);
    if (tid == 0 && inv_rms_out) inv_rms_out[row] = inv;
    for (int64_t c = tid; c < cols; c += blockDim.x) {
        yr[c] = xr[c] * inv * gamma[c];
    }
}

void rmsnorm_forward_cuda(const float* x, const float* gamma,
                          float* y, float* inv_rms_out,
                          int64_t rows, int64_t cols, float eps) {
    if (rows <= 0 || cols <= 0) return;
    int threads = 256;
    while (threads > cols && threads > 32) threads >>= 1;
    rmsnorm_fwd_kernel<<<int(rows), threads, threads * sizeof(float)>>>(
        x, gamma, y, inv_rms_out, cols, eps);
}

__global__ void rmsnorm_bwd_kernel(const float* __restrict__ x,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ inv_rms,
                                   const float* __restrict__ dy,
                                   float* __restrict__ dx,
                                   float* __restrict__ dgamma_per_row,
                                   int64_t cols) {
    extern __shared__ float smem[];
    int64_t row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + row * cols;
    const float* dyr = dy + row * cols;
    float* dxr = dx + row * cols;
    float inv = inv_rms[row];
    float inv2 = inv * inv;

    float local = 0.0f;
    for (int64_t c = tid; c < cols; c += blockDim.x) {
        float g = dyr[c] * gamma[c];
        local += g * xr[c];
    }
    smem[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float dot = smem[0];
    float coef = dot * inv2 / float(cols);

    for (int64_t c = tid; c < cols; c += blockDim.x) {
        float g = dyr[c] * gamma[c];
        dxr[c] = (g - xr[c] * coef) * inv;
        dgamma_per_row[row * cols + c] = dyr[c] * xr[c] * inv;
    }
}

void rmsnorm_backward_cuda(const float* x, const float* gamma,
                           const float* inv_rms,
                           const float* dy, float* dx, float* dgamma,
                           int64_t rows, int64_t cols) {
    if (rows <= 0 || cols <= 0) return;
    float* dg_rows = nullptr;
    cudaMalloc(&dg_rows, sizeof(float) * rows * cols);

    int threads = 256;
    while (threads > cols && threads > 32) threads >>= 1;
    rmsnorm_bwd_kernel<<<int(rows), threads, threads * sizeof(float)>>>(
        x, gamma, inv_rms, dy, dx, dg_rows, cols);

    if (dgamma) {
        column_sum_kernel<<<cdiv(cols, kThreads), kThreads>>>(
            dg_rows, dgamma, rows, cols);
    }
    cudaFree(dg_rows);
}

// ============================================================
// GELU / SiLU
// ============================================================

__device__ inline float gelu_fwd(float v) {
    constexpr float c = 0.7978845608028654f;
    constexpr float a = 0.044715f;
    float t = tanhf(c * (v + a * v * v * v));
    return 0.5f * v * (1.0f + t);
}

__global__ void gelu_fwd_kernel(const float* x, float* y, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = gelu_fwd(x[i]);
}

__global__ void gelu_bwd_kernel(const float* x, const float* dy, float* dx, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    constexpr float c = 0.7978845608028654f;
    constexpr float a = 0.044715f;
    float v = x[i];
    float inner = c * (v + a * v * v * v);
    float t = tanhf(inner);
    float sech2 = 1.0f - t * t;
    float d_inner = c * (1.0f + 3.0f * a * v * v);
    float dv = 0.5f * (1.0f + t) + 0.5f * v * sech2 * d_inner;
    dx[i] = dy[i] * dv;
}

void gelu_forward_cuda(const float* x, float* y, int64_t n) {
    if (n <= 0) return;
    gelu_fwd_kernel<<<cdiv(n, kThreads), kThreads>>>(x, y, n);
}
void gelu_backward_cuda(const float* x, const float* dy, float* dx, int64_t n) {
    if (n <= 0) return;
    gelu_bwd_kernel<<<cdiv(n, kThreads), kThreads>>>(x, dy, dx, n);
}

__global__ void silu_fwd_kernel(const float* x, float* y, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float s = 1.0f / (1.0f + __expf(-v));
    y[i] = v * s;
}

__global__ void silu_bwd_kernel(const float* x, const float* dy, float* dx, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float s = 1.0f / (1.0f + __expf(-v));
    dx[i] = dy[i] * (s + v * s * (1.0f - s));
}

void silu_forward_cuda(const float* x, float* y, int64_t n) {
    if (n <= 0) return;
    silu_fwd_kernel<<<cdiv(n, kThreads), kThreads>>>(x, y, n);
}
void silu_backward_cuda(const float* x, const float* dy, float* dx, int64_t n) {
    if (n <= 0) return;
    silu_bwd_kernel<<<cdiv(n, kThreads), kThreads>>>(x, dy, dx, n);
}

// ============================================================
// Embedding
// ============================================================

__global__ void embedding_fwd_kernel(const float* __restrict__ weight,
                                     const int64_t* __restrict__ ids,
                                     float* __restrict__ y,
                                     int64_t n_ids, int64_t dim, int64_t vocab) {
    int64_t i = blockIdx.x;
    int tid = threadIdx.x;
    if (i >= n_ids) return;
    int64_t id = ids[i];
    if (id < 0 || id >= vocab) {
        for (int64_t d = tid; d < dim; d += blockDim.x) y[i * dim + d] = 0.0f;
        return;
    }
    for (int64_t d = tid; d < dim; d += blockDim.x) {
        y[i * dim + d] = weight[id * dim + d];
    }
}

void embedding_forward_cuda(const float* weight, const int64_t* ids,
                            float* y, int64_t n_ids, int64_t dim, int64_t vocab) {
    if (n_ids <= 0 || dim <= 0) return;
    int threads = 128;
    while (threads > dim && threads > 32) threads >>= 1;
    embedding_fwd_kernel<<<int(n_ids), threads>>>(weight, ids, y, n_ids, dim, vocab);
}

__global__ void embedding_bwd_kernel(const float* __restrict__ dy,
                                     const int64_t* __restrict__ ids,
                                     float* __restrict__ dweight,
                                     int64_t n_ids, int64_t dim, int64_t vocab) {
    int64_t i = blockIdx.x;
    int tid = threadIdx.x;
    if (i >= n_ids) return;
    int64_t id = ids[i];
    if (id < 0 || id >= vocab) return;
    for (int64_t d = tid; d < dim; d += blockDim.x) {
        atomicAdd(dweight + id * dim + d, dy[i * dim + d]);
    }
}

void embedding_backward_cuda(const float* dy, const int64_t* ids,
                             float* dweight, int64_t n_ids, int64_t dim,
                             int64_t vocab) {
    if (n_ids <= 0 || dim <= 0) return;
    int threads = 128;
    while (threads > dim && threads > 32) threads >>= 1;
    embedding_bwd_kernel<<<int(n_ids), threads>>>(dy, ids, dweight, n_ids, dim, vocab);
}

// ============================================================
// Fused softmax + cross-entropy
// ============================================================

__global__ void xent_fwd_kernel(const float* __restrict__ logits,
                                const int64_t* __restrict__ targets,
                                float* __restrict__ log_probs,
                                float* __restrict__ row_loss,
                                int* __restrict__ row_valid,
                                int64_t v, int64_t ignore_index) {
    extern __shared__ float smem[];
    int64_t row = blockIdx.x;
    int tid = threadIdx.x;
    const float* row_in = logits + row * v;
    float* row_lp = log_probs + row * v;

    float lm = -INFINITY;
    for (int64_t c = tid; c < v; c += blockDim.x) lm = fmaxf(lm, row_in[c]);
    smem[tid] = lm;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float row_max = smem[0];

    float local = 0.0f;
    for (int64_t c = tid; c < v; c += blockDim.x) local += __expf(row_in[c] - row_max);
    smem[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float log_z = logf(smem[0]) + row_max;
    for (int64_t c = tid; c < v; c += blockDim.x) row_lp[c] = row_in[c] - log_z;

    if (tid == 0) {
        int64_t t = targets[row];
        bool valid = (t != ignore_index) && (t >= 0) && (t < v);
        row_loss[row] = valid ? -row_lp[t] : 0.0f;
        row_valid[row] = valid ? 1 : 0;
    }
}

__global__ void scalar_reduce_kernel(const float* __restrict__ in,
                                     const int* __restrict__ valid,
                                     float* __restrict__ loss_out,
                                     int64_t* __restrict__ valid_out,
                                     int64_t n) {
    double s = 0.0;
    int64_t cnt = 0;
    for (int64_t i = 0; i < n; ++i) {
        s += in[i];
        cnt += valid[i];
    }
    int64_t denom = (cnt > 0) ? cnt : 1;
    loss_out[0] = float(s / double(denom));
    valid_out[0] = cnt;
}

float softmax_xent_forward_cuda(const float* logits, const int64_t* targets,
                                float* log_probs_out, int64_t* valid_count_out,
                                int64_t n, int64_t v, int64_t ignore_index) {
    if (n <= 0 || v <= 0) {
        if (valid_count_out) *valid_count_out = 0;
        return 0.0f;
    }
    int threads = 256;
    while (threads > v && threads > 32) threads >>= 1;

    float* row_loss; cudaMalloc(&row_loss, sizeof(float) * n);
    int* row_valid; cudaMalloc(&row_valid, sizeof(int) * n);

    xent_fwd_kernel<<<int(n), threads, threads * sizeof(float)>>>(
        logits, targets, log_probs_out, row_loss, row_valid, v, ignore_index);

    float* d_loss; cudaMalloc(&d_loss, sizeof(float));
    int64_t* d_valid; cudaMalloc(&d_valid, sizeof(int64_t));
    scalar_reduce_kernel<<<1, 1>>>(row_loss, row_valid, d_loss, d_valid, n);

    float h_loss = 0.0f;
    int64_t h_valid = 0;
    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_valid, d_valid, sizeof(int64_t), cudaMemcpyDeviceToHost);
    if (valid_count_out) *valid_count_out = h_valid;

    cudaFree(row_loss);
    cudaFree(row_valid);
    cudaFree(d_loss);
    cudaFree(d_valid);
    return h_loss;
}

__global__ void xent_bwd_kernel(const float* __restrict__ log_probs,
                                const int64_t* __restrict__ targets,
                                float scale,
                                float* __restrict__ dlogits,
                                int64_t v, int64_t ignore_index) {
    int64_t row = blockIdx.x;
    int tid = threadIdx.x;
    const float* lp = log_probs + row * v;
    float* dl = dlogits + row * v;
    int64_t t = targets[row];
    bool valid = (t != ignore_index) && (t >= 0) && (t < v);
    if (!valid) {
        for (int64_t c = tid; c < v; c += blockDim.x) dl[c] = 0.0f;
        return;
    }
    for (int64_t c = tid; c < v; c += blockDim.x) {
        float p = __expf(lp[c]);
        dl[c] = scale * p;
    }
    if (tid == 0) dl[t] -= scale;
}

void softmax_xent_backward_cuda(const float* log_probs, const int64_t* targets,
                                float grad_loss, int64_t valid_count,
                                float* dlogits,
                                int64_t n, int64_t v, int64_t ignore_index) {
    if (n <= 0 || v <= 0) return;
    int64_t denom = (valid_count > 0) ? valid_count : 1;
    float scale = grad_loss / float(denom);
    int threads = 256;
    while (threads > v && threads > 32) threads >>= 1;
    xent_bwd_kernel<<<int(n), threads>>>(log_probs, targets, scale, dlogits, v, ignore_index);
}

}  // namespace transformer
}  // namespace dnn

#endif  // DNN_ENABLE_CUDA
