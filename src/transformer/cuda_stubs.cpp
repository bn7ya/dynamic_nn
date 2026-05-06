/**
 * Stubs for the CUDA transformer ops when the build was configured
 * without ``DNN_ENABLE_CUDA``. Each stub raises a clear runtime_error;
 * `cuda_transformer_available()` returns false so the Python dispatch
 * layer can route around them silently.
 *
 * The matching real implementations live in ``src/cuda/transformer_*.cu``
 * and are only compiled in when CUDA is enabled.
 */

#ifndef DNN_ENABLE_CUDA

#include "dnn/transformer/ops.hpp"

#include <stdexcept>

namespace dnn {
namespace transformer {

bool cuda_transformer_available() noexcept { return false; }

namespace {
[[noreturn]] inline void unavailable(const char* name) {
    throw std::runtime_error(std::string(name) + ": CUDA transformer ops not built");
}
}

void matmul_batched_cuda(const float*, const float*, float*,
                         const int64_t*, int, const int64_t*, const int64_t*,
                         int64_t, int64_t, int64_t, bool, bool) {
    unavailable("matmul_batched_cuda");
}

void softmax_lastdim_cuda(const float*, float*, int64_t, int64_t) {
    unavailable("softmax_lastdim_cuda");
}
void softmax_lastdim_backward_cuda(const float*, const float*, float*,
                                   int64_t, int64_t) {
    unavailable("softmax_lastdim_backward_cuda");
}

void layernorm_forward_cuda(const float*, const float*, const float*,
                            float*, float*, float*,
                            int64_t, int64_t, float) {
    unavailable("layernorm_forward_cuda");
}
void layernorm_backward_cuda(const float*, const float*,
                             const float*, const float*,
                             const float*, float*,
                             float*, float*, int64_t, int64_t) {
    unavailable("layernorm_backward_cuda");
}

void rmsnorm_forward_cuda(const float*, const float*, float*, float*,
                          int64_t, int64_t, float) {
    unavailable("rmsnorm_forward_cuda");
}
void rmsnorm_backward_cuda(const float*, const float*, const float*,
                           const float*, float*, float*, int64_t, int64_t) {
    unavailable("rmsnorm_backward_cuda");
}

void gelu_forward_cuda(const float*, float*, int64_t)  { unavailable("gelu_forward_cuda"); }
void gelu_backward_cuda(const float*, const float*, float*, int64_t) {
    unavailable("gelu_backward_cuda");
}
void silu_forward_cuda(const float*, float*, int64_t)  { unavailable("silu_forward_cuda"); }
void silu_backward_cuda(const float*, const float*, float*, int64_t) {
    unavailable("silu_backward_cuda");
}

void embedding_forward_cuda(const float*, const int64_t*, float*,
                            int64_t, int64_t, int64_t) {
    unavailable("embedding_forward_cuda");
}
void embedding_backward_cuda(const float*, const int64_t*, float*,
                             int64_t, int64_t, int64_t) {
    unavailable("embedding_backward_cuda");
}

float softmax_xent_forward_cuda(const float*, const int64_t*, float*, int64_t*,
                                int64_t, int64_t, int64_t) {
    unavailable("softmax_xent_forward_cuda");
}
void softmax_xent_backward_cuda(const float*, const int64_t*, float, int64_t,
                                float*, int64_t, int64_t, int64_t) {
    unavailable("softmax_xent_backward_cuda");
}

}  // namespace transformer
}  // namespace dnn

#endif  // !DNN_ENABLE_CUDA
