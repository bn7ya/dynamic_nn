/**
 * CUDA Optimizer Kernels
 *
 * Implements GPU-accelerated optimizer update kernels for:
 * - SGD (Stochastic Gradient Descent)
 * - SGD with Momentum
 * - Adam (Adaptive Moment Estimation)
 * - RMSprop (Root Mean Square Propagation)
 *
 * All kernels match the Python implementation for algorithm parity.
 */

#ifdef DNN_ENABLE_CUDA

#include "dnn/cuda/cuda_common.hpp"
#include "dnn/cuda/cuda_ops.hpp"
#include <cuda_runtime.h>
#include <cmath>

namespace dnn {
namespace cuda {

// ============================================================================
// SGD Kernels
// ============================================================================

/**
 * Basic SGD update kernel: W -= lr * grad
 */
template<typename T>
__global__ void sgd_update_kernel(
    T* weights,
    const T* gradients,
    T learning_rate,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        weights[idx] -= learning_rate * gradients[idx];
    }
}

/**
 * SGD with weight decay: W -= lr * (grad + weight_decay * W)
 */
template<typename T>
__global__ void sgd_weight_decay_kernel(
    T* weights,
    const T* gradients,
    T learning_rate,
    T weight_decay,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T grad = gradients[idx] + weight_decay * weights[idx];
        weights[idx] -= learning_rate * grad;
    }
}

/**
 * Gradient clipping kernel: grad = clamp(grad, -clip_value, clip_value)
 */
template<typename T>
__global__ void gradient_clip_kernel(
    T* gradients,
    T clip_value,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T g = gradients[idx];
        gradients[idx] = fmaxf(-clip_value, fminf(clip_value, g));
    }
}

// ============================================================================
// SGD with Momentum Kernels
// ============================================================================

/**
 * SGD with Momentum update kernel:
 * v = momentum * v + lr * grad
 * W -= v
 */
template<typename T>
__global__ void sgd_momentum_kernel(
    T* weights,
    const T* gradients,
    T* velocity,
    T learning_rate,
    T momentum,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T v = momentum * velocity[idx] + learning_rate * gradients[idx];
        velocity[idx] = v;
        weights[idx] -= v;
    }
}

/**
 * SGD with Momentum and weight decay:
 * v = momentum * v + lr * (grad + weight_decay * W)
 * W -= v
 */
template<typename T>
__global__ void sgd_momentum_weight_decay_kernel(
    T* weights,
    const T* gradients,
    T* velocity,
    T learning_rate,
    T momentum,
    T weight_decay,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T grad = gradients[idx] + weight_decay * weights[idx];
        T v = momentum * velocity[idx] + learning_rate * grad;
        velocity[idx] = v;
        weights[idx] -= v;
    }
}

// ============================================================================
// Adam Kernels
// ============================================================================

/**
 * Adam optimizer update kernel:
 * m = beta1 * m + (1 - beta1) * grad
 * v = beta2 * v + (1 - beta2) * grad^2
 * m_hat = m / (1 - beta1^t)
 * v_hat = v / (1 - beta2^t)
 * W -= lr * m_hat / (sqrt(v_hat) + eps)
 */
template<typename T>
__global__ void adam_update_kernel(
    T* weights,
    const T* gradients,
    T* m,  // First moment
    T* v,  // Second moment
    T learning_rate,
    T beta1,
    T beta2,
    T epsilon,
    T bias_correction1,  // 1 - beta1^t
    T bias_correction2,  // 1 - beta2^t
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T grad = gradients[idx];

        // Update biased first moment estimate
        T m_new = beta1 * m[idx] + (T(1) - beta1) * grad;
        m[idx] = m_new;

        // Update biased second raw moment estimate
        T v_new = beta2 * v[idx] + (T(1) - beta2) * grad * grad;
        v[idx] = v_new;

        // Compute bias-corrected estimates
        T m_hat = m_new / bias_correction1;
        T v_hat = v_new / bias_correction2;

        // Update parameters
        weights[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

/**
 * Adam with weight decay (AdamW):
 * W -= lr * weight_decay * W  (applied separately)
 * Then standard Adam update
 */
template<typename T>
__global__ void adamw_update_kernel(
    T* weights,
    const T* gradients,
    T* m,
    T* v,
    T learning_rate,
    T beta1,
    T beta2,
    T epsilon,
    T weight_decay,
    T bias_correction1,
    T bias_correction2,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T grad = gradients[idx];
        T w = weights[idx];

        // Decoupled weight decay
        w -= learning_rate * weight_decay * w;

        // Update biased first moment estimate
        T m_new = beta1 * m[idx] + (T(1) - beta1) * grad;
        m[idx] = m_new;

        // Update biased second raw moment estimate
        T v_new = beta2 * v[idx] + (T(1) - beta2) * grad * grad;
        v[idx] = v_new;

        // Compute bias-corrected estimates
        T m_hat = m_new / bias_correction1;
        T v_hat = v_new / bias_correction2;

        // Update parameters
        weights[idx] = w - learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

// ============================================================================
// RMSprop Kernels
// ============================================================================

/**
 * RMSprop optimizer update kernel:
 * cache = decay * cache + (1 - decay) * grad^2
 * W -= lr * grad / (sqrt(cache) + eps)
 */
template<typename T>
__global__ void rmsprop_update_kernel(
    T* weights,
    const T* gradients,
    T* cache,
    T learning_rate,
    T decay_rate,
    T epsilon,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T grad = gradients[idx];

        // Update cache
        T c = decay_rate * cache[idx] + (T(1) - decay_rate) * grad * grad;
        cache[idx] = c;

        // Update parameters
        weights[idx] -= learning_rate * grad / (sqrtf(c) + epsilon);
    }
}

/**
 * RMSprop with momentum:
 * cache = decay * cache + (1 - decay) * grad^2
 * v = momentum * v + lr * grad / (sqrt(cache) + eps)
 * W -= v
 */
template<typename T>
__global__ void rmsprop_momentum_kernel(
    T* weights,
    const T* gradients,
    T* cache,
    T* velocity,
    T learning_rate,
    T decay_rate,
    T momentum,
    T epsilon,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T grad = gradients[idx];

        // Update cache
        T c = decay_rate * cache[idx] + (T(1) - decay_rate) * grad * grad;
        cache[idx] = c;

        // Update velocity
        T v = momentum * velocity[idx] + learning_rate * grad / (sqrtf(c) + epsilon);
        velocity[idx] = v;

        // Update parameters
        weights[idx] -= v;
    }
}

// ============================================================================
// Launcher Functions
// ============================================================================

// Block size for all optimizer kernels
constexpr int OPTIMIZER_BLOCK_SIZE = 256;

/**
 * Launch SGD update kernel.
 */
template<typename T>
void launch_sgd_update(
    T* weights,
    const T* gradients,
    T learning_rate,
    size_t n,
    cudaStream_t stream
) {
    int blocks = (n + OPTIMIZER_BLOCK_SIZE - 1) / OPTIMIZER_BLOCK_SIZE;
    sgd_update_kernel<<<blocks, OPTIMIZER_BLOCK_SIZE, 0, stream>>>(
        weights, gradients, learning_rate, n);
}

/**
 * Launch gradient clipping kernel.
 */
template<typename T>
void launch_gradient_clip(
    T* gradients,
    T clip_value,
    size_t n,
    cudaStream_t stream
) {
    int blocks = (n + OPTIMIZER_BLOCK_SIZE - 1) / OPTIMIZER_BLOCK_SIZE;
    gradient_clip_kernel<<<blocks, OPTIMIZER_BLOCK_SIZE, 0, stream>>>(
        gradients, clip_value, n);
}

/**
 * Launch SGD with momentum update kernel.
 */
template<typename T>
void launch_sgd_momentum(
    T* weights,
    const T* gradients,
    T* velocity,
    T learning_rate,
    T momentum,
    size_t n,
    cudaStream_t stream
) {
    int blocks = (n + OPTIMIZER_BLOCK_SIZE - 1) / OPTIMIZER_BLOCK_SIZE;
    sgd_momentum_kernel<<<blocks, OPTIMIZER_BLOCK_SIZE, 0, stream>>>(
        weights, gradients, velocity, learning_rate, momentum, n);
}

/**
 * Launch Adam update kernel.
 */
template<typename T>
void launch_adam_update(
    T* weights,
    const T* gradients,
    T* m,
    T* v,
    T learning_rate,
    T beta1,
    T beta2,
    T epsilon,
    size_t timestep,
    size_t n,
    cudaStream_t stream
) {
    // Compute bias correction terms
    T beta1_t = powf(beta1, static_cast<T>(timestep));
    T beta2_t = powf(beta2, static_cast<T>(timestep));
    T bias_correction1 = T(1) - beta1_t;
    T bias_correction2 = T(1) - beta2_t;

    int blocks = (n + OPTIMIZER_BLOCK_SIZE - 1) / OPTIMIZER_BLOCK_SIZE;
    adam_update_kernel<<<blocks, OPTIMIZER_BLOCK_SIZE, 0, stream>>>(
        weights, gradients, m, v, learning_rate, beta1, beta2, epsilon,
        bias_correction1, bias_correction2, n);
}

/**
 * Launch RMSprop update kernel.
 */
template<typename T>
void launch_rmsprop_update(
    T* weights,
    const T* gradients,
    T* cache,
    T learning_rate,
    T decay_rate,
    T epsilon,
    size_t n,
    cudaStream_t stream
) {
    int blocks = (n + OPTIMIZER_BLOCK_SIZE - 1) / OPTIMIZER_BLOCK_SIZE;
    rmsprop_update_kernel<<<blocks, OPTIMIZER_BLOCK_SIZE, 0, stream>>>(
        weights, gradients, cache, learning_rate, decay_rate, epsilon, n);
}

// Explicit template instantiations for float
template void launch_sgd_update<float>(float*, const float*, float, size_t, cudaStream_t);
template void launch_gradient_clip<float>(float*, float, size_t, cudaStream_t);
template void launch_sgd_momentum<float>(float*, const float*, float*, float, float, size_t, cudaStream_t);
template void launch_adam_update<float>(float*, const float*, float*, float*, float, float, float, float, size_t, size_t, cudaStream_t);
template void launch_rmsprop_update<float>(float*, const float*, float*, float, float, float, size_t, cudaStream_t);

// Explicit template instantiations for double
template void launch_sgd_update<double>(double*, const double*, double, size_t, cudaStream_t);
template void launch_gradient_clip<double>(double*, double, size_t, cudaStream_t);
template void launch_sgd_momentum<double>(double*, const double*, double*, double, double, size_t, cudaStream_t);
template void launch_adam_update<double>(double*, const double*, double*, double*, double, double, double, double, size_t, size_t, cudaStream_t);
template void launch_rmsprop_update<double>(double*, const double*, double*, double, double, double, size_t, cudaStream_t);

} // namespace cuda
} // namespace dnn

#endif // DNN_ENABLE_CUDA
