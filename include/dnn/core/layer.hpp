#pragma once

#include "tensor.hpp"
#include "node.hpp"
#include "activations.hpp"
#include "initializers.hpp"
#include "random.hpp"
#include "device.hpp"
#include "../exceptions/dnn_exception.hpp"
#include "../simd/simd_ops.hpp"
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>

#ifdef DNN_ENABLE_CUDA
#include "../cuda/cuda_tensor.hpp"
#include "../cuda/cuda_ops.hpp"
#endif

namespace dnn {
namespace core {

/**
 * Layer types supported by the network.
 */
enum class LayerType {
    Dense,          // Fully connected
    Input,          // Input layer (no weights)
    Output          // Output layer
};

/**
 * Metrics for a layer.
 */
struct LayerMetrics {
    double avg_node_efficiency = 0.0;
    double min_node_efficiency = 1.0;
    double max_node_efficiency = 0.0;
    double output_variance = 0.0;
    double gradient_flow = 0.0;
    double computation_time_ms = 0.0;
    size_t active_nodes = 0;
    size_t total_nodes = 0;
    size_t trainable_nodes = 0;
    size_t dead_nodes = 0;
    size_t saturated_nodes = 0;
    bool is_healthy = true;
    std::string health_status = "normal";
};

/**
 * Dense (fully-connected) layer.
 * Template supports float and double precision.
 */
template<typename T = float>
class Layer {
public:
    // Minimum batch size before the OpenMP team-spawn cost is worth paying.
    // Single source of truth referenced by both forward_cpu and backward_cpu;
    // changing it here updates both per-batch parallel-for guards.
    static constexpr size_t kOpenMpBatchThreshold = 4;

    /**
     * Construct a dense layer.
     * @param input_size Number of inputs
     * @param output_size Number of outputs (neurons)
     * @param activation Activation function type
     * @param seed Random seed for initialization
     * @param device Device to run computations on (CPU or CUDA)
     */
    Layer(size_t input_size, size_t output_size,
          ActivationType activation = ActivationType::ReLU,
          uint64_t seed = 42,
          Device device = Device::CPU)
        : layer_type_(LayerType::Dense)
        , input_size_(input_size)
        , output_size_(output_size)
        , activation_type_(activation)
        , device_(device)
        , weights_(std::vector<size_t>{output_size, input_size})
        , biases_(std::vector<size_t>{output_size})
        , active_mask_(output_size, uint8_t(1))
        , rng_(std::make_unique<Random>(seed)) {

        // Initialize nodes
        nodes_.reserve(output_size);
        for (size_t i = 0; i < output_size; ++i) {
            nodes_.emplace_back(i, true);
        }

        // Create activation function
        activation_ = Activation<T>::create(activation);

        // Initialize weights using He initialization (good for ReLU)
        auto initializer = auto_initializer<T>(activation_->name(), input_size, output_size);
        initializer->initialize(weights_, *rng_);

        // Initialize biases to zero
        biases_.fill(T(0));

        // If CUDA requested, move tensors to GPU
        if (device_ == Device::CUDA) {
            to_device(Device::CUDA);
        }
    }

    /**
     * Move layer to specified device.
     * @param device Target device (CPU or CUDA)
     */
    void to_device(Device device) {
        if (device == device_) return;  // Already on target device

        device_ = device;

#ifdef DNN_ENABLE_CUDA
        if (device == Device::CUDA) {
            // Lazy: forward_cuda_dev rebuilds gpu_weights_ / gpu_biases_ on
            // first use. We just drop any stale state here.
            invalidate_gpu_mirrors_();
        } else {
            // Move back to CPU - weights/biases should already be updated
            if (gpu_weights_) {
                weights_ = gpu_weights_->to_host();
            }
            if (gpu_biases_) {
                biases_ = gpu_biases_->to_host();
            }
            invalidate_gpu_mirrors_();
        }
#else
        if (device == Device::CUDA) {
            throw std::runtime_error("CUDA not enabled. Rebuild with DNN_ENABLE_CUDA=ON");
        }
#endif
    }

    /**
     * Get current device.
     */
    Device device() const { return device_; }

    /**
     * Forward pass.
     */
    Tensor<T> forward(const Tensor<T>& input) {
#ifdef DNN_ENABLE_CUDA
        if (device_ == Device::CUDA) {
            return forward_cuda(input);
        }
#endif
        return forward_cpu(input);
    }

#ifdef DNN_ENABLE_CUDA
    /**
     * Device-resident forward. Accepts a GPU tensor, returns a GPU tensor.
     * Used by Network::forward when device_ == CUDA so the whole layer
     * chain runs with only one host↔device transfer at the network
     * boundary. Caches input/pre-activation/output on device for backward.
     *
     * Input is consumed (moved) into cached_input_gpu_. Caller must not
     * reuse the passed tensor after this call.
     */
    cuda::CudaTensor<T> forward_cuda_dev(cuda::CudaTensor<T> gpu_input) {
        // Lazy upload of weights/biases on first use or after invalidation.
        if (!gpu_weights_) {
            gpu_weights_ = std::make_unique<cuda::CudaTensor<T>>(weights_);
        }
        if (!gpu_biases_) {
            gpu_biases_ = std::make_unique<cuda::CudaTensor<T>>(biases_);
        }
        ensure_gpu_active_mask_();

        const size_t rank = gpu_input.ndim();
        if (rank != 1 && rank != 2) {
            throw exceptions::ShapeException("forward_cuda_dev",
                "Input must be 1D or 2D");
        }
        const size_t batch_size = (rank == 2) ? gpu_input.shape()[0] : 1;
        const size_t in_size = (rank == 2) ? gpu_input.shape()[1] : gpu_input.shape()[0];
        // The CPU forward_cpu silently truncates when the upstream layer
        // grew its output_size_ past this layer's input_size_ (architecture
        // mutations don't update adjacent layers — see
        // include/dnn/dynamics/CLAUDE.md, fan-in repair happens only at
        // compact() time). Match that on GPU by slicing the input to the
        // first input_size_ columns when in_size > input_size_. For
        // in_size < input_size_ we error (same as CPU's OOB UB).
        if (in_size < input_size_) {
            throw exceptions::ShapeException("forward_cuda_dev",
                "Input width " + std::to_string(in_size) +
                " < layer input_size_ " + std::to_string(input_size_));
        }
        if (in_size > input_size_) {
            cuda::CudaTensor<T> sliced(rank == 2
                ? std::vector<size_t>{batch_size, input_size_}
                : std::vector<size_t>{input_size_});
            // strided device-to-device copy: take the first input_size_ floats
            // of each "row" (or just the head for rank-1).
            cudaMemcpy2D(sliced.device_data(),
                         input_size_ * sizeof(T),       // dst pitch
                         gpu_input.device_data(),
                         in_size * sizeof(T),           // src pitch
                         input_size_ * sizeof(T),       // bytes per row
                         batch_size,                    // num rows
                         cudaMemcpyDeviceToDevice);
            gpu_input = std::move(sliced);
        }

        std::vector<size_t> output_shape = (rank == 2)
            ? std::vector<size_t>{batch_size, output_size_}
            : std::vector<size_t>{output_size_};

        // Cache the input by taking ownership.
        cached_input_gpu_ = std::make_unique<cuda::CudaTensor<T>>(std::move(gpu_input));

        // Compute pre-activation: pre = X @ W^T + b   (or W @ x + b for rank-1)
        cuda::CudaTensor<T> pre_act(output_shape);
        if (rank == 1) {
            // y = W @ x  (gemv with bias added via broadcast below)
            cuda::cuda_gemv(*gpu_weights_, *cached_input_gpu_, pre_act,
                            T(1), T(0), /*transpose=*/false);
            // y += b
            cuda::cuda_add(pre_act, *gpu_biases_, pre_act);
        } else {
            // Y = X @ W^T
            cuda::cuda_gemm(*cached_input_gpu_, *gpu_weights_, pre_act,
                            T(1), T(0),
                            /*transpose_A=*/false, /*transpose_B=*/true);
            // Y[b, o] += biases[o] — on-device broadcast, no host loop.
            cuda::cuda_add_bias_broadcast(pre_act, *gpu_biases_);
        }

        // Apply active mask (zeroes inactive output columns). On-device,
        // replaces the old D2H→zero→H2D triplet.
        cuda::cuda_apply_mask_broadcast(pre_act, *gpu_active_mask_);

        // Record per-node activation metrics from the first batch row.
        // One D2H of output_size_ floats per layer per step, not the full
        // batch tensor — see CLAUDE.md "Cheap per-node metric recording".
        record_first_row_activation_gpu_(pre_act);

        // Apply activation on device.
        cuda::CudaTensor<T> activated(output_shape);
        if (!apply_activation_gpu_(pre_act, activated)) {
            // Activation has no GPU kernel (e.g. legacy fallthrough). D2H,
            // apply on host, H2D. Slow path used only when missing kernel.
            Tensor<T> pre_host = pre_act.to_host();
            Tensor<T> act_host = activation_->forward(pre_host);
            activated.to_device(act_host);
        }

        // Cache for backward.
        cached_pre_activation_gpu_ = std::make_unique<cuda::CudaTensor<T>>(
            std::move(pre_act));
        // We need cached_output_gpu_ AND to return a separate tensor (since
        // CudaTensor is move-only and the caller takes ownership). Allocate
        // a copy via cuda_copy.
        cached_output_gpu_ = std::make_unique<cuda::CudaTensor<T>>(output_shape);
        cuda::cuda_copy(activated, *cached_output_gpu_);
        return activated;
    }

    /**
     * Device-resident backward. Accepts GPU grad_output, returns GPU
     * grad_input. Uses the GPU-resident caches populated by
     * forward_cuda_dev. Accumulates weight + bias gradients into the host
     * weight_gradients_/bias_gradients_ via a single dW + db D2H per layer
     * per step (much cheaper than the old per-layer round-trip storm).
     */
    cuda::CudaTensor<T> backward_cuda_dev(const cuda::CudaTensor<T>& gpu_grad_output) {
        if (!cached_input_gpu_ || !cached_pre_activation_gpu_ || !cached_output_gpu_) {
            throw std::runtime_error(
                "backward_cuda_dev: forward_cuda_dev was not called first");
        }
        ensure_gpu_active_mask_();

        const size_t rank = cached_input_gpu_->ndim();
        const size_t batch_size = (rank == 2) ? cached_input_gpu_->shape()[0] : 1;

        // Match grad_output shape to our cached pre_activation. After
        // architecture mutations grew this layer's output_size_, the next
        // layer may return a narrower grad_input than our output_size_;
        // pad with zeros so the activation backward kernel sees matching
        // shapes. (Mirrors how the forward path slices oversized inputs.)
        const size_t grad_in_size = (rank == 2)
            ? gpu_grad_output.shape()[1] : gpu_grad_output.shape()[0];
        std::vector<size_t> full_shape = (rank == 2)
            ? std::vector<size_t>{batch_size, output_size_}
            : std::vector<size_t>{output_size_};
        cuda::CudaTensor<T> grad_out_padded(full_shape);
        if (grad_in_size == output_size_) {
            cuda::cuda_copy(gpu_grad_output, grad_out_padded);
        } else if (grad_in_size < output_size_) {
            // zero-fill, then copy the available cols.
            cuda::cuda_fill(grad_out_padded, T(0));
            cudaMemcpy2D(grad_out_padded.device_data(),
                         output_size_ * sizeof(T),
                         gpu_grad_output.device_data(),
                         grad_in_size * sizeof(T),
                         grad_in_size * sizeof(T),
                         batch_size,
                         cudaMemcpyDeviceToDevice);
        } else {
            // grad wider than our output (shouldn't happen, but slice
            // defensively to first output_size_ cols).
            cudaMemcpy2D(grad_out_padded.device_data(),
                         output_size_ * sizeof(T),
                         gpu_grad_output.device_data(),
                         grad_in_size * sizeof(T),
                         output_size_ * sizeof(T),
                         batch_size,
                         cudaMemcpyDeviceToDevice);
        }

        // grad_activation = d/dz of activation, applied to grad_output.
        cuda::CudaTensor<T> grad_activation(full_shape);
        if (!apply_activation_backward_gpu_(*cached_pre_activation_gpu_,
                                            *cached_output_gpu_,
                                            grad_out_padded, grad_activation)) {
            // Slow path: D2H, run host activation backward, H2D.
            Tensor<T> pre_host = cached_pre_activation_gpu_->to_host();
            Tensor<T> out_host = cached_output_gpu_->to_host();
            Tensor<T> grad_out_host = grad_out_padded.to_host();
            Tensor<T> grad_act_host = activation_->backward(
                pre_host, out_host, grad_out_host);
            grad_activation.to_device(grad_act_host);
        }

        // Zero gradients for inactive nodes (mask multiply, on-device).
        cuda::cuda_apply_mask_broadcast(grad_activation, *gpu_active_mask_);

        // Record gradient metrics from first batch row (one tiny D2H).
        record_first_row_gradient_gpu_(grad_activation);

        if (rank == 1) {
            // Single-sample path (init pass only — not the trainer hot path).
            // Outer product + bias accumulation done on host after one D2H.
            Tensor<T> grad_act_host = grad_activation.to_host();
            Tensor<T> input_host = cached_input_gpu_->to_host();
            for (size_t i = 0; i < output_size_; ++i) {
                if (!is_node_active(i)) continue;
                if (nodes_[i].is_trainable()) {
                    for (size_t j = 0; j < input_size_; ++j) {
                        weight_gradients_.at(i, j) +=
                            grad_act_host[i] * input_host[j];
                    }
                    bias_gradients_[i] += grad_act_host[i];
                }
            }
            // grad_input = W^T @ grad_activation
            cuda::CudaTensor<T> grad_input(std::vector<size_t>{input_size_});
            cuda::cuda_gemv(*gpu_weights_, grad_activation, grad_input,
                            T(1), T(0), /*transpose=*/true);
            return grad_input;
        }

        // Batched (rank-2) hot path. Everything stays on device — no D2H.
        // dW = grad_activation^T @ cached_input  -> (O, I)
        cuda::CudaTensor<T> dW(std::vector<size_t>{output_size_, input_size_});
        cuda::cuda_gemm(grad_activation, *cached_input_gpu_, dW,
                        T(1), T(0),
                        /*transpose_A=*/true, /*transpose_B=*/false);

        // db = sum(grad_activation, axis=0) -> (O,)
        cuda::CudaTensor<T> db(std::vector<size_t>{output_size_});
        cuda::cuda_sum_axis(grad_activation, db, /*axis=*/0);

        // Apply the active+trainable mask on device, then accumulate into the
        // GPU-resident gradient buffers. Zeros rows of dW (and elements of db)
        // for nodes that are inactive (soft-removed) or not trainable, so the
        // subsequent on-device SGD step naturally leaves their weights alone.
        ensure_gpu_update_mask_();
        cuda::cuda_apply_mask_rows_broadcast(dW, *gpu_update_mask_);
        // db (O,) and mask (O,): apply_mask_broadcast treats this as
        // B=1, O=output_size_ — element-wise multiply, exactly what we want.
        cuda::cuda_apply_mask_broadcast(db, *gpu_update_mask_);

        if (!gpu_weight_gradients_) {
            gpu_weight_gradients_ = std::make_unique<cuda::CudaTensor<T>>(
                std::vector<size_t>{output_size_, input_size_});
            cuda::cuda_fill(*gpu_weight_gradients_, T(0));
        }
        if (!gpu_bias_gradients_) {
            gpu_bias_gradients_ = std::make_unique<cuda::CudaTensor<T>>(
                std::vector<size_t>{output_size_});
            cuda::cuda_fill(*gpu_bias_gradients_, T(0));
        }
        cuda::cuda_add(*gpu_weight_gradients_, dW, *gpu_weight_gradients_);
        cuda::cuda_add(*gpu_bias_gradients_, db, *gpu_bias_gradients_);

        // grad_input = grad_activation @ W   -> (B, I)
        cuda::CudaTensor<T> grad_input(std::vector<size_t>{batch_size, input_size_});
        cuda::cuda_gemm(grad_activation, *gpu_weights_, grad_input,
                        T(1), T(0),
                        /*transpose_A=*/false, /*transpose_B=*/false);
        return grad_input;
    }
#endif

private:
    /**
     * CPU forward pass implementation.
     */
    Tensor<T> forward_cpu(const Tensor<T>& input) {
        // Cache input for backward pass
        cached_input_ = input.clone();

        // Linear transformation: output = input @ weights^T + biases
        Tensor<T> linear_output;

        if (input.rank() == 1) {
            // Single-sample path. Stays scalar — gemv is rarely the hot
            // path (the trainer's batched_train_forward routes through
            // rank-2). Keeps per-node activation recording inline.
            linear_output = Tensor<T>(std::vector<size_t>{output_size_});
            for (size_t i = 0; i < output_size_; ++i) {
                if (!is_node_active(i)) {
                    linear_output[i] = T(0);
                    continue;
                }
                T sum = biases_[i];
                for (size_t j = 0; j < input_size_; ++j) {
                    sum += input[j] * weights_.at(i, j);
                }
                linear_output[i] = sum;
                nodes_[i].record_activation(static_cast<float>(sum));
            }
        } else if (input.rank() == 2) {
            // Batched path. Route through SIMDOps::matmul (blocked, AVX/SSE
            // vectorised, OpenMP-parallel on the outer row block). The
            // matmul needs row-major operands, so we build a transposed
            // weights tensor (input_size, output_size) on the stack each
            // forward. Cost is O(O*I); the matmul is O(B*O*I) so this is
            // amortised. Local (not cached) to avoid mutable-state races
            // under the concurrent runtime path.
            size_t batch_size = input.shape()[0];
            linear_output = Tensor<T>(std::vector<size_t>{batch_size, output_size_});

            Tensor<T> weights_t(std::vector<size_t>{input_size_, output_size_});
            for (size_t i = 0; i < output_size_; ++i) {
                for (size_t j = 0; j < input_size_; ++j) {
                    weights_t.at(j, i) = weights_.at(i, j);
                }
            }

            // Y(B, O) = X(B, I) @ W_t(I, O)
            get_simd_ops_()->matmul(
                input.data(),
                weights_t.data(),
                linear_output.data(),
                /*m=*/batch_size,
                /*k=*/input_size_,
                /*n=*/output_size_,
                /*lda=*/input_size_,
                /*ldb=*/output_size_,
                /*ldc=*/output_size_);

            // Add bias broadcast: Y[b, o] += biases[o].
            for (size_t b = 0; b < batch_size; ++b) {
                T* row = linear_output.data() + b * output_size_;
                const T* bias = biases_.data();
                for (size_t i = 0; i < output_size_; ++i) row[i] += bias[i];
            }

            // Post-zero inactive output columns (soft-topology contract).
            // The matmul above is dense; columns for inactive neurons must
            // be cleared so downstream layers and the active-mask
            // accounting see zero. Cost is O(B * inactive_count); active
            // nodes pay nothing.
            for (size_t i = 0; i < output_size_; ++i) {
                if (is_node_active(i)) continue;
                for (size_t b = 0; b < batch_size; ++b) {
                    linear_output.at(b, i) = T(0);
                }
            }

            // NOTE: per-node activation recording (nodes_[i].record_activation)
            // is intentionally omitted in the rank-2 batched path to match the
            // legacy CPU rank-2 baseline. Per-sample metrics come from the
            // trainer's init forward pass (rank-1) at trainer.hpp:418.
        } else {
            throw exceptions::ShapeException("forward", "Input must be 1D or 2D");
        }

        // Cache pre-activation for backward pass
        cached_pre_activation_ = linear_output.clone();

        // Apply activation
        cached_output_ = activation_->forward(linear_output);
        return cached_output_.clone();
    }

#ifdef DNN_ENABLE_CUDA
    /**
     * CUDA forward pass entrypoint for legacy Tensor-in/Tensor-out callers.
     * The hot path goes through Network::forward → forward_cuda_dev
     * directly; this wrapper only fires when something calls
     * Layer::forward(Tensor) on a CUDA layer in isolation (tests, ad-hoc
     * inference). Does one H2D up front and one D2H at the end.
     */
    Tensor<T> forward_cuda(const Tensor<T>& input) {
        cuda::CudaTensor<T> gpu_input(input);
        cuda::CudaTensor<T> gpu_output = forward_cuda_dev(std::move(gpu_input));
        return gpu_output.to_host();
    }

    /**
     * Apply the layer's activation on device. Returns false when there is
     * no GPU kernel for activation_type_ — caller falls back to host.
     */
    bool apply_activation_gpu_(const cuda::CudaTensor<T>& x,
                                cuda::CudaTensor<T>& y) {
        switch (activation_type_) {
            case ActivationType::ReLU:    cuda::cuda_relu(x, y);    return true;
            case ActivationType::Sigmoid: cuda::cuda_sigmoid(x, y); return true;
            case ActivationType::Tanh:    cuda::cuda_tanh(x, y);    return true;
            case ActivationType::Softmax: cuda::cuda_softmax(x, y); return true;
            default: return false;
        }
    }

    /**
     * Apply the layer's activation backward on device. ReLU/Sigmoid/
     * Tanh/Softmax run on-device; returns false only for activations
     * with no GPU backward kernel (host fallback).
     */
    bool apply_activation_backward_gpu_(const cuda::CudaTensor<T>& pre_act,
                                         const cuda::CudaTensor<T>& post_act,
                                         const cuda::CudaTensor<T>& grad_out,
                                         cuda::CudaTensor<T>& grad_in) {
        switch (activation_type_) {
            case ActivationType::ReLU:
                cuda::cuda_relu_backward(pre_act, grad_out, grad_in);
                return true;
            case ActivationType::Sigmoid:
                cuda::cuda_sigmoid_backward(post_act, grad_out, grad_in);
                return true;
            case ActivationType::Tanh:
                cuda::cuda_tanh_backward(post_act, grad_out, grad_in);
                return true;
            case ActivationType::Softmax:
                cuda::cuda_softmax_backward(post_act, grad_out, grad_in);
                return true;
            default:
                return false;
        }
    }

    /**
     * D2H one row of (B, O) pre-activation (or all of rank-1) for cheap
     * per-node activation metric recording. Costs output_size_ floats per
     * layer per forward — negligible vs. a full activation D2H.
     */
    void record_first_row_activation_gpu_(const cuda::CudaTensor<T>& pre_act) {
        const size_t rank = pre_act.ndim();
        const size_t row_size = (rank == 2) ? pre_act.shape()[1] : pre_act.size();
        if (row_size != output_size_) return;
        // Row 0 is the first row_size contiguous elements (row-major) for
        // both rank-1 and rank-2; copy exactly those instead of D2H-ing
        // the whole (B, O) tensor.
        Tensor<T> host_row(std::vector<size_t>{row_size});
        pre_act.copy_to_host_strided(host_row.data(), 0, row_size);
        for (size_t i = 0; i < output_size_; ++i) {
            nodes_[i].record_activation(static_cast<float>(host_row[i]));
        }
    }

    /**
     * Same idea for the gradient backward pass.
     */
    void record_first_row_gradient_gpu_(const cuda::CudaTensor<T>& grad_act) {
        const size_t rank = grad_act.ndim();
        const size_t row_size = (rank == 2) ? grad_act.shape()[1] : grad_act.size();
        if (row_size != output_size_) return;
        Tensor<T> host_row(std::vector<size_t>{row_size});
        grad_act.copy_to_host_strided(host_row.data(), 0, row_size);
        for (size_t i = 0; i < output_size_; ++i) {
            nodes_[i].record_gradient(static_cast<float>(host_row[i]));
        }
    }
#endif

public:

    /**
     * Backward pass.
     * @param grad_output Gradient from next layer
     * @return Gradient with respect to input
     */
    Tensor<T> backward(const Tensor<T>& grad_output) {
#ifdef DNN_ENABLE_CUDA
        if (device_ == Device::CUDA) {
            return backward_cuda(grad_output);
        }
#endif
        return backward_cpu(grad_output);
    }

private:
    /**
     * CPU backward pass implementation.
     */
    Tensor<T> backward_cpu(const Tensor<T>& grad_output) {
        // Compute activation gradient
        Tensor<T> grad_activation = activation_->backward(
            cached_pre_activation_, cached_output_, grad_output);

        // Zero gradients for inactive (dormant) nodes so their parameters
        // don't drift and no gradient flows upstream through them.
        zero_inactive_gradients(grad_activation);

        // Compute gradients for weights and biases
        if (cached_input_.rank() == 1) {
            // Single sample
            for (size_t i = 0; i < output_size_; ++i) {
                if (!is_node_active(i)) continue;
                if (nodes_[i].is_trainable()) {
                    // Weight gradient: outer product
                    for (size_t j = 0; j < input_size_; ++j) {
                        weight_gradients_.at(i, j) += grad_activation[i] * cached_input_[j];
                    }
                    // Bias gradient
                    bias_gradients_[i] += grad_activation[i];
                }

                // Record gradient for metrics
                nodes_[i].record_gradient(static_cast<float>(grad_activation[i]));
            }

            // Compute input gradient: grad_input = weights^T @ grad_activation
            Tensor<T> grad_input(std::vector<size_t>{input_size_});
            for (size_t j = 0; j < input_size_; ++j) {
                T sum = T(0);
                for (size_t i = 0; i < output_size_; ++i) {
                    sum += weights_.at(i, j) * grad_activation[i];
                }
                grad_input[j] = sum;
            }
            return grad_input;

        } else if (cached_input_.rank() == 2) {
            // Batched path. Both matmuls go through SIMDOps::matmul so we
            // get SIMD vectorisation + OpenMP-parallel-on-output-row from
            // a single call. weight-grad accumulation no longer runs
            // serially over the batch — the matmul itself parallelises.
            size_t batch_size = cached_input_.shape()[0];
            simd::SIMDOps<T>* ops = get_simd_ops_();

            // 1. Weight gradient:
            //    dW (O, I) += grad_activation^T (O, B) @ cached_input (B, I)
            //    Materialize grad_activation^T into a small scratch
            //    (O × B floats; cheap for typical batch sizes).
            Tensor<T> grad_act_t(std::vector<size_t>{output_size_, batch_size});
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t i = 0; i < output_size_; ++i) {
                    grad_act_t.at(i, b) = grad_activation.at(b, i);
                }
            }
            Tensor<T> dW_scratch(std::vector<size_t>{output_size_, input_size_}, T(0));
            ops->matmul(
                grad_act_t.data(),
                cached_input_.data(),
                dW_scratch.data(),
                /*m=*/output_size_,
                /*k=*/batch_size,
                /*n=*/input_size_,
                /*lda=*/batch_size,
                /*ldb=*/input_size_,
                /*ldc=*/input_size_);

            // Accumulate into weight_gradients_ filtered by active +
            // trainable. Bias gradient = sum along batch dim.
            for (size_t i = 0; i < output_size_; ++i) {
                if (!is_node_active(i)) continue;
                if (nodes_[i].is_trainable()) {
                    for (size_t j = 0; j < input_size_; ++j) {
                        weight_gradients_.at(i, j) += dW_scratch.at(i, j);
                    }
                    T bias_sum = T(0);
                    for (size_t b = 0; b < batch_size; ++b) {
                        bias_sum += grad_activation.at(b, i);
                    }
                    bias_gradients_[i] += bias_sum;
                }
            }

            // NOTE: per-node gradient recording omitted in rank-2 path —
            // legacy baseline also omitted it. Trainer init pass (rank-1)
            // populates node gradient metrics.

            // 2. Input gradient:
            //    grad_input (B, I) = grad_activation (B, O) @ weights (O, I)
            //    Note weights_ is laid out (output_size, input_size) row-
            //    major, so it's the operand exactly — no transpose needed.
            Tensor<T> grad_input(std::vector<size_t>{batch_size, input_size_});
            ops->matmul(
                grad_activation.data(),
                weights_.data(),
                grad_input.data(),
                /*m=*/batch_size,
                /*k=*/output_size_,
                /*n=*/input_size_,
                /*lda=*/output_size_,
                /*ldb=*/input_size_,
                /*ldc=*/input_size_);

            return grad_input;
        }

        throw exceptions::ShapeException("backward", "Cached input has invalid shape");
    }

#ifdef DNN_ENABLE_CUDA
    /**
     * CUDA backward entrypoint for Tensor-in/Tensor-out callers. The hot
     * path is Network::backward → backward_cuda_dev directly. This wrapper
     * runs one H2D + one D2H for ad-hoc Tensor-only callers.
     *
     * Requires forward_cuda (or forward_cuda_dev) to have been called
     * first so the GPU-resident caches are populated.
     */
    Tensor<T> backward_cuda(const Tensor<T>& grad_output) {
        cuda::CudaTensor<T> gpu_grad(grad_output);
        cuda::CudaTensor<T> gpu_grad_input = backward_cuda_dev(gpu_grad);
        return gpu_grad_input.to_host();
    }
#endif

public:

    /**
     * Apply accumulated gradients and update weights.
     * @param learning_rate Learning rate for update
     */
    /**
     * Clamp every accumulated gradient element to [-clip_value,
     * +clip_value]. Clips the same buffers apply_gradients() will
     * consume: the GPU gradient mirrors on the CUDA path, the host
     * accumulators on the CPU path.
     */
    void clip_gradients(T clip_value) {
        if (clip_value <= T(0)) return;
#ifdef DNN_ENABLE_CUDA
        if (device_ == Device::CUDA && gpu_weight_gradients_
            && gpu_bias_gradients_) {
            cuda::launch_gradient_clip(
                static_cast<T*>(gpu_weight_gradients_->device_data()),
                clip_value, gpu_weight_gradients_->size());
            cuda::launch_gradient_clip(
                static_cast<T*>(gpu_bias_gradients_->device_data()),
                clip_value, gpu_bias_gradients_->size());
            return;
        }
#endif
        if (!weight_gradients_.empty()) {
            T* w = weight_gradients_.data();
            for (size_t k = 0; k < weight_gradients_.size(); ++k) {
                w[k] = std::max(-clip_value, std::min(clip_value, w[k]));
            }
        }
        if (!bias_gradients_.empty()) {
            T* b = bias_gradients_.data();
            for (size_t k = 0; k < bias_gradients_.size(); ++k) {
                b[k] = std::max(-clip_value, std::min(clip_value, b[k]));
            }
        }
    }

    /**
     * Add zero-mean Gaussian noise (stddev = `stddev`) to every weight
     * in the given active node's row. Used by the Phase-3 random
     * perturbation. Does not touch inactive nodes. Caller must invoke
     * mark_weights_dirty() afterwards so the GPU mirrors reupload.
     */
    void perturb_node(size_t node_idx, T stddev, Random& rng) {
        if (node_idx >= output_size_ || stddev <= T(0)) return;
        if (!is_node_active(node_idx)) return;
        for (size_t j = 0; j < input_size_; ++j) {
            weights_.at(node_idx, j) += rng.template normal<T>(T(0), stddev);
        }
    }

    /**
     * Invalidate device-resident weight/bias mirrors after a direct
     * host-side weight mutation (e.g. perturbation) so the next forward
     * reuploads.
     */
    void mark_weights_dirty() {
#ifdef DNN_ENABLE_CUDA
        invalidate_gpu_mirrors_();
#endif
    }

    void apply_gradients(T learning_rate) {
#ifdef DNN_ENABLE_CUDA
        if (device_ == Device::CUDA && gpu_weights_ && gpu_biases_
            && gpu_weight_gradients_ && gpu_bias_gradients_) {
            // On-device SGD: weights -= lr * dW, biases -= lr * db. Then
            // zero the GPU gradient buffers for the next batch. The
            // active+trainable mask was already applied during accumulation
            // in backward_cuda_dev, so we don't need to re-mask here.
            cuda::cuda_axpy(*gpu_weights_, -learning_rate, *gpu_weight_gradients_);
            cuda::cuda_axpy(*gpu_biases_, -learning_rate, *gpu_bias_gradients_);
            cuda::cuda_fill(*gpu_weight_gradients_, T(0));
            cuda::cuda_fill(*gpu_bias_gradients_, T(0));
            // weights_ / biases_ on the host are now stale. Mark them so
            // any code path that needs fresh host values (compact, model
            // serialization, has_numerical_issues) calls sync_host_weights_
            // before reading. The CPU forward path is not reachable from a
            // CUDA layer, so during training this never matters.
            host_weights_dirty_ = true;
            return;
        }
#endif

        // CPU SGD: in-place update of host weights/biases.
        for (size_t i = 0; i < output_size_; ++i) {
            if (!is_node_active(i)) continue;
            if (nodes_[i].is_trainable()) {
                for (size_t j = 0; j < input_size_; ++j) {
                    weights_.at(i, j) -= learning_rate * weight_gradients_.at(i, j);
                }
                biases_[i] -= learning_rate * bias_gradients_[i];
            }
        }

#ifdef DNN_ENABLE_CUDA
        // GPU mirrors (if any) reference the old values; drop so the next
        // forward_cuda_dev reuploads. Only reached when CUDA is enabled but
        // we fell through the CUDA path above (no GPU gradients yet, e.g.
        // first batch after a topology mutation invalidated the mirrors).
        gpu_weights_.reset();
        gpu_biases_.reset();
#endif

        // Reset host gradient accumulators.
        zero_gradients();
    }

    /**
     * Zero out accumulated gradients.
     */
    void zero_gradients() {
        if (weight_gradients_.empty()) {
            weight_gradients_ = Tensor<T>(std::vector<size_t>{output_size_, input_size_}, T(0));
            bias_gradients_ = Tensor<T>(std::vector<size_t>{output_size_}, T(0));
        } else {
            weight_gradients_.fill(T(0));
            bias_gradients_.fill(T(0));
        }
    }

    // Accessors
    LayerType layer_type() const { return layer_type_; }
    size_t input_size() const { return input_size_; }
    size_t output_size() const { return output_size_; }
    size_t num_nodes() const { return nodes_.size(); }
    ActivationType activation_type() const { return activation_type_; }

    Tensor<T>& weights() { return weights_; }
    const Tensor<T>& weights() const { return weights_; }
    Tensor<T>& biases() { return biases_; }
    const Tensor<T>& biases() const { return biases_; }
    Tensor<T>& weight_gradients() { return weight_gradients_; }
    const Tensor<T>& weight_gradients() const { return weight_gradients_; }
    Tensor<T>& bias_gradients() { return bias_gradients_; }
    const Tensor<T>& bias_gradients() const { return bias_gradients_; }

    std::vector<Node>& nodes() { return nodes_; }
    const std::vector<Node>& nodes() const { return nodes_; }
    Node& node(size_t idx) { return nodes_[idx]; }
    const Node& node(size_t idx) const { return nodes_[idx]; }

    /**
     * Count trainable nodes.
     */
    size_t trainable_count() const {
        return std::count_if(nodes_.begin(), nodes_.end(),
            [](const Node& n) { return n.is_trainable(); });
    }

    /**
     * Set fraction of nodes to be trainable.
     * Keeps the least efficient nodes trainable (they need more training).
     */
    void set_trainable_fraction(double fraction) {
        fraction = std::max(0.0, std::min(1.0, fraction));
        size_t target_trainable = static_cast<size_t>(fraction * nodes_.size());
        target_trainable = std::max(size_t(1), target_trainable);

        // Get efficiency scores
        std::vector<std::pair<double, size_t>> efficiencies;
        for (size_t i = 0; i < nodes_.size(); ++i) {
            efficiencies.emplace_back(nodes_[i].efficiency_score(), i);
        }

        // Sort by efficiency (ascending - least efficient first)
        std::sort(efficiencies.begin(), efficiencies.end());

        // Set trainability - least efficient nodes stay trainable
        for (size_t i = 0; i < nodes_.size(); ++i) {
            size_t node_idx = efficiencies[i].second;
            nodes_[node_idx].set_trainable(i < target_trainable);
        }
        // gpu_update_mask_ encodes (active && is_trainable); trainability
        // just changed, so the mask must be rebuilt on next backward.
#ifdef DNN_ENABLE_CUDA
        gpu_update_mask_.reset();
#endif
    }

    /**
     * Compute layer metrics.
     */
    LayerMetrics compute_metrics() const {
        LayerMetrics metrics;
        metrics.total_nodes = nodes_.size();
        metrics.trainable_nodes = trainable_count();

        double efficiency_sum = 0.0;
        size_t counted = 0;
        for (size_t i = 0; i < nodes_.size(); ++i) {
            // Dormant nodes are excluded from health metrics; they're effectively
            // not part of the running model until reactivated or compacted away.
            if (!is_node_active(i)) continue;
            const auto& node = nodes_[i];
            auto node_metrics = node.compute_metrics();
            double eff = node_metrics.efficiency_score;

            efficiency_sum += eff;
            metrics.min_node_efficiency = std::min(metrics.min_node_efficiency, eff);
            metrics.max_node_efficiency = std::max(metrics.max_node_efficiency, eff);

            if (node.is_dead()) metrics.dead_nodes++;
            if (node.is_saturated()) metrics.saturated_nodes++;
            if (node_metrics.status == "normal") metrics.active_nodes++;
            ++counted;
        }

        metrics.avg_node_efficiency = counted > 0 ? efficiency_sum / counted : 0.0;

        // Compute output variance from cached output
        if (!cached_output_.empty()) {
            metrics.output_variance = cached_output_.variance();
        }

        // Determine health status
        if (metrics.dead_nodes > nodes_.size() / 2) {
            metrics.is_healthy = false;
            metrics.health_status = "many_dead_nodes";
        } else if (metrics.saturated_nodes > nodes_.size() / 2) {
            metrics.is_healthy = false;
            metrics.health_status = "many_saturated_nodes";
        } else if (metrics.avg_node_efficiency < 0.2) {
            metrics.is_healthy = false;
            metrics.health_status = "low_efficiency";
        } else {
            metrics.is_healthy = true;
            metrics.health_status = "healthy";
        }

        return metrics;
    }

    /**
     * Update node efficiency scores.
     */
    void update_node_efficiency() {
        for (auto& node : nodes_) {
            node.compute_metrics();
        }
    }

    /**
     * Reset node metrics for new training phase.
     */
    void reset_node_metrics() {
        for (auto& node : nodes_) {
            node.reset_metrics();
        }
    }

    /**
     * Compute variance z-scores and adapt variance weights for all nodes.
     * Should be called once before training starts.
     */
    void compute_initial_variance_zscores() {
        if (nodes_.empty()) return;

        // Collect variances of active nodes only. Inactive (soft-removed)
        // nodes never see activations, so their default-zero variance
        // would skew the mean/std and make active nodes look like outliers.
        std::vector<size_t> active_idx;
        std::vector<double> variances;
        active_idx.reserve(nodes_.size());
        variances.reserve(nodes_.size());

        for (size_t i = 0; i < nodes_.size(); ++i) {
            if (!is_node_active(i)) continue;
            active_idx.push_back(i);
            variances.push_back(nodes_[i].compute_metrics().activation_variance);
        }

        if (variances.empty()) return;

        // Compute mean
        double sum = 0.0;
        for (double v : variances) {
            sum += v;
        }
        double mean = sum / variances.size();

        // Compute standard deviation
        double sq_sum = 0.0;
        for (double v : variances) {
            sq_sum += (v - mean) * (v - mean);
        }
        double std_dev = std::sqrt(sq_sum / variances.size());

        // Avoid division by zero
        if (std_dev < 1e-8) {
            std_dev = 1e-8;
        }

        // Compute z-scores and adapt variance weights (active nodes only)
        for (size_t k = 0; k < active_idx.size(); ++k) {
            double z = (variances[k] - mean) / std_dev;
            nodes_[active_idx[k]].adapt_variance_weight(z);
        }
    }

    /**
     * Compute gradient threshold from current gradient statistics.
     * Returns mean + std of gradient magnitudes.
     */
    double compute_gradient_threshold() const {
        if (nodes_.empty()) return 0.1;

        // Active nodes only: inactive nodes have no gradient history and
        // would drag the threshold toward zero.
        std::vector<double> grad_mags;
        grad_mags.reserve(nodes_.size());

        for (size_t i = 0; i < nodes_.size(); ++i) {
            if (!is_node_active(i)) continue;
            grad_mags.push_back(nodes_[i].compute_metrics().gradient_magnitude_avg);
        }

        if (grad_mags.empty()) return 0.1;

        // Compute mean
        double sum = 0.0;
        for (double g : grad_mags) {
            sum += g;
        }
        double mean = sum / grad_mags.size();

        // Compute standard deviation
        double sq_sum = 0.0;
        for (double g : grad_mags) {
            sq_sum += (g - mean) * (g - mean);
        }
        double std_dev = std::sqrt(sq_sum / grad_mags.size());

        return mean + std_dev;
    }

    /**
     * Set gradient threshold for all nodes in this layer.
     */
    void set_nodes_grad_threshold(double threshold) {
        for (auto& node : nodes_) {
            node.set_grad_threshold(threshold);
        }
    }

    /**
     * Adapt gradient/contribution weights for all nodes based on current gradient statistics.
     * Should be called each epoch after training.
     */
    void adapt_node_weights() {
        for (size_t i = 0; i < nodes_.size(); ++i) {
            if (!is_node_active(i)) continue;
            nodes_[i].adapt_gradient_weight();
        }
    }

    /**
     * Add nodes to the layer.
     *
     * Prefers reactivating dormant slots (preserving their parameters) before
     * physically growing the weight matrix. This is the soft-add half of the
     * dynamic-topology contract: a remove followed by an add of the same
     * count round-trips with weights intact.
     *
     * @return Number of slots reactivated (the rest were freshly allocated).
     */
    size_t add_nodes(size_t count) {
        if (count == 0) return 0;

        // First, reactivate dormant slots in index order.
        size_t reactivated = 0;
        for (size_t i = 0; i < active_mask_.size() && reactivated < count; ++i) {
            if (!active_mask_[i]) {
                active_mask_[i] = uint8_t(1);
                ++reactivated;
            }
        }

        size_t to_allocate = count - reactivated;
        if (to_allocate == 0) {
            // Mask values changed (some 0s flipped to 1s) but shape didn't.
            // Drop the GPU mask mirror; keep weights/biases mirrors.
            if (reactivated > 0) invalidate_gpu_active_mask_();
            ++topology_version_;
            return reactivated;
        }

        size_t old_output_size = output_size_;
        output_size_ += to_allocate;

        // Add new nodes
        for (size_t i = 0; i < to_allocate; ++i) {
            nodes_.emplace_back(old_output_size + i, true);
        }

        // Expand weights matrix
        Tensor<T> new_weights(std::vector<size_t>{output_size_, input_size_});

        // Copy old weights
        for (size_t i = 0; i < old_output_size; ++i) {
            for (size_t j = 0; j < input_size_; ++j) {
                new_weights.at(i, j) = weights_.at(i, j);
            }
        }

        // Initialize new weights
        for (size_t i = old_output_size; i < output_size_; ++i) {
            for (size_t j = 0; j < input_size_; ++j) {
                new_weights.at(i, j) = rng_->normal<T>(T(0), std::sqrt(T(2) / input_size_));
            }
        }

        weights_ = std::move(new_weights);

        // Expand biases
        Tensor<T> new_biases(std::vector<size_t>{output_size_}, T(0));
        for (size_t i = 0; i < old_output_size; ++i) {
            new_biases[i] = biases_[i];
        }
        biases_ = std::move(new_biases);

        // Extend the active mask for the freshly allocated rows.
        active_mask_.resize(output_size_, uint8_t(1));

        // Reset gradient accumulators
        weight_gradients_ = Tensor<T>();
        bias_gradients_ = Tensor<T>();

        // weights_/biases_ shapes just changed (output_size_ grew). Drop
        // GPU mirrors + transposed-weight cache so they lazy-rebuild at
        // the new size. CLAUDE.md invariant: every host shape change
        // invalidates GPU mirrors.
        invalidate_gpu_mirrors_();

        ++topology_version_;
        return reactivated;
    }

    /**
     * Mark nodes as inactive (soft remove).
     *
     * Parameters are preserved so the slot can be reactivated later by
     * add_nodes(). Hard removal happens only at compact() time, which is
     * called once at the end of training.
     */
    void remove_nodes(std::vector<size_t> indices) {
        if (indices.empty()) return;
        bool any_changed = false;
        for (size_t idx : indices) {
            if (idx < active_mask_.size() && active_mask_[idx] != 0) {
                active_mask_[idx] = uint8_t(0);
                any_changed = true;
            }
        }
        if (any_changed) invalidate_gpu_active_mask_();
        ++topology_version_;
    }

    /**
     * Hard-remove all dormant nodes from this layer's dense buffers.
     * Returns the number of nodes physically removed.
     *
     * Intended to be called once at end of training (Network::compact()),
     * not in the hot training loop.
     */
    size_t compact() {
#ifdef DNN_ENABLE_CUDA
        // compact() rebuilds weights_ / biases_ on host. If the on-device
        // optimizer wrote new values that haven't been mirrored back yet,
        // pull them now so we don't compact stale data.
        sync_host_weights_();
#endif
        std::vector<size_t> to_remove;
        for (size_t i = 0; i < active_mask_.size(); ++i) {
            if (!active_mask_[i]) to_remove.push_back(i);
        }
        if (to_remove.empty()) return 0;

        // Sort descending so we can erase from nodes_ in-place.
        std::sort(to_remove.begin(), to_remove.end(), std::greater<size_t>());
        for (size_t idx : to_remove) {
            if (idx < nodes_.size()) {
                nodes_.erase(nodes_.begin() + idx);
            }
        }

        size_t new_output_size = nodes_.size();
        Tensor<T> new_weights(std::vector<size_t>{new_output_size, input_size_});
        Tensor<T> new_biases(std::vector<size_t>{new_output_size});

        size_t dest_i = 0;
        for (size_t i = 0; i < output_size_; ++i) {
            if (active_mask_[i]) {
                for (size_t j = 0; j < input_size_; ++j) {
                    new_weights.at(dest_i, j) = weights_.at(i, j);
                }
                new_biases[dest_i] = biases_[i];
                ++dest_i;
            }
        }

        weights_ = std::move(new_weights);
        biases_ = std::move(new_biases);
        output_size_ = new_output_size;

        active_mask_.assign(output_size_, uint8_t(1));

        weight_gradients_ = Tensor<T>();
        bias_gradients_ = Tensor<T>();

        // compact() shrinks the dense buffers — every GPU mirror and the
        // transposed-weight cache must rebuild on the new shape.
        invalidate_gpu_mirrors_();

        ++topology_version_;
        return to_remove.size();
    }

    // Soft-topology accessors.
    bool is_node_active(size_t i) const {
        return i < active_mask_.size() && active_mask_[i] != 0;
    }
    void set_node_active(size_t i, bool a) {
        if (i < active_mask_.size()) {
            uint8_t prev = active_mask_[i];
            active_mask_[i] = a ? uint8_t(1) : uint8_t(0);
            if (active_mask_[i] != prev) invalidate_gpu_active_mask_();
            ++topology_version_;
        }
    }
    size_t active_node_count() const {
        size_t n = 0;
        for (uint8_t v : active_mask_) if (v) ++n;
        return n;
    }
    bool is_active() const { return layer_active_; }
    void set_active(bool a) {
        if (layer_active_ != a) {
            layer_active_ = a;
            ++topology_version_;
        }
    }
    uint64_t topology_version() const { return topology_version_; }

    /**
     * Prune nodes below efficiency threshold.
     * @param threshold Efficiency threshold (0.0 - 1.0)
     * @param min_nodes Minimum nodes to keep
     * @return Number of nodes removed
     */
    size_t prune_inefficient_nodes(double threshold, size_t min_nodes = 1) {
        std::vector<size_t> to_remove;

        for (size_t i = 0; i < nodes_.size() && nodes_.size() - to_remove.size() > min_nodes; ++i) {
            if (nodes_[i].efficiency_score() < threshold) {
                to_remove.push_back(i);
            }
        }

        // Keep minimum nodes
        while (nodes_.size() - to_remove.size() < min_nodes && !to_remove.empty()) {
            to_remove.pop_back();
        }

        if (!to_remove.empty()) {
            remove_nodes(to_remove);
        }

        return to_remove.size();
    }

    /**
     * Get total number of parameters.
     */
    size_t num_parameters() const {
        return output_size_ * input_size_ + output_size_;
    }

    /**
     * Clone the layer's *inference* state.
     *
     * Copies: weights_, biases_, nodes_ (per-node metrics/efficiency
     * weights), active_mask_, layer_active_, topology_version_.
     *
     * Does NOT copy: weight_gradients_ / bias_gradients_, the forward
     * caches (cached_input_/pre_activation_/output_), or any GPU
     * mirrors. A clone taken mid-training therefore has no gradient
     * state — it is intended for inheritance / inference, not for
     * resuming a backward pass. Use clone_with_state() if you need the
     * gradient accumulators (e.g. checkpointing a training run).
     */
    std::unique_ptr<Layer> clone() const {
        auto copy = std::make_unique<Layer>(input_size_, output_size_,
                                            activation_type_, rng_->seed());
        copy->weights_ = weights_.clone();
        copy->biases_ = biases_.clone();
        copy->nodes_ = nodes_;
        copy->active_mask_ = active_mask_;
        copy->layer_active_ = layer_active_;
        copy->topology_version_ = topology_version_;
        return copy;
    }

    /**
     * Like clone(), additionally copying the host gradient accumulators
     * (weight_gradients_ / bias_gradients_) so a checkpoint can resume
     * mid-training. GPU mirrors and forward caches are still not copied
     * (they are rebuilt lazily on the next forward).
     */
    std::unique_ptr<Layer> clone_with_state() const {
        auto copy = clone();
        copy->weight_gradients_ = weight_gradients_.clone();
        copy->bias_gradients_ = bias_gradients_.clone();
        return copy;
    }

private:
    /**
     * Zero out the inactive-node columns of a tensor sized along output_size_.
     * Used by:
     *   - backward (gradients): so dormant nodes' params don't drift and no
     *     gradient flows upstream through them.
     *   - forward_cuda (pre-activation): so the dense gemv/gemm output gets
     *     masked the same way forward_cpu's per-node loop does. Without this
     *     mid-training "remove node" had no visible effect on CUDA.
     */
    void zero_inactive_components(Tensor<T>& tensor) const {
        if (active_mask_.empty()) return;
        if (tensor.rank() == 1) {
            for (size_t i = 0; i < output_size_; ++i) {
                if (!is_node_active(i)) tensor[i] = T(0);
            }
        } else if (tensor.rank() == 2) {
            size_t batch_size = tensor.shape()[0];
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t i = 0; i < output_size_; ++i) {
                    if (!is_node_active(i)) tensor.at(b, i) = T(0);
                }
            }
        }
    }
    // Legacy alias kept so existing call sites compile unchanged.
    void zero_inactive_gradients(Tensor<T>& grad_activation) const {
        zero_inactive_components(grad_activation);
    }

    LayerType layer_type_;
    size_t input_size_;
    size_t output_size_;
    ActivationType activation_type_;
    Device device_ = Device::CPU;

    Tensor<T> weights_;
    Tensor<T> biases_;
    std::vector<Node> nodes_;

    // Soft-topology state. active_mask_[i] == 0 means the i-th node is
    // dormant: forward pass emits zero for it, backward pass accumulates no
    // gradient against it, and apply_gradients skips it. Parameters are
    // preserved across remove_nodes/add_nodes round-trips. Hard removal
    // happens only in compact().
    std::vector<uint8_t> active_mask_;
    bool layer_active_ = true;
    uint64_t topology_version_ = 0;

    // Cached values for backward pass
    Tensor<T> cached_input_;
    Tensor<T> cached_pre_activation_;
    Tensor<T> cached_output_;

    // Gradient accumulators
    Tensor<T> weight_gradients_;
    Tensor<T> bias_gradients_;

    std::unique_ptr<Activation<T>> activation_;
    std::unique_ptr<Random> rng_;

    // Lazily-built SIMD ops (AVX-512/AVX2/AVX/SSE/Scalar picked at runtime by
    // simd::SIMDOps<T>::create()). Used by forward_cpu/backward_cpu rank-2.
    // The pointee is stateless; `mutable` lets const accessors lazily build
    // it. SIMDOps is single-threaded externally — concurrent forwards under
    // the runtime path each build their own local transposed weights tensor
    // (see forward_cpu rank-2) rather than sharing a cached one.
    mutable std::unique_ptr<simd::SIMDOps<T>> simd_ops_;

#ifdef DNN_ENABLE_CUDA
    // GPU tensors for CUDA mode
    std::unique_ptr<cuda::CudaTensor<T>> gpu_weights_;
    std::unique_ptr<cuda::CudaTensor<T>> gpu_biases_;

    // Cached GPU active-mask: active_mask_ cast to T, shape (output_size_,).
    // Used by cuda_apply_mask_broadcast in forward_cuda_dev to zero columns
    // of soft-removed nodes without a host round-trip. Invalidated whenever
    // active_mask_ mutates (add_nodes / remove_nodes / set_node_active /
    // compact).
    std::unique_ptr<cuda::CudaTensor<T>> gpu_active_mask_;

    // GPU-resident forward caches. When device_ == CUDA, the forward_cuda_dev
    // path stores these (rather than the host versions) so backward_cuda_dev
    // can run the activation backward and the gemm on the GPU without a host
    // round-trip per layer. The host cached_input_ / cached_pre_activation_ /
    // cached_output_ remain authoritative for the CPU path.
    std::unique_ptr<cuda::CudaTensor<T>> cached_input_gpu_;
    std::unique_ptr<cuda::CudaTensor<T>> cached_pre_activation_gpu_;
    std::unique_ptr<cuda::CudaTensor<T>> cached_output_gpu_;

    // GPU-resident gradient accumulators. Mirror the host weight_gradients_
    // / bias_gradients_ when device_ == CUDA; populated by backward_cuda_dev
    // and consumed by apply_gradients (on-device SGD step). Avoids the
    // dW / db D2H per layer per batch.
    std::unique_ptr<cuda::CudaTensor<T>> gpu_weight_gradients_;
    std::unique_ptr<cuda::CudaTensor<T>> gpu_bias_gradients_;

    // Cached GPU update-mask: (active_mask_ AND nodes_[i].is_trainable()) cast
    // to T, shape (output_size_,). Multiplied against dW rows and db elements
    // in backward_cuda_dev so the on-device SGD step naturally skips inactive
    // and non-trainable neurons. Invalidated whenever active_mask_ or
    // trainability changes.
    std::unique_ptr<cuda::CudaTensor<T>> gpu_update_mask_;

    // True when GPU weights/biases have been updated by the on-device
    // optimizer step but the host weights_/biases_ haven't been refreshed
    // yet. sync_host_weights_() clears this. Read by code paths that need
    // fresh host values (compact, model save, has_numerical_issues).
    bool host_weights_dirty_ = false;
#endif

    // Centralized invalidator for the GPU weight/bias mirrors plus the
    // derived active-mask mirror. Called from every site that mutates
    // weights_/biases_ shape or active_mask_ in a shape-incompatible way.
    // No-op on CPU-only builds. Per include/dnn/core/CLAUDE.md invariant
    // "CUDA mirror invalidation runs on every host shape change".
    void invalidate_gpu_mirrors_() {
#ifdef DNN_ENABLE_CUDA
        gpu_weights_.reset();
        gpu_biases_.reset();
        gpu_active_mask_.reset();
        gpu_update_mask_.reset();
        gpu_weight_gradients_.reset();
        gpu_bias_gradients_.reset();
        // The forward caches reference the old layer shape; drop them so the
        // next forward rebuilds at the new size.
        cached_input_gpu_.reset();
        cached_pre_activation_gpu_.reset();
        cached_output_gpu_.reset();
        host_weights_dirty_ = false;
#endif
    }

    // Cheap invalidation when only active_mask_ values changed (no shape
    // change). Drops the active + update masks; weights/biases mirrors stay
    // valid.
    void invalidate_gpu_active_mask_() {
#ifdef DNN_ENABLE_CUDA
        gpu_active_mask_.reset();
        gpu_update_mask_.reset();
#endif
    }

#ifdef DNN_ENABLE_CUDA
    // Build / refresh gpu_active_mask_ from the host active_mask_ vector.
    // Called from forward_cuda_dev before the masking step.
    void ensure_gpu_active_mask_() {
        if (gpu_active_mask_ && gpu_active_mask_->size() == output_size_) {
            return;
        }
        Tensor<T> host_mask(std::vector<size_t>{output_size_});
        for (size_t i = 0; i < output_size_; ++i) {
            host_mask[i] = active_mask_[i] ? T(1) : T(0);
        }
        gpu_active_mask_ = std::make_unique<cuda::CudaTensor<T>>(host_mask);
    }

    // Build / refresh gpu_update_mask_ = (active_mask_ AND is_trainable) cast
    // to T. Used by the on-device SGD step to skip inactive or non-trainable
    // neurons. Rebuilt lazily on shape change and whenever the host signals
    // a trainability change (TrainableScheduler).
    void ensure_gpu_update_mask_() {
        if (gpu_update_mask_ && gpu_update_mask_->size() == output_size_) {
            return;
        }
        Tensor<T> host_mask(std::vector<size_t>{output_size_});
        for (size_t i = 0; i < output_size_; ++i) {
            const bool keep = active_mask_[i] != 0
                              && i < nodes_.size()
                              && nodes_[i].is_trainable();
            host_mask[i] = keep ? T(1) : T(0);
        }
        gpu_update_mask_ = std::make_unique<cuda::CudaTensor<T>>(host_mask);
    }

    // Pull GPU weights/biases back to host. Called before any code path that
    // needs fresh host values during training (compact, save, numerical
    // checks). No-op if not dirty.
    void sync_host_weights_() {
        if (!host_weights_dirty_) return;
        if (gpu_weights_) weights_ = gpu_weights_->to_host();
        if (gpu_biases_) biases_ = gpu_biases_->to_host();
        host_weights_dirty_ = false;
    }
#endif

    // Lazy SIMD ops accessor.
    simd::SIMDOps<T>* get_simd_ops_() const {
        if (!simd_ops_) {
            simd_ops_ = simd::SIMDOps<T>::create();
        }
        return simd_ops_.get();
    }
};

// Type alias
using LayerF = Layer<float>;
using LayerD = Layer<double>;

} // namespace core
} // namespace dnn
