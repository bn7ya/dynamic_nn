#pragma once

#include "tensor.hpp"
#include "node.hpp"
#include "activations.hpp"
#include "initializers.hpp"
#include "random.hpp"
#include "device.hpp"
#include "../exceptions/dnn_exception.hpp"
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
            // Move weights and biases to GPU
            gpu_weights_ = std::make_unique<cuda::CudaTensor<T>>(weights_);
            gpu_biases_ = std::make_unique<cuda::CudaTensor<T>>(biases_);
        } else {
            // Move back to CPU - weights/biases should already be updated
            if (gpu_weights_) {
                weights_ = gpu_weights_->to_host();
                gpu_weights_.reset();
            }
            if (gpu_biases_) {
                biases_ = gpu_biases_->to_host();
                gpu_biases_.reset();
            }
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
            // Single sample: input is (input_size,)
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

                // Record for node metrics
                nodes_[i].record_activation(static_cast<float>(sum));
            }
        } else if (input.rank() == 2) {
            // Batch: input is (batch_size, input_size). Each (b, i) cell
            // is independent so we parallelise the outer batch loop.
            size_t batch_size = input.shape()[0];
            linear_output = Tensor<T>(std::vector<size_t>{batch_size, output_size_});

#ifdef DNN_HAS_OPENMP
            #pragma omp parallel for schedule(static) if (batch_size > 4)
#endif
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t i = 0; i < output_size_; ++i) {
                    if (!is_node_active(i)) {
                        linear_output.at(b, i) = T(0);
                        continue;
                    }
                    T sum = biases_[i];
                    for (size_t j = 0; j < input_size_; ++j) {
                        sum += input.at(b, j) * weights_.at(i, j);
                    }
                    linear_output.at(b, i) = sum;
                }
            }
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
     * CUDA forward pass implementation.
     */
    Tensor<T> forward_cuda(const Tensor<T>& input) {
        // Move input to GPU
        cuda::CudaTensor<T> gpu_input(input);

        // Cache input for backward pass (keep on CPU for now)
        cached_input_ = input.clone();

        size_t batch_size = (input.rank() == 2) ? input.shape()[0] : 1;

        // Prepare output tensor on GPU
        // For batch: output is (batch_size, output_size)
        // Linear: Y = X @ W^T + B
        std::vector<size_t> output_shape = (input.rank() == 2)
            ? std::vector<size_t>{batch_size, output_size_}
            : std::vector<size_t>{output_size_};

        cuda::CudaTensor<T> gpu_output(output_shape);

        // Ensure weights are on GPU
        if (!gpu_weights_) {
            gpu_weights_ = std::make_unique<cuda::CudaTensor<T>>(weights_);
        }
        if (!gpu_biases_) {
            gpu_biases_ = std::make_unique<cuda::CudaTensor<T>>(biases_);
        }

        if (input.rank() == 1) {
            // Single sample: y = W @ x + b
            // Using gemv: y = alpha * A @ x + beta * y
            // First copy biases to output
            cuda::cuda_copy(*gpu_biases_, gpu_output);
            // Then: output = 1.0 * weights @ input + 1.0 * output (biases)
            cuda::cuda_gemv(*gpu_weights_, gpu_input, gpu_output, T(1), T(1), false);
        } else {
            // Batch: Y = X @ W^T + B (broadcast biases)
            // Using gemm: C = alpha * A @ B + beta * C
            // output = input @ weights^T, then add biases

            // First: output = input @ weights^T
            cuda::cuda_gemm(gpu_input, *gpu_weights_, gpu_output, T(1), T(0), false, true);

            // Add biases to each row (broadcast)
            // Create a temporary for bias broadcast
            Tensor<T> bias_broadcast(output_shape);
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t i = 0; i < output_size_; ++i) {
                    bias_broadcast.at(b, i) = biases_[i];
                }
            }
            cuda::CudaTensor<T> gpu_bias_broadcast(bias_broadcast);
            cuda::cuda_add(gpu_output, gpu_bias_broadcast, gpu_output);
        }

        // Copy result back to CPU for pre-activation cache
        Tensor<T> linear_output = gpu_output.to_host();
        cached_pre_activation_ = linear_output.clone();

        // Apply activation on GPU
        cuda::CudaTensor<T> gpu_activated(output_shape);

        switch (activation_type_) {
            case ActivationType::ReLU:
                cuda::cuda_relu(gpu_output, gpu_activated);
                break;
            case ActivationType::Sigmoid:
                cuda::cuda_sigmoid(gpu_output, gpu_activated);
                break;
            case ActivationType::Tanh:
                cuda::cuda_tanh(gpu_output, gpu_activated);
                break;
            case ActivationType::Softmax:
                cuda::cuda_softmax(gpu_output, gpu_activated);
                break;
            default:
                // Fall back to CPU activation
                cached_output_ = activation_->forward(linear_output);
                return cached_output_.clone();
        }

        // Copy activated output back to CPU
        cached_output_ = gpu_activated.to_host();

        // Record node activations for metrics (sample from first batch item)
        for (size_t i = 0; i < output_size_ && i < cached_pre_activation_.size(); ++i) {
            nodes_[i].record_activation(static_cast<float>(
                input.rank() == 1 ? cached_pre_activation_[i] : cached_pre_activation_.at(0, i)));
        }

        return cached_output_.clone();
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
            // Batch
            size_t batch_size = cached_input_.shape()[0];

            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t i = 0; i < output_size_; ++i) {
                    if (!is_node_active(i)) continue;
                    if (nodes_[i].is_trainable()) {
                        for (size_t j = 0; j < input_size_; ++j) {
                            weight_gradients_.at(i, j) +=
                                grad_activation.at(b, i) * cached_input_.at(b, j);
                        }
                        bias_gradients_[i] += grad_activation.at(b, i);
                    }
                }
            }

            // Compute input gradient. Each (b, j) cell is independent
            // (no cross-write), so the outer batch loop parallelises
            // safely. Weight-gradient accumulation above stays serial
            // over batch because rows of weight_gradients_ are written
            // by every b.
            Tensor<T> grad_input(std::vector<size_t>{batch_size, input_size_});
#ifdef DNN_HAS_OPENMP
            #pragma omp parallel for schedule(static) if (batch_size > 4)
#endif
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t j = 0; j < input_size_; ++j) {
                    T sum = T(0);
                    for (size_t i = 0; i < output_size_; ++i) {
                        sum += weights_.at(i, j) * grad_activation.at(b, i);
                    }
                    grad_input.at(b, j) = sum;
                }
            }
            return grad_input;
        }

        throw exceptions::ShapeException("backward", "Cached input has invalid shape");
    }

#ifdef DNN_ENABLE_CUDA
    /**
     * CUDA backward pass implementation.
     */
    Tensor<T> backward_cuda(const Tensor<T>& grad_output) {
        // Compute activation gradient on CPU (for now, since we cached on CPU)
        Tensor<T> grad_activation = activation_->backward(
            cached_pre_activation_, cached_output_, grad_output);

        // Zero gradients for inactive (dormant) nodes so their parameters
        // don't drift and no gradient flows upstream through them.
        zero_inactive_gradients(grad_activation);

        // Move grad_activation to GPU
        cuda::CudaTensor<T> gpu_grad_activation(grad_activation);

        // Move cached input to GPU
        cuda::CudaTensor<T> gpu_cached_input(cached_input_);

        size_t batch_size = (cached_input_.rank() == 2) ? cached_input_.shape()[0] : 1;

        // Compute weight gradients: dW = grad_activation^T @ cached_input
        // For batch: dW = sum over batch of (grad[b] outer input[b])
        // Using gemm: dW += grad_activation^T @ cached_input

        // Compute input gradient: grad_input = grad_activation @ weights
        // Using gemm: grad_input = grad_activation @ weights

        if (cached_input_.rank() == 1) {
            // Single sample
            // Weight gradient: dW[i,j] = grad_activation[i] * cached_input[j]
            // This is an outer product

            // For now, compute on CPU and accumulate
            for (size_t i = 0; i < output_size_; ++i) {
                if (!is_node_active(i)) continue;
                if (nodes_[i].is_trainable()) {
                    for (size_t j = 0; j < input_size_; ++j) {
                        weight_gradients_.at(i, j) += grad_activation[i] * cached_input_[j];
                    }
                    bias_gradients_[i] += grad_activation[i];
                }
                nodes_[i].record_gradient(static_cast<float>(grad_activation[i]));
            }

            // Compute input gradient on GPU: grad_input = weights^T @ grad_activation
            cuda::CudaTensor<T> gpu_grad_input(std::vector<size_t>{input_size_});
            cuda::cuda_gemv(*gpu_weights_, gpu_grad_activation, gpu_grad_input, T(1), T(0), true);

            return gpu_grad_input.to_host();

        } else {
            // Batch mode
            // Weight gradient: dW = grad_activation^T @ cached_input
            // Shape: (output_size, batch_size) @ (batch_size, input_size) = (output_size, input_size)

            // Compute weight gradients on GPU
            cuda::CudaTensor<T> gpu_weight_grad(std::vector<size_t>{output_size_, input_size_});
            cuda::cuda_gemm(gpu_grad_activation, gpu_cached_input, gpu_weight_grad, T(1), T(0), true, false);

            // Accumulate weight gradients to CPU
            Tensor<T> weight_grad_cpu = gpu_weight_grad.to_host();
            for (size_t i = 0; i < output_size_; ++i) {
                if (!is_node_active(i)) continue;
                if (nodes_[i].is_trainable()) {
                    for (size_t j = 0; j < input_size_; ++j) {
                        weight_gradients_.at(i, j) += weight_grad_cpu.at(i, j);
                    }
                }
            }

            // Bias gradient: sum of grad_activation along batch dimension
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t i = 0; i < output_size_; ++i) {
                    if (!is_node_active(i)) continue;
                    if (nodes_[i].is_trainable()) {
                        bias_gradients_[i] += grad_activation.at(b, i);
                    }
                }
            }

            // Record metrics
            for (size_t i = 0; i < output_size_; ++i) {
                nodes_[i].record_gradient(static_cast<float>(grad_activation.at(0, i)));
            }

            // Compute input gradient on GPU: grad_input = grad_activation @ weights
            cuda::CudaTensor<T> gpu_grad_input(std::vector<size_t>{batch_size, input_size_});
            cuda::cuda_gemm(gpu_grad_activation, *gpu_weights_, gpu_grad_input, T(1), T(0), false, false);

            return gpu_grad_input.to_host();
        }
    }
#endif

public:

    /**
     * Apply accumulated gradients and update weights.
     * @param learning_rate Learning rate for update
     */
    void apply_gradients(T learning_rate) {
        for (size_t i = 0; i < output_size_; ++i) {
            if (!is_node_active(i)) continue;
            if (nodes_[i].is_trainable()) {
                for (size_t j = 0; j < input_size_; ++j) {
                    weights_.at(i, j) -= learning_rate * weight_gradients_.at(i, j);
                }
                biases_[i] -= learning_rate * bias_gradients_[i];
            }
        }

        // Reset gradients
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

        // Collect all node variances
        std::vector<double> variances;
        variances.reserve(nodes_.size());

        for (const auto& node : nodes_) {
            auto metrics = node.compute_metrics();
            variances.push_back(metrics.activation_variance);
        }

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

        // Compute z-scores and adapt variance weights
        for (size_t i = 0; i < nodes_.size(); ++i) {
            double z = (variances[i] - mean) / std_dev;
            nodes_[i].adapt_variance_weight(z);
        }
    }

    /**
     * Compute gradient threshold from current gradient statistics.
     * Returns mean + std of gradient magnitudes.
     */
    double compute_gradient_threshold() const {
        if (nodes_.empty()) return 0.1;

        std::vector<double> grad_mags;
        grad_mags.reserve(nodes_.size());

        for (const auto& node : nodes_) {
            auto metrics = node.compute_metrics();
            grad_mags.push_back(metrics.gradient_magnitude_avg);
        }

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
        for (auto& node : nodes_) {
            node.adapt_gradient_weight();
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
        for (size_t idx : indices) {
            if (idx < active_mask_.size()) {
                active_mask_[idx] = uint8_t(0);
            }
        }
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

#ifdef DNN_ENABLE_CUDA
        // Force GPU buffers to be re-uploaded on next forward.
        gpu_weights_.reset();
        gpu_biases_.reset();
#endif

        ++topology_version_;
        return to_remove.size();
    }

    // Soft-topology accessors.
    bool is_node_active(size_t i) const {
        return i < active_mask_.size() && active_mask_[i] != 0;
    }
    void set_node_active(size_t i, bool a) {
        if (i < active_mask_.size()) {
            active_mask_[i] = a ? uint8_t(1) : uint8_t(0);
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
     * Clone the layer.
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

private:
    /**
     * Zero out grad_activation entries for inactive nodes.
     * Used by both CPU and CUDA backward to ensure dormant nodes' params
     * don't drift and no gradient flows upstream through them.
     */
    void zero_inactive_gradients(Tensor<T>& grad_activation) const {
        if (active_mask_.empty()) return;
        if (grad_activation.rank() == 1) {
            for (size_t i = 0; i < output_size_; ++i) {
                if (!is_node_active(i)) grad_activation[i] = T(0);
            }
        } else if (grad_activation.rank() == 2) {
            size_t batch_size = grad_activation.shape()[0];
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t i = 0; i < output_size_; ++i) {
                    if (!is_node_active(i)) grad_activation.at(b, i) = T(0);
                }
            }
        }
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

#ifdef DNN_ENABLE_CUDA
    // GPU tensors for CUDA mode
    std::unique_ptr<cuda::CudaTensor<T>> gpu_weights_;
    std::unique_ptr<cuda::CudaTensor<T>> gpu_biases_;
#endif
};

// Type alias
using LayerF = Layer<float>;
using LayerD = Layer<double>;

} // namespace core
} // namespace dnn
