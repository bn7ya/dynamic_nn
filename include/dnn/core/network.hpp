#pragma once

#include "tensor.hpp"
#include "layer.hpp"
#include "device.hpp"
#include "random.hpp"
#include "../training/cost_functions.hpp"
#include "../exceptions/dnn_exception.hpp"
#include <vector>
#include <memory>
#include <cmath>
#include <string>

namespace dnn {
namespace core {

// Use training::CostFunctionType in this namespace
using training::CostFunctionType;

/**
 * Network configuration.
 */
struct NetworkConfig {
    uint64_t seed = 42;                              // ONLY required hyperparameter
    std::vector<size_t> input_shape;                 // Shape of input data
    size_t output_size = 0;                          // Number of output neurons
    CostFunctionType cost_function = CostFunctionType::CrossEntropy;  // User-selected cost function
    ActivationType output_activation = ActivationType::Softmax;  // Output activation
    ActivationType hidden_activation = ActivationType::ReLU;     // Hidden layer activation
    Device device = Device::CPU;                     // Device for computation (CPU or CUDA)
};

/**
 * Training state snapshot.
 */
struct TrainingState {
    uint64_t epoch = 0;
    uint64_t batch = 0;
    uint64_t total_batches = 0;
    double current_cost = 0.0;
    double current_efficiency = 0.0;
    size_t total_layers = 0;
    size_t total_nodes = 0;
    size_t trainable_nodes = 0;
    size_t total_parameters = 0;
    std::string health_status = "initializing";
};

/**
 * Network-level metrics.
 */
struct NetworkMetrics {
    double avg_layer_efficiency = 0.0;
    double min_layer_efficiency = 1.0;
    double max_layer_efficiency = 0.0;
    double gradient_flow_health = 0.0;
    double parameter_utilization = 0.0;
    size_t total_layers = 0;
    size_t total_nodes = 0;
    size_t total_parameters = 0;
    size_t dead_nodes = 0;
    size_t saturated_nodes = 0;
    bool is_healthy = true;
    std::string health_status = "healthy";
};

/**
 * Dynamic Neural Network.
 * Automatically adapts architecture during training.
 */
template<typename T = float>
class Network {
public:
    /**
     * Construct network from configuration.
     */
    explicit Network(const NetworkConfig& config)
        : config_(config)
        , device_(config.device)
        , rng_(std::make_unique<Random>(config.seed)) {

        // Compute input size from shape
        input_size_ = 1;
        for (size_t dim : config.input_shape) {
            input_size_ *= dim;
        }

        if (input_size_ == 0 || config.output_size == 0) {
            throw exceptions::InvalidArgumentException("config",
                "Input and output sizes must be positive");
        }

        // Compute initial hidden layer size (geometric mean)
        size_t initial_hidden = compute_initial_hidden_size();

        // Create initial architecture: input -> hidden -> output
        // Start with single hidden layer, will grow/shrink dynamically
        add_layer(input_size_, initial_hidden, config.hidden_activation);
        add_layer(initial_hidden, config.output_size, config.output_activation);

        state_.total_layers = layers_.size();
        update_state();
    }

    /**
     * Move network to specified device.
     * @param device Target device (CPU or CUDA)
     */
    void to(Device device) {
        if (device == device_) return;

        device_ = device;
        for (auto& layer : layers_) {
            layer->to_device(device);
        }
    }

    /**
     * Get current device.
     */
    Device device() const { return device_; }

    /**
     * Forward pass through the network.
     *
     * On CUDA, performs a single host→device transfer at the start and a
     * single device→host transfer at the end; intermediate layer outputs
     * stay on the GPU between layers. On CPU, uses each layer's SIMD-
     * accelerated forward_cpu (OpenMP-parallel matmul). See
     * include/dnn/core/CLAUDE.md for the device-resident chain contract.
     */
    Tensor<T> forward(const Tensor<T>& input) {
        // Flatten input if needed
        Tensor<T> x = input;
        if (input.size() == input_size_ && input.rank() != 1) {
            x = input.flatten();
        }

#ifdef DNN_ENABLE_CUDA
        if (device_ == Device::CUDA && !layers_.empty()) {
            cuda::CudaTensor<T> gpu_x(x);
            for (auto& layer : layers_) {
                gpu_x = layer->forward_cuda_dev(std::move(gpu_x));
            }
            return gpu_x.to_host();
        }
#endif

        // CPU path: each layer's forward_cpu uses SIMDOps::matmul + OpenMP.
        for (auto& layer : layers_) {
            x = layer->forward(x);
        }

        return x;
    }

    /**
     * Predict for a batch of inputs.
     */
    std::vector<Tensor<T>> predict_batch(const std::vector<Tensor<T>>& inputs) {
        std::vector<Tensor<T>> outputs;
        outputs.reserve(inputs.size());
        for (const auto& input : inputs) {
            outputs.push_back(forward(input));
        }
        return outputs;
    }

    /**
     * Backward pass (compute gradients).
     * @param grad_output Gradient from loss function
     *
     * Symmetrical to forward(): on CUDA, one H2D at the start, gradients
     * propagate device-resident through every layer, no D2H at the end
     * (the input-side grad is discarded — only the per-layer weight and
     * bias gradients are accumulated for the optimizer step).
     */
    void backward(const Tensor<T>& grad_output) {
#ifdef DNN_ENABLE_CUDA
        if (device_ == Device::CUDA && !layers_.empty()) {
            cuda::CudaTensor<T> gpu_grad(grad_output);
            for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
                gpu_grad = (*it)->backward_cuda_dev(gpu_grad);
            }
            return;
        }
#endif

        Tensor<T> grad = grad_output;

        // Backpropagate through layers in reverse (CPU path).
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            grad = (*it)->backward(grad);
        }
    }

    /**
     * Apply gradients to update weights.
     */
    void apply_gradients(T learning_rate) {
        for (auto& layer : layers_) {
            layer->apply_gradients(learning_rate);
        }
    }

    /**
     * Zero out all gradients.
     */
    void zero_gradients() {
        for (auto& layer : layers_) {
            layer->zero_gradients();
        }
    }

    /**
     * Add a new layer to the network.
     */
    void add_layer(size_t input_size, size_t output_size,
                   ActivationType activation) {
        auto layer = std::make_unique<Layer<T>>(
            input_size, output_size, activation,
            rng_->seed() + layers_.size(),
            device_);
        layers_.push_back(std::move(layer));
        update_state();
    }

    /**
     * Insert a new layer at the given index.
     */
    void insert_layer(size_t index, size_t num_nodes) {
        if (index == 0 || index >= layers_.size()) {
            throw exceptions::InvalidArgumentException("index",
                "Cannot insert at boundary layers");
        }

        // Get adjacent layer sizes
        size_t prev_output = layers_[index - 1]->output_size();
        size_t next_input = layers_[index]->input_size();

        if (prev_output != next_input) {
            throw exceptions::ShapeException("insert_layer",
                "Adjacent layer sizes don't match");
        }

        // Create new layer (device_ matters: without it, the layer defaults
        // to CPU and the network silently splits across devices on CUDA).
        auto new_layer = std::make_unique<Layer<T>>(
            prev_output, num_nodes, config_.hidden_activation,
            rng_->seed() + layers_.size(), device_);

        layers_.insert(layers_.begin() + index, std::move(new_layer));

        // Adjust the following layer — rebuild with new input_size_ and
        // the same device_ as the rest of the network.
        auto& next_layer = layers_[index + 1];
        auto rebuilt = std::make_unique<Layer<T>>(
            num_nodes, next_layer->output_size(),
            next_layer->activation_type(),
            rng_->seed() + index, device_);
        layers_[index + 1] = std::move(rebuilt);

        update_state();
    }

    /**
     * Remove a layer from the network.
     */
    void remove_layer(size_t index) {
        if (layers_.size() <= 2) {
            throw exceptions::InvalidArgumentException("index",
                "Cannot remove layers when only 2 remain");
        }
        if (index == 0 || index >= layers_.size() - 1) {
            throw exceptions::InvalidArgumentException("index",
                "Cannot remove first or last layer");
        }

        // Need to adjust adjacent layers
        size_t prev_output = layers_[index - 1]->output_size();
        size_t next_output = layers_[index + 1]->output_size();

        // Remove the layer
        layers_.erase(layers_.begin() + index);

        // Rebuild the layer that was after the removed one (preserve device_;
        // see insert_layer for why a missing device_ silently splits the
        // network across CPU/GPU on CUDA builds).
        auto& layer_after = layers_[index];
        auto rebuilt = std::make_unique<Layer<T>>(
            prev_output, next_output,
            layer_after->activation_type(),
            rng_->seed() + index, device_);
        layers_[index] = std::move(rebuilt);

        update_state();
    }

    // Layer access
    size_t num_layers() const { return layers_.size(); }

    Layer<T>& layer(size_t idx) { return *layers_[idx]; }
    const Layer<T>& layer(size_t idx) const { return *layers_[idx]; }

    std::vector<std::unique_ptr<Layer<T>>>& layers() { return layers_; }
    const std::vector<std::unique_ptr<Layer<T>>>& layers() const { return layers_; }

    // Configuration and state
    const NetworkConfig& config() const { return config_; }
    const TrainingState& state() const { return state_; }
    TrainingState& state() { return state_; }

    /**
     * Get total number of parameters.
     */
    size_t num_parameters() const {
        size_t total = 0;
        for (const auto& layer : layers_) {
            total += layer->num_parameters();
        }
        return total;
    }

    /**
     * Get total number of nodes.
     */
    size_t num_nodes() const {
        size_t total = 0;
        for (const auto& layer : layers_) {
            total += layer->num_nodes();
        }
        return total;
    }

    /**
     * Get number of trainable nodes.
     */
    size_t num_trainable_nodes() const {
        size_t total = 0;
        for (const auto& layer : layers_) {
            total += layer->trainable_count();
        }
        return total;
    }

    /**
     * Compute network metrics.
     */
    NetworkMetrics compute_metrics() const {
        NetworkMetrics metrics;
        metrics.total_layers = layers_.size();

        double efficiency_sum = 0.0;
        for (const auto& layer : layers_) {
            auto layer_metrics = layer->compute_metrics();

            efficiency_sum += layer_metrics.avg_node_efficiency;
            metrics.min_layer_efficiency = std::min(
                metrics.min_layer_efficiency, layer_metrics.avg_node_efficiency);
            metrics.max_layer_efficiency = std::max(
                metrics.max_layer_efficiency, layer_metrics.avg_node_efficiency);

            metrics.total_nodes += layer_metrics.total_nodes;
            metrics.total_parameters += layer->num_parameters();
            metrics.dead_nodes += layer_metrics.dead_nodes;
            metrics.saturated_nodes += layer_metrics.saturated_nodes;

            if (!layer_metrics.is_healthy) {
                metrics.is_healthy = false;
            }
        }

        metrics.avg_layer_efficiency = efficiency_sum / layers_.size();

        // Determine overall health
        if (metrics.dead_nodes > metrics.total_nodes / 2) {
            metrics.health_status = "critical_dead_nodes";
            metrics.is_healthy = false;
        } else if (metrics.saturated_nodes > metrics.total_nodes / 2) {
            metrics.health_status = "critical_saturated_nodes";
            metrics.is_healthy = false;
        } else if (metrics.avg_layer_efficiency < 0.2) {
            metrics.health_status = "low_efficiency";
            metrics.is_healthy = false;
        }

        return metrics;
    }

    /**
     * Reset all node metrics.
     */
    void reset_metrics() {
        for (auto& layer : layers_) {
            layer->reset_node_metrics();
        }
    }

    /**
     * Update efficiency scores.
     */
    void update_efficiency() {
        for (auto& layer : layers_) {
            layer->update_node_efficiency();
        }
    }

    /**
     * Set trainable fraction across all layers.
     */
    void set_trainable_fraction(double fraction) {
        for (auto& layer : layers_) {
            layer->set_trainable_fraction(fraction);
        }
        update_state();
    }

    /**
     * Inherit layers from another network based on efficiency threshold.
     * @param parent Network to inherit from
     * @param efficiency_threshold Only inherit layers above this efficiency
     */
    void inherit_from(const Network& parent, double efficiency_threshold = 0.5) {
        // Clear existing layers (except input shape)
        layers_.clear();

        // Compute efficiency for each parent layer
        for (size_t i = 0; i < parent.num_layers(); ++i) {
            auto layer_metrics = parent.layer(i).compute_metrics();

            if (layer_metrics.avg_node_efficiency >= efficiency_threshold ||
                i == 0 || i == parent.num_layers() - 1) {
                // Inherit this layer (always inherit first and last)
                auto cloned = parent.layer(i).clone();
                layers_.push_back(std::move(cloned));
            }
        }

        // Fix layer connections if needed
        for (size_t i = 1; i < layers_.size(); ++i) {
            if (layers_[i - 1]->output_size() != layers_[i]->input_size()) {
                // Need to add a bridge layer
                auto bridge = std::make_unique<Layer<T>>(
                    layers_[i - 1]->output_size(),
                    layers_[i]->input_size(),
                    config_.hidden_activation,
                    rng_->seed() + i);
                layers_.insert(layers_.begin() + i, std::move(bridge));
                ++i;  // Skip the inserted layer
            }
        }

        update_state();
    }

    /**
     * Clone the network.
     */
    std::unique_ptr<Network> clone() const {
        auto copy = std::make_unique<Network>(config_);
        copy->layers_.clear();
        for (const auto& layer : layers_) {
            copy->layers_.push_back(layer->clone());
        }
        copy->state_ = state_;
        return copy;
    }

    /**
     * Mark a hidden layer as inactive (soft remove).
     *
     * Parameters are preserved; the layer keeps participating in forward and
     * backward passes during training, but a bumped topology_version_ tells
     * the controller it's scheduled for removal. Hard removal happens at
     * compact() time, where the next layer is rebuilt to take this layer's
     * input shape directly. Use set_layer_active(idx, true) to undo.
     */
    void mark_layer_inactive(size_t index) {
        if (layers_.size() <= 2) return;
        if (index == 0 || index >= layers_.size() - 1) return;
        layers_[index]->set_active(false);
        ++topology_version_;
    }

    void set_layer_active(size_t index, bool active) {
        if (index >= layers_.size()) return;
        if (layers_[index]->is_active() != active) {
            layers_[index]->set_active(active);
            ++topology_version_;
        }
    }

    /**
     * Hard-remove all dormant capacity from the network.
     *
     * Compacts each layer's dormant nodes, then erases inactive hidden
     * layers (rebuilding the following layer to keep shapes consistent).
     * Intended to be called once at end of training to produce a lean
     * inference-ready model.
     *
     * Soft-delete contract: the fan-in repair pass at the end **resets
     * the rebuilt layer's weights**, which is only safe end-of-training.
     * Throws InvalidArgumentException if called while a Trainer has
     * marked training as in-progress, unless `allow_unsafe = true` is
     * passed (intended for tests that exercise compact() in isolation).
     *
     * Returns the total number of nodes physically removed.
     */
    size_t compact(bool allow_unsafe = false) {
        if (training_in_progress_ && !allow_unsafe) {
            throw exceptions::InvalidArgumentException("compact",
                "compact() called while training is in progress; "
                "this would silently reset rebuilt layers' weights. "
                "Call only after Trainer::train_phased() returns, or "
                "pass allow_unsafe=true if you know what you're doing.");
        }
        size_t total_nodes_removed = 0;

        // Pass 1: compact each layer's dormant nodes.
        for (auto& layer : layers_) {
            total_nodes_removed += layer->compact();
        }

        // Pass 2: erase inactive hidden layers, rebuilding the next layer
        // with the correct input shape so the chain stays connected.
        // Iterate from back to front so indices remain valid as we erase.
        for (size_t i = layers_.size(); i-- > 0; ) {
            if (i == 0 || i >= layers_.size() - 1) continue;
            if (!layers_[i]->is_active()) {
                size_t prev_output = layers_[i - 1]->output_size();
                size_t next_output = layers_[i + 1]->output_size();
                ActivationType next_act = layers_[i + 1]->activation_type();
                layers_.erase(layers_.begin() + i);
                auto rebuilt = std::make_unique<Layer<T>>(
                    prev_output, next_output, next_act,
                    rng_->seed() + i, device_);
                layers_[i] = std::move(rebuilt);
            }
        }

        // Repair input-size mismatches that compaction can introduce when a
        // layer's output count shrinks: rebuild the downstream layer to
        // accept the new fan-in. Weights of the rebuilt layer are reset --
        // this is acceptable because compact() is end-of-training.
        for (size_t i = 1; i < layers_.size(); ++i) {
            if (layers_[i]->input_size() != layers_[i - 1]->output_size()) {
                size_t new_in = layers_[i - 1]->output_size();
                size_t out = layers_[i]->output_size();
                ActivationType act = layers_[i]->activation_type();
                layers_[i] = std::make_unique<Layer<T>>(
                    new_in, out, act, rng_->seed() + i, device_);
            }
        }

        ++topology_version_;
        update_state();
        return total_nodes_removed;
    }

    uint64_t topology_version() const { return topology_version_; }

    /**
     * Mark the network as currently being trained.
     *
     * Trainer<T>::train_phased{,_runtime}() set this for the duration of
     * the call so that compact() refuses to run mid-training (its
     * fan-in repair pass resets weights). Cleared at the end of training
     * via RAII or explicit set_training_in_progress(false).
     */
    void set_training_in_progress(bool in_progress) {
        training_in_progress_ = in_progress;
    }
    bool is_training_in_progress() const { return training_in_progress_; }

    /**
     * Check for numerical issues.
     */
    bool has_numerical_issues() const {
        for (const auto& layer : layers_) {
            if (layer->weights().has_nan() || layer->weights().has_inf()) {
                return true;
            }
            if (layer->biases().has_nan() || layer->biases().has_inf()) {
                return true;
            }
        }
        return false;
    }

private:
    /**
     * Compute initial hidden layer size as geometric mean of input and output.
     */
    size_t compute_initial_hidden_size() const {
        double geometric_mean = std::sqrt(
            static_cast<double>(input_size_) * config_.output_size);
        size_t hidden = static_cast<size_t>(geometric_mean);

        // Ensure reasonable bounds
        hidden = std::max(hidden, size_t(4));
        hidden = std::min(hidden, std::max(input_size_, config_.output_size));

        return hidden;
    }

    /**
     * Update internal state.
     */
    void update_state() {
        state_.total_layers = layers_.size();
        state_.total_nodes = num_nodes();
        state_.trainable_nodes = num_trainable_nodes();
        state_.total_parameters = num_parameters();
    }

    NetworkConfig config_;
    size_t input_size_ = 0;
    Device device_ = Device::CPU;
    std::vector<std::unique_ptr<Layer<T>>> layers_;
    TrainingState state_;
    std::unique_ptr<Random> rng_;
    // Bumped on every soft topology mutation (node mask flip, layer
    // activate/deactivate, compact()). Workers reading the network can
    // detect that topology has shifted under them by comparing a cached
    // version.
    uint64_t topology_version_ = 0;
    // Set by Trainer<T>::train_phased{,_runtime}() while training runs.
    // Guards compact() against being called mid-training (which would
    // silently reset rebuilt-layer weights via the fan-in repair pass).
    bool training_in_progress_ = false;
};

// Type aliases
using NetworkF = Network<float>;
using NetworkD = Network<double>;

} // namespace core
} // namespace dnn
