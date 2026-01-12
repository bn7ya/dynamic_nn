#pragma once

#include "tensor.hpp"
#include "node.hpp"
#include "activations.hpp"
#include "initializers.hpp"
#include "random.hpp"
#include "../exceptions/dnn_exception.hpp"
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>

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
     */
    Layer(size_t input_size, size_t output_size,
          ActivationType activation = ActivationType::ReLU,
          uint64_t seed = 42)
        : layer_type_(LayerType::Dense)
        , input_size_(input_size)
        , output_size_(output_size)
        , activation_type_(activation)
        , weights_(std::vector<size_t>{output_size, input_size})
        , biases_(std::vector<size_t>{output_size})
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
    }

    /**
     * Forward pass.
     */
    Tensor<T> forward(const Tensor<T>& input) {
        // Cache input for backward pass
        cached_input_ = input.clone();

        // Linear transformation: output = input @ weights^T + biases
        Tensor<T> linear_output;

        if (input.rank() == 1) {
            // Single sample: input is (input_size,)
            linear_output = Tensor<T>(std::vector<size_t>{output_size_});
            for (size_t i = 0; i < output_size_; ++i) {
                T sum = biases_[i];
                for (size_t j = 0; j < input_size_; ++j) {
                    sum += input[j] * weights_.at(i, j);
                }
                linear_output[i] = sum;

                // Record for node metrics
                nodes_[i].record_activation(static_cast<float>(sum));
            }
        } else if (input.rank() == 2) {
            // Batch: input is (batch_size, input_size)
            size_t batch_size = input.shape()[0];
            linear_output = Tensor<T>(std::vector<size_t>{batch_size, output_size_});

            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t i = 0; i < output_size_; ++i) {
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

    /**
     * Backward pass.
     * @param grad_output Gradient from next layer
     * @return Gradient with respect to input
     */
    Tensor<T> backward(const Tensor<T>& grad_output) {
        // Compute activation gradient
        Tensor<T> grad_activation = activation_->backward(
            cached_pre_activation_, cached_output_, grad_output);

        // Compute gradients for weights and biases
        if (cached_input_.rank() == 1) {
            // Single sample
            for (size_t i = 0; i < output_size_; ++i) {
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
                    if (nodes_[i].is_trainable()) {
                        for (size_t j = 0; j < input_size_; ++j) {
                            weight_gradients_.at(i, j) +=
                                grad_activation.at(b, i) * cached_input_.at(b, j);
                        }
                        bias_gradients_[i] += grad_activation.at(b, i);
                    }
                }
            }

            // Compute input gradient
            Tensor<T> grad_input(std::vector<size_t>{batch_size, input_size_});
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

    /**
     * Apply accumulated gradients and update weights.
     * @param learning_rate Learning rate for update
     */
    void apply_gradients(T learning_rate) {
        for (size_t i = 0; i < output_size_; ++i) {
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
        for (const auto& node : nodes_) {
            auto node_metrics = node.compute_metrics();
            double eff = node_metrics.efficiency_score;

            efficiency_sum += eff;
            metrics.min_node_efficiency = std::min(metrics.min_node_efficiency, eff);
            metrics.max_node_efficiency = std::max(metrics.max_node_efficiency, eff);

            if (node.is_dead()) metrics.dead_nodes++;
            if (node.is_saturated()) metrics.saturated_nodes++;
            if (node_metrics.status == "normal") metrics.active_nodes++;
        }

        metrics.avg_node_efficiency = efficiency_sum / nodes_.size();

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
     */
    void add_nodes(size_t count) {
        size_t old_output_size = output_size_;
        output_size_ += count;

        // Add new nodes
        for (size_t i = 0; i < count; ++i) {
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

        // Reset gradient accumulators
        weight_gradients_ = Tensor<T>();
        bias_gradients_ = Tensor<T>();
    }

    /**
     * Remove nodes from the layer.
     * @param indices Indices of nodes to remove (must be sorted ascending)
     */
    void remove_nodes(std::vector<size_t> indices) {
        if (indices.empty()) return;

        // Sort indices in descending order for removal
        std::sort(indices.begin(), indices.end(), std::greater<size_t>());

        // Remove from highest index first
        for (size_t idx : indices) {
            if (idx < nodes_.size()) {
                nodes_.erase(nodes_.begin() + idx);
            }
        }

        // Rebuild weights matrix
        size_t new_output_size = nodes_.size();
        Tensor<T> new_weights(std::vector<size_t>{new_output_size, input_size_});
        Tensor<T> new_biases(std::vector<size_t>{new_output_size});

        size_t dest_i = 0;
        for (size_t i = 0; i < output_size_; ++i) {
            bool keep = std::find(indices.begin(), indices.end(), i) == indices.end();
            if (keep) {
                for (size_t j = 0; j < input_size_; ++j) {
                    new_weights.at(dest_i, j) = weights_.at(i, j);
                }
                new_biases[dest_i] = biases_[i];
                dest_i++;
            }
        }

        weights_ = std::move(new_weights);
        biases_ = std::move(new_biases);
        output_size_ = new_output_size;

        // Update node indices
        for (size_t i = 0; i < nodes_.size(); ++i) {
            // Node indices would need to be updated - they're stored internally
        }

        // Reset gradient accumulators
        weight_gradients_ = Tensor<T>();
        bias_gradients_ = Tensor<T>();
    }

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
        return copy;
    }

private:
    LayerType layer_type_;
    size_t input_size_;
    size_t output_size_;
    ActivationType activation_type_;

    Tensor<T> weights_;
    Tensor<T> biases_;
    std::vector<Node> nodes_;

    // Cached values for backward pass
    Tensor<T> cached_input_;
    Tensor<T> cached_pre_activation_;
    Tensor<T> cached_output_;

    // Gradient accumulators
    Tensor<T> weight_gradients_;
    Tensor<T> bias_gradients_;

    std::unique_ptr<Activation<T>> activation_;
    std::unique_ptr<Random> rng_;
};

// Type alias
using LayerF = Layer<float>;
using LayerD = Layer<double>;

} // namespace core
} // namespace dnn
