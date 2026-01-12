#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <atomic>
#include <array>
#include <string>

namespace dnn {
namespace core {

/**
 * Metrics for tracking node efficiency.
 */
struct NodeMetrics {
    double activation_mean = 0.0;       // Mean activation value
    double activation_variance = 0.0;   // Variance in activations
    double gradient_magnitude_avg = 0.0; // Average gradient magnitude
    double contribution_score = 0.0;    // Contribution to layer output
    uint64_t activation_count = 0;      // Number of samples processed
    uint64_t dead_count = 0;            // Times activation was zero (for ReLU)
    double efficiency_score = 0.0;      // Computed efficiency (0.0 - 1.0)
    std::string status = "normal";      // "normal", "dead", "saturated", "underutilized"
};

/**
 * Adaptive weights for efficiency score computation.
 * Weights adapt based on network behavior and are constrained to sum to 1.
 */
struct EfficiencyWeights {
    double w_variance = 0.25;
    double w_gradient = 0.30;
    double w_alive = 0.25;           // Static, never changes
    double w_contribution = 0.20;
    double grad_threshold = 0.1;     // Auto-detected from data

    /**
     * Normalize weights to sum to 1.0
     */
    void normalize() {
        double sum = w_variance + w_gradient + w_alive + w_contribution;
        if (sum > 0) {
            w_variance /= sum;
            w_gradient /= sum;
            w_alive /= sum;
            w_contribution /= sum;
        }
    }

    /**
     * Adapt variance weight based on z-score.
     * High z-score (high variance) -> lower weight (already good)
     * Low z-score (low variance) -> higher weight (need improvement)
     */
    void adapt_variance_from_zscore(double z) {
        if (z > 1.0) {
            w_variance = 0.15;  // High variance already, less emphasis
        } else if (z < -1.0) {
            w_variance = 0.35;  // Low variance, more emphasis needed
        } else {
            w_variance = 0.25;  // Normal range
        }
        normalize();
    }

    /**
     * Adapt gradient/contribution weights based on gradient magnitude.
     * High gradients -> lower gradient weight, higher contribution weight.
     */
    void adapt_gradient_contribution(double grad_mag) {
        if (grad_mag > grad_threshold) {
            double excess = (grad_mag - grad_threshold) / grad_threshold;
            double transfer = std::min(0.15, excess * 0.10);

            w_gradient = std::max(0.10, 0.30 - transfer);
            w_contribution = std::min(0.40, 0.20 + transfer);
            normalize();
        }
    }

    /**
     * Set gradient threshold (auto-detected from data).
     */
    void set_grad_threshold(double threshold) {
        grad_threshold = threshold;
    }

    /**
     * Get current weights as array [var, grad, alive, contrib].
     */
    std::array<double, 4> get_weights() const {
        return {w_variance, w_gradient, w_alive, w_contribution};
    }
};

/**
 * Individual neuron/node in a neural network layer.
 * Tracks efficiency metrics for dynamic adjustment.
 */
class Node {
public:
    /**
     * Construct a node.
     * @param index Position in the layer
     * @param trainable Whether this node's weights can be updated
     */
    explicit Node(size_t index, bool trainable = true)
        : index_(index)
        , trainable_(trainable)
        , activation_(0.0f)
        , bias_(0.0f)
        , activation_sum_(0.0)
        , activation_sq_sum_(0.0)
        , gradient_sum_(0.0)
        , gradient_sq_sum_(0.0)
        , contribution_sum_(0.0)
        , sample_count_(0)
        , dead_count_(0) {}

    // Identification
    size_t index() const { return index_; }

    // Current state
    float activation() const { return activation_; }
    void set_activation(float value) { activation_ = value; }

    float bias() const { return bias_; }
    void set_bias(float value) { bias_ = value; }

    // Trainability
    bool is_trainable() const { return trainable_; }
    void set_trainable(bool trainable) { trainable_ = trainable; }

    /**
     * Record activation value for metrics tracking.
     */
    void record_activation(float value) {
        activation_ = value;
        activation_sum_ += value;
        activation_sq_sum_ += value * value;
        sample_count_++;

        // Track dead neurons (for ReLU-like activations)
        if (std::abs(value) < 1e-7f) {
            dead_count_++;
        }
    }

    /**
     * Record gradient for metrics tracking.
     */
    void record_gradient(float gradient) {
        double abs_grad = std::abs(gradient);
        gradient_sum_ += abs_grad;
        gradient_sq_sum_ += abs_grad * abs_grad;
    }

    /**
     * Record contribution to layer output.
     */
    void record_contribution(float contribution) {
        contribution_sum_ += std::abs(contribution);
    }

    /**
     * Compute current efficiency metrics.
     */
    NodeMetrics compute_metrics() const {
        NodeMetrics metrics;

        if (sample_count_ == 0) {
            metrics.efficiency_score = 0.5;  // Unknown
            metrics.status = "unknown";
            return metrics;
        }

        // Activation statistics
        metrics.activation_count = sample_count_;
        metrics.activation_mean = activation_sum_ / sample_count_;
        double mean_sq = activation_sq_sum_ / sample_count_;
        metrics.activation_variance = mean_sq - metrics.activation_mean * metrics.activation_mean;

        // Gradient statistics
        metrics.gradient_magnitude_avg = gradient_sum_ / sample_count_;

        // Contribution score
        metrics.contribution_score = contribution_sum_ / sample_count_;

        // Dead count
        metrics.dead_count = dead_count_;

        // Compute efficiency score
        metrics.efficiency_score = compute_efficiency_score(metrics);

        // Determine status
        metrics.status = determine_status(metrics);

        return metrics;
    }

    /**
     * Reset metrics for new training phase.
     */
    void reset_metrics() {
        activation_sum_ = 0.0;
        activation_sq_sum_ = 0.0;
        gradient_sum_ = 0.0;
        gradient_sq_sum_ = 0.0;
        contribution_sum_ = 0.0;
        sample_count_ = 0;
        dead_count_ = 0;
    }

    /**
     * Get efficiency score (0.0 - 1.0).
     */
    double efficiency_score() const {
        return compute_metrics().efficiency_score;
    }

    /**
     * Check if node is considered "dead" (always outputs zero).
     */
    bool is_dead() const {
        if (sample_count_ == 0) return false;
        return (static_cast<double>(dead_count_) / sample_count_) > 0.99;
    }

    /**
     * Check if node is saturated (always at extreme values).
     */
    bool is_saturated() const {
        auto metrics = compute_metrics();
        return metrics.activation_variance < 0.001 &&
               std::abs(metrics.activation_mean) > 0.9;
    }

    /**
     * Get current efficiency weights.
     */
    const EfficiencyWeights& efficiency_weights() const {
        return weights_;
    }

    /**
     * Get mutable efficiency weights.
     */
    EfficiencyWeights& efficiency_weights() {
        return weights_;
    }

    /**
     * Adapt variance weight based on z-score (called once before training).
     */
    void adapt_variance_weight(double z_score) {
        weights_.adapt_variance_from_zscore(z_score);
    }

    /**
     * Adapt gradient/contribution weights based on gradient magnitude.
     */
    void adapt_gradient_weight() {
        if (sample_count_ > 0) {
            double grad_mag = gradient_sum_ / sample_count_;
            weights_.adapt_gradient_contribution(grad_mag);
        }
    }

    /**
     * Set gradient threshold for adaptive weights.
     */
    void set_grad_threshold(double threshold) {
        weights_.set_grad_threshold(threshold);
    }

private:
    double compute_efficiency_score(const NodeMetrics& metrics) const {
        // Weighted combination of factors using adaptive weights

        // 1. Variance score: nodes should have varied activations
        //    Too low variance = potentially dead or saturated
        double variance_score = std::min(1.0, metrics.activation_variance / 0.25);

        // 2. Gradient score: gradients should flow
        //    Zero gradients = not learning
        double gradient_score = std::min(1.0, metrics.gradient_magnitude_avg / 0.1);

        // 3. Dead ratio penalty
        double dead_ratio = static_cast<double>(dead_count_) / sample_count_;
        double alive_score = 1.0 - dead_ratio;

        // 4. Contribution score
        double contribution_score = std::min(1.0, metrics.contribution_score / 0.1);

        // Weighted combination using adaptive weights
        double efficiency =
            weights_.w_variance * variance_score +
            weights_.w_gradient * gradient_score +
            weights_.w_alive * alive_score +
            weights_.w_contribution * contribution_score;

        return std::max(0.0, std::min(1.0, efficiency));
    }

    std::string determine_status(const NodeMetrics& metrics) const {
        double dead_ratio = static_cast<double>(dead_count_) / sample_count_;

        if (dead_ratio > 0.99) {
            return "dead";
        }
        if (metrics.activation_variance < 0.001 &&
            std::abs(metrics.activation_mean) > 0.9) {
            return "saturated";
        }
        if (metrics.gradient_magnitude_avg < 1e-7) {
            return "vanishing_gradient";
        }
        if (metrics.efficiency_score < 0.3) {
            return "underutilized";
        }
        return "normal";
    }

    size_t index_;
    bool trainable_;
    float activation_;
    float bias_;

    // Running statistics for efficiency tracking
    double activation_sum_;
    double activation_sq_sum_;
    double gradient_sum_;
    double gradient_sq_sum_;
    double contribution_sum_;
    uint64_t sample_count_;
    uint64_t dead_count_;

    // Adaptive efficiency weights
    EfficiencyWeights weights_;
};

} // namespace core
} // namespace dnn
