#pragma once

#include "../core/network.hpp"
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <string>

namespace dnn {
namespace training {

using core::Network;

/**
 * Efficiency metrics for a training epoch.
 */
struct EfficiencyMetrics {
    double node_utilization = 0.0;        // % of nodes contributing meaningfully
    double gradient_health = 0.0;         // Gradient flow quality (no vanishing/exploding)
    double parameter_efficiency = 0.0;    // Accuracy per parameter
    double compute_efficiency = 0.0;      // Accuracy per FLOP (estimated)
    double convergence_rate = 0.0;        // Rate of improvement
    double stability_score = 0.0;         // Variance in recent performance
    double overall_efficiency = 0.0;      // Combined score
};

/**
 * Stopping decision.
 */
struct StoppingDecision {
    bool should_stop = false;
    std::string reason;
    double final_efficiency = 0.0;
    double final_cost = 0.0;
    uint64_t epochs_trained = 0;
    size_t no_improvement_epochs = 0;
};

/**
 * Early stopping based on efficiency metrics, NOT just cost.
 *
 * This differs from traditional early stopping which only looks at
 * validation loss. Instead, we consider:
 * - How well the network utilizes its parameters
 * - Gradient flow health
 * - Convergence rate and stability
 */
template<typename T = float>
class EarlyStopping {
public:
    EarlyStopping(size_t patience = 10,
                  double min_improvement = 0.001,
                  size_t min_epochs = 10)
        : patience_(patience)
        , min_improvement_(min_improvement)
        , min_epochs_(min_epochs)
        , best_efficiency_(0.0)
        , epochs_without_improvement_(0)
        , total_epochs_(0) {}

    /**
     * Record metrics for an epoch.
     */
    void record_epoch(const EfficiencyMetrics& metrics, double cost) {
        total_epochs_++;
        cost_history_.push_back(cost);
        efficiency_history_.push_back(metrics.overall_efficiency);

        // Check for improvement in efficiency
        if (metrics.overall_efficiency > best_efficiency_ + min_improvement_) {
            best_efficiency_ = metrics.overall_efficiency;
            epochs_without_improvement_ = 0;
        } else {
            epochs_without_improvement_++;
        }

        // Store recent history for stability calculation
        if (recent_efficiencies_.size() >= 10) {
            recent_efficiencies_.pop_front();
        }
        recent_efficiencies_.push_back(metrics.overall_efficiency);
    }

    /**
     * Compute efficiency metrics from network state.
     */
    template<typename NetworkT>
    EfficiencyMetrics compute_metrics(const Network<NetworkT>& network,
                                      double current_cost,
                                      double previous_cost) const {
        EfficiencyMetrics metrics;

        auto network_metrics = network.compute_metrics();

        // Node utilization: fraction of nodes that are "active"
        size_t active_nodes = network_metrics.total_nodes -
                             network_metrics.dead_nodes -
                             network_metrics.saturated_nodes;
        metrics.node_utilization = static_cast<double>(active_nodes) /
                                   network_metrics.total_nodes;

        // Gradient health: based on layer efficiency variance
        metrics.gradient_health = network_metrics.is_healthy ? 1.0 : 0.5;

        // Parameter efficiency: inversely related to parameters per accuracy point
        // Higher is better (more accuracy per parameter)
        if (current_cost > 0.001) {
            double accuracy_proxy = 1.0 / current_cost;
            metrics.parameter_efficiency = std::min(1.0,
                accuracy_proxy / (network_metrics.total_parameters / 1000.0));
        }

        // Convergence rate
        if (previous_cost > 0) {
            double improvement = (previous_cost - current_cost) / previous_cost;
            metrics.convergence_rate = std::max(0.0, std::min(1.0, improvement * 10));
        }

        // Stability: based on variance in recent efficiencies
        if (recent_efficiencies_.size() >= 3) {
            double mean = 0;
            for (double e : recent_efficiencies_) mean += e;
            mean /= recent_efficiencies_.size();

            double variance = 0;
            for (double e : recent_efficiencies_) {
                double diff = e - mean;
                variance += diff * diff;
            }
            variance /= recent_efficiencies_.size();

            // Lower variance = higher stability
            metrics.stability_score = 1.0 / (1.0 + variance * 10);
        } else {
            metrics.stability_score = 0.5;
        }

        // Overall efficiency: weighted combination
        metrics.overall_efficiency =
            0.25 * metrics.node_utilization +
            0.20 * metrics.gradient_health +
            0.20 * metrics.parameter_efficiency +
            0.20 * metrics.convergence_rate +
            0.15 * metrics.stability_score;

        return metrics;
    }

    /**
     * Check if training should stop.
     */
    StoppingDecision should_stop() const {
        StoppingDecision decision;
        decision.epochs_trained = total_epochs_;
        decision.no_improvement_epochs = epochs_without_improvement_;
        decision.final_efficiency = best_efficiency_;

        if (!cost_history_.empty()) {
            decision.final_cost = cost_history_.back();
        }

        // Don't stop before minimum epochs
        if (total_epochs_ < min_epochs_) {
            decision.should_stop = false;
            decision.reason = "Below minimum epochs";
            return decision;
        }

        // Check patience
        if (epochs_without_improvement_ >= patience_) {
            decision.should_stop = true;
            decision.reason = "Efficiency plateau for " +
                             std::to_string(patience_) + " epochs";
            return decision;
        }

        // Check for efficiency regression
        if (efficiency_history_.size() >= 5) {
            double recent_avg = 0;
            for (size_t i = efficiency_history_.size() - 5; i < efficiency_history_.size(); ++i) {
                recent_avg += efficiency_history_[i];
            }
            recent_avg /= 5;

            if (recent_avg < best_efficiency_ * 0.8) {
                decision.should_stop = true;
                decision.reason = "Significant efficiency regression";
                return decision;
            }
        }

        // Check for extremely low efficiency
        if (best_efficiency_ < 0.1 && total_epochs_ > min_epochs_ * 2) {
            decision.should_stop = true;
            decision.reason = "Efficiency remains critically low";
            return decision;
        }

        decision.should_stop = false;
        decision.reason = "Training should continue";
        return decision;
    }

    /**
     * Get current status message.
     */
    std::string status_message() const {
        auto decision = should_stop();
        std::ostringstream oss;
        oss << "Epoch " << total_epochs_
            << ", Best Efficiency: " << best_efficiency_
            << ", No improvement: " << epochs_without_improvement_
            << "/" << patience_
            << " - " << decision.reason;
        return oss.str();
    }

    /**
     * Get best efficiency score.
     */
    double best_efficiency() const { return best_efficiency_; }

    /**
     * Get efficiency history.
     */
    const std::vector<double>& efficiency_history() const { return efficiency_history_; }

    /**
     * Get cost history.
     */
    const std::vector<double>& cost_history() const { return cost_history_; }

    /**
     * Reset the early stopping tracker.
     */
    void reset() {
        cost_history_.clear();
        efficiency_history_.clear();
        recent_efficiencies_.clear();
        best_efficiency_ = 0.0;
        epochs_without_improvement_ = 0;
        total_epochs_ = 0;
    }

    // Configuration
    void set_patience(size_t patience) { patience_ = patience; }
    void set_min_improvement(double improvement) { min_improvement_ = improvement; }
    void set_min_epochs(size_t epochs) { min_epochs_ = epochs; }

private:
    size_t patience_;
    double min_improvement_;
    size_t min_epochs_;

    std::vector<double> cost_history_;
    std::vector<double> efficiency_history_;
    std::deque<double> recent_efficiencies_;

    double best_efficiency_;
    size_t epochs_without_improvement_;
    size_t total_epochs_;
};

} // namespace training
} // namespace dnn
