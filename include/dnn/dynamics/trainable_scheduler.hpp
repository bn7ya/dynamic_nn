#pragma once

#include "../core/network.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

namespace dnn {
namespace dynamics {

using core::Network;

/**
 * Schedule types for trainable node reduction.
 */
enum class TrainableSchedule {
    Linear,          // Linear decrease
    Exponential,     // Exponential decrease
    Cosine,          // Cosine annealing
    Step             // Step-wise decrease
};

/**
 * Configuration for trainable scheduler.
 */
struct TrainableConfig {
    double initial_fraction = 1.0;      // Start with 100% trainable
    double final_fraction = 0.01;       // End with 1% trainable
    uint64_t transition_epochs = 100;   // Over how many epochs
    TrainableSchedule schedule = TrainableSchedule::Exponential;
    size_t step_count = 5;              // For Step schedule
};

/**
 * Selection strategy for which nodes remain trainable.
 */
enum class SelectionStrategy {
    LeastEfficient,   // Keep training least efficient nodes (need improvement)
    MostEfficient,    // Keep training most efficient nodes
    Random,           // Random selection
    Distributed       // Even distribution across layers
};

/**
 * Schedules the reduction of trainable nodes from 100% to 1%.
 * This allows the network to "freeze" learned patterns while
 * continuing to train underperforming areas.
 */
template<typename T = float>
class TrainableScheduler {
public:
    TrainableScheduler(Network<T>& network,
                       const TrainableConfig& config = TrainableConfig())
        : network_(network)
        , config_(config)
        , current_epoch_(0)
        , strategy_(SelectionStrategy::LeastEfficient) {}

    /**
     * Update for new epoch.
     */
    void update_epoch(uint64_t epoch) {
        current_epoch_ = epoch;
        apply_trainability();
    }

    /**
     * Get current trainable fraction.
     */
    double current_trainable_fraction() const {
        return compute_fraction_for_epoch(current_epoch_);
    }

    /**
     * Get number of trainable nodes.
     */
    size_t trainable_node_count() const {
        return network_.num_trainable_nodes();
    }

    /**
     * Get total number of nodes.
     */
    size_t total_node_count() const {
        return network_.num_nodes();
    }

    /**
     * Set selection strategy.
     */
    void set_selection_strategy(SelectionStrategy strategy) {
        strategy_ = strategy;
    }

    /**
     * Manually set trainable fraction.
     */
    void set_trainable_fraction(double fraction) {
        fraction = std::max(config_.final_fraction,
                           std::min(config_.initial_fraction, fraction));
        network_.set_trainable_fraction(fraction);
    }

    /**
     * Get configuration.
     */
    const TrainableConfig& config() const { return config_; }

    /**
     * Get schedule visualization data.
     */
    std::vector<std::pair<uint64_t, double>> get_schedule_curve(
            size_t num_points = 100) const {
        std::vector<std::pair<uint64_t, double>> curve;
        curve.reserve(num_points);

        for (size_t i = 0; i < num_points; ++i) {
            uint64_t epoch = (i * config_.transition_epochs) / num_points;
            double fraction = compute_fraction_for_epoch(epoch);
            curve.emplace_back(epoch, fraction);
        }

        return curve;
    }

private:
    double compute_fraction_for_epoch(uint64_t epoch) const {
        if (epoch >= config_.transition_epochs) {
            return config_.final_fraction;
        }

        double progress = static_cast<double>(epoch) / config_.transition_epochs;
        double range = config_.initial_fraction - config_.final_fraction;

        switch (config_.schedule) {
            case TrainableSchedule::Linear:
                return config_.initial_fraction - progress * range;

            case TrainableSchedule::Exponential: {
                // Exponential decay: f(t) = final + (initial - final) * exp(-k*t)
                // Solve for k such that f(1) = final + epsilon
                double k = 5.0;  // Decay constant
                double decay = std::exp(-k * progress);
                return config_.final_fraction + range * decay;
            }

            case TrainableSchedule::Cosine: {
                // Cosine annealing: f(t) = final + 0.5*(initial-final)*(1+cos(pi*t))
                double cosine = 0.5 * (1.0 + std::cos(M_PI * progress));
                return config_.final_fraction + range * cosine;
            }

            case TrainableSchedule::Step: {
                // Step-wise decrease
                size_t step = static_cast<size_t>(progress * config_.step_count);
                double step_size = range / config_.step_count;
                return config_.initial_fraction - step * step_size;
            }

            default:
                return config_.initial_fraction - progress * range;
        }
    }

    void apply_trainability() {
        double fraction = current_trainable_fraction();
        size_t total = network_.num_nodes();
        size_t target_trainable = static_cast<size_t>(fraction * total);
        target_trainable = std::max(size_t(1), target_trainable);

        switch (strategy_) {
            case SelectionStrategy::LeastEfficient:
                apply_least_efficient_selection(target_trainable);
                break;
            case SelectionStrategy::MostEfficient:
                apply_most_efficient_selection(target_trainable);
                break;
            case SelectionStrategy::Random:
                apply_random_selection(target_trainable);
                break;
            case SelectionStrategy::Distributed:
                apply_distributed_selection(target_trainable);
                break;
        }
    }

    void apply_least_efficient_selection(size_t target) {
        // Collect all nodes with efficiency scores
        std::vector<std::tuple<double, size_t, size_t>> all_nodes;

        for (size_t l = 0; l < network_.num_layers(); ++l) {
            auto& layer = network_.layer(l);
            for (size_t n = 0; n < layer.num_nodes(); ++n) {
                double eff = layer.node(n).efficiency_score();
                all_nodes.emplace_back(eff, l, n);
            }
        }

        // Sort by efficiency (ascending - least efficient first)
        std::sort(all_nodes.begin(), all_nodes.end());

        // Set trainability
        for (size_t i = 0; i < all_nodes.size(); ++i) {
            auto [eff, layer_idx, node_idx] = all_nodes[i];
            bool trainable = (i < target);
            network_.layer(layer_idx).node(node_idx).set_trainable(trainable);
        }
    }

    void apply_most_efficient_selection(size_t target) {
        std::vector<std::tuple<double, size_t, size_t>> all_nodes;

        for (size_t l = 0; l < network_.num_layers(); ++l) {
            auto& layer = network_.layer(l);
            for (size_t n = 0; n < layer.num_nodes(); ++n) {
                double eff = layer.node(n).efficiency_score();
                all_nodes.emplace_back(eff, l, n);
            }
        }

        // Sort by efficiency (descending - most efficient first)
        std::sort(all_nodes.begin(), all_nodes.end(), std::greater<>());

        for (size_t i = 0; i < all_nodes.size(); ++i) {
            auto [eff, layer_idx, node_idx] = all_nodes[i];
            bool trainable = (i < target);
            network_.layer(layer_idx).node(node_idx).set_trainable(trainable);
        }
    }

    void apply_random_selection(size_t target) {
        std::vector<std::pair<size_t, size_t>> all_nodes;

        for (size_t l = 0; l < network_.num_layers(); ++l) {
            for (size_t n = 0; n < network_.layer(l).num_nodes(); ++n) {
                all_nodes.emplace_back(l, n);
            }
        }

        // Shuffle
        core::Random rng(current_epoch_);  // Deterministic per epoch
        std::vector<size_t> indices(all_nodes.size());
        std::iota(indices.begin(), indices.end(), 0);
        rng.shuffle(indices);

        // Set trainability
        for (size_t i = 0; i < all_nodes.size(); ++i) {
            size_t idx = indices[i];
            auto [layer_idx, node_idx] = all_nodes[idx];
            bool trainable = (i < target);
            network_.layer(layer_idx).node(node_idx).set_trainable(trainable);
        }
    }

    void apply_distributed_selection(size_t target) {
        // Distribute trainable nodes evenly across layers
        size_t num_layers = network_.num_layers();
        size_t per_layer = target / num_layers;
        size_t remainder = target % num_layers;

        for (size_t l = 0; l < num_layers; ++l) {
            auto& layer = network_.layer(l);
            size_t layer_target = per_layer + (l < remainder ? 1 : 0);
            layer_target = std::min(layer_target, layer.num_nodes());

            // Select least efficient nodes in this layer
            std::vector<std::pair<double, size_t>> layer_nodes;
            for (size_t n = 0; n < layer.num_nodes(); ++n) {
                layer_nodes.emplace_back(layer.node(n).efficiency_score(), n);
            }
            std::sort(layer_nodes.begin(), layer_nodes.end());

            for (size_t i = 0; i < layer.num_nodes(); ++i) {
                size_t node_idx = layer_nodes[i].second;
                layer.node(node_idx).set_trainable(i < layer_target);
            }
        }
    }

    Network<T>& network_;
    TrainableConfig config_;
    uint64_t current_epoch_;
    SelectionStrategy strategy_;
};

} // namespace dynamics
} // namespace dnn
