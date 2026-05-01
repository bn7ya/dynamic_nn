#pragma once

#include "../core/network.hpp"
#include "../exceptions/dnn_exception.hpp"
#include "health_monitor.hpp"
#include <memory>
#include <vector>

namespace dnn {
namespace dynamics {

using core::Network;
using core::ActivationType;

/**
 * Tunable thresholds that gate node/layer add and remove decisions.
 * Surfaced through TrainerConfig so callers (and the dynamic-thresholds
 * Python layer) can override the defaults before training starts.
 *
 * Defaults match the historical hardcoded LayerManager constructor body
 * so the 2-arg constructor and (network, health_monitor, LayerManagerConfig{})
 * are bit-identical.
 */
struct LayerManagerConfig {
    double efficiency_threshold = 0.5;   // Below this, nodes are "inefficient"
    double saturation_threshold = 0.7;   // Above this, layer is saturated
    double redundancy_threshold = 0.01;  // Below this, a layer is redundant
};

/**
 * Decision made by the layer manager.
 */
struct LayerDecision {
    enum class Action {
        None,
        AddLayer,
        RemoveLayer,
        AddNodes,
        RemoveNodes,
        AddAndRemove  // Both operations
    };

    Action action = Action::None;
    size_t layer_index = 0;
    size_t node_count = 0;
    std::vector<size_t> nodes_to_remove;
    std::string reason;
    double confidence = 0.0;  // 0.0 - 1.0
};

/**
 * Manages dynamic layer creation and removal.
 * Analyzes network efficiency to make structural decisions.
 */
template<typename T = float>
class LayerManager {
public:
    LayerManager(Network<T>& network, HealthMonitor<T>& health_monitor)
        : LayerManager(network, health_monitor, LayerManagerConfig{}) {}

    LayerManager(Network<T>& network,
                 HealthMonitor<T>& health_monitor,
                 const LayerManagerConfig& config)
        : network_(network)
        , health_monitor_(health_monitor)
        , min_layers_(2)
        , max_layers_(100)
        , min_nodes_per_layer_(4)
        , max_nodes_per_layer_(10000)
        , efficiency_threshold_(config.efficiency_threshold)
        , saturation_threshold_(config.saturation_threshold)
        , redundancy_threshold_(config.redundancy_threshold)
        , adaptive_threshold_enabled_(true)
        // sigmoid_{k,base,range}: shape parameters of the saturation-curve
        // mapping efficiency in [0,1] to the per-epoch saturation threshold.
        // Tuned to keep the curve gentle around 0.5 efficiency (the typical
        // operating point) while still bottoming out near base=0.3 and topping
        // out near base+range=0.8 at the extremes. These are functional-form
        // parameters; user-overridable thresholds live in LayerManagerConfig.
        , sigmoid_k_(5.0)
        , sigmoid_base_(0.3)
        , sigmoid_range_(0.5) {
        // Validate config consistency. The decision logic in
        // analyze_with_efficiency() compares against efficiency_threshold and
        // saturation_threshold; inverting them silently breaks add-vs-remove
        // routing.
        if (!(config.efficiency_threshold >= 0.0 &&
              config.efficiency_threshold <= 1.0)) {
            throw exceptions::InvalidArgumentException("efficiency_threshold",
                "efficiency_threshold must be in [0, 1]");
        }
        if (!(config.saturation_threshold >= 0.0 &&
              config.saturation_threshold <= 1.0)) {
            throw exceptions::InvalidArgumentException("saturation_threshold",
                "saturation_threshold must be in [0, 1]");
        }
        if (config.efficiency_threshold >= config.saturation_threshold) {
            throw exceptions::InvalidArgumentException("LayerManagerConfig",
                "efficiency_threshold must be strictly less than "
                "saturation_threshold; otherwise the add-vs-remove decision "
                "logic collapses.");
        }
        if (!(config.redundancy_threshold >= 0.0 &&
              config.redundancy_threshold <= 1.0)) {
            throw exceptions::InvalidArgumentException("redundancy_threshold",
                "redundancy_threshold must be in [0, 1]");
        }
    }

    /**
     * Compute adaptive saturation threshold using sigmoid function.
     * Low efficiency -> lower threshold -> more aggressive changes
     * High efficiency -> higher threshold -> preserve architecture
     */
    double compute_adaptive_saturation_threshold(double efficiency) const {
        if (!adaptive_threshold_enabled_) {
            return saturation_threshold_;
        }
        // Sigmoid-based threshold: base + range * sigmoid(k * (efficiency - 0.5))
        double sigmoid = 1.0 / (1.0 + std::exp(-sigmoid_k_ * (efficiency - 0.5)));
        return sigmoid_base_ + sigmoid_range_ * sigmoid;
    }

    /**
     * Enable/disable adaptive saturation threshold.
     */
    void set_adaptive_threshold_enabled(bool enabled) {
        adaptive_threshold_enabled_ = enabled;
    }

    /**
     * Configure sigmoid parameters for adaptive threshold.
     */
    void set_sigmoid_params(double k, double base, double range) {
        sigmoid_k_ = k;
        sigmoid_base_ = base;
        sigmoid_range_ = range;
    }

    /**
     * Analyze network and recommend structural changes.
     * Uses default saturation threshold.
     */
    LayerDecision analyze() const {
        return analyze_with_efficiency(0.5);  // Default efficiency
    }

    /**
     * Analyze network with adaptive threshold based on current efficiency.
     */
    LayerDecision analyze_with_efficiency(double current_efficiency) const {
        LayerDecision decision;

        // Compute adaptive saturation threshold
        double adaptive_sat_threshold = compute_adaptive_saturation_threshold(current_efficiency);

        // Check health first
        auto health = health_monitor_.diagnose();
        if (health.state == HealthState::Critical) {
            decision.action = LayerDecision::Action::None;
            decision.reason = "Network in critical state";
            return decision;
        }

        // Check if we should add a layer (use adaptive threshold)
        if (should_add_layer_adaptive(adaptive_sat_threshold)) {
            decision.action = LayerDecision::Action::AddLayer;
            decision.layer_index = find_best_insertion_point();
            decision.node_count = compute_new_layer_size(decision.layer_index);
            decision.reason = "High saturation, more capacity needed";
            decision.confidence = 0.8;
            return decision;
        }

        // Check if we should remove a layer
        if (should_remove_layer()) {
            decision.action = LayerDecision::Action::RemoveLayer;
            decision.layer_index = find_least_efficient_layer();
            decision.reason = "Redundant layer detected";
            decision.confidence = 0.7;
            return decision;
        }

        // Check individual layers for node adjustments
        for (size_t i = 0; i < network_.num_layers(); ++i) {
            auto layer_metrics = network_.layer(i).compute_metrics();

            // Add nodes if layer is saturated (use adaptive threshold)
            if (layer_metrics.avg_node_efficiency > adaptive_sat_threshold &&
                network_.layer(i).num_nodes() < max_nodes_per_layer_) {
                decision.action = LayerDecision::Action::AddNodes;
                decision.layer_index = i;
                decision.node_count = network_.layer(i).num_nodes() / 4;  // 25% growth
                decision.node_count = std::max(size_t(1), decision.node_count);
                decision.reason = "Layer is saturated, needs more capacity";
                decision.confidence = 0.6;
                return decision;
            }

            // Remove inefficient nodes
            if (layer_metrics.avg_node_efficiency < efficiency_threshold_ &&
                layer_metrics.dead_nodes > 0 &&
                network_.layer(i).num_nodes() > min_nodes_per_layer_) {

                decision.action = LayerDecision::Action::RemoveNodes;
                decision.layer_index = i;

                // Find nodes to remove
                for (size_t j = 0; j < network_.layer(i).num_nodes(); ++j) {
                    auto& node = network_.layer(i).node(j);
                    if (node.efficiency_score() < efficiency_threshold_ * 0.5) {
                        decision.nodes_to_remove.push_back(j);
                    }
                }

                // Keep minimum nodes
                while (network_.layer(i).num_nodes() - decision.nodes_to_remove.size() <
                       min_nodes_per_layer_) {
                    decision.nodes_to_remove.pop_back();
                }

                if (!decision.nodes_to_remove.empty()) {
                    decision.reason = "Removing inefficient nodes";
                    decision.confidence = 0.65;
                    return decision;
                }
            }
        }

        decision.action = LayerDecision::Action::None;
        decision.reason = "No structural changes needed";
        return decision;
    }

    /**
     * Execute a decision.
     */
    void execute(const LayerDecision& decision) {
        switch (decision.action) {
            case LayerDecision::Action::AddLayer:
                add_layer(decision.layer_index, decision.node_count);
                break;

            case LayerDecision::Action::RemoveLayer:
                remove_layer(decision.layer_index);
                break;

            case LayerDecision::Action::AddNodes:
                add_nodes(decision.layer_index, decision.node_count);
                break;

            case LayerDecision::Action::RemoveNodes:
                remove_nodes(decision.layer_index, decision.nodes_to_remove);
                break;

            case LayerDecision::Action::None:
            default:
                break;
        }
    }

    /**
     * Analyze and automatically execute recommended changes.
     */
    LayerDecision auto_adjust() {
        auto decision = analyze();
        if (decision.action != LayerDecision::Action::None) {
            execute(decision);
        }
        return decision;
    }

    /**
     * Add a new layer.
     */
    void add_layer(size_t after_index, size_t num_nodes) {
        if (!health_monitor_.allow_layer_addition()) {
            return;
        }
        if (network_.num_layers() >= max_layers_) {
            return;
        }

        network_.insert_layer(after_index + 1, num_nodes);
        health_monitor_.record_change(1, static_cast<int>(num_nodes));
    }

    /**
     * Remove a layer (soft).
     *
     * Marks the layer inactive without dropping its weights, so the
     * controller can reactivate it later. Hard removal is deferred to
     * Network::compact() at end of training.
     */
    void remove_layer(size_t index) {
        if (!health_monitor_.allow_layer_removal()) {
            return;
        }
        if (network_.num_layers() <= min_layers_) {
            return;
        }

        size_t nodes_removed = network_.layer(index).num_nodes();
        network_.mark_layer_inactive(index);
        health_monitor_.record_change(-1, -static_cast<int>(nodes_removed));
    }

    /**
     * Add nodes to a layer.
     */
    void add_nodes(size_t layer_index, size_t count) {
        if (layer_index >= network_.num_layers()) return;

        auto& layer = network_.layer(layer_index);
        if (layer.num_nodes() + count > max_nodes_per_layer_) {
            count = max_nodes_per_layer_ - layer.num_nodes();
        }

        if (count > 0) {
            layer.add_nodes(count);
            health_monitor_.record_change(0, static_cast<int>(count));
        }
    }

    /**
     * Remove nodes from a layer.
     */
    void remove_nodes(size_t layer_index, const std::vector<size_t>& indices) {
        if (layer_index >= network_.num_layers()) return;

        auto& layer = network_.layer(layer_index);
        if (layer.num_nodes() - indices.size() < min_nodes_per_layer_) {
            return;
        }

        std::vector<size_t> sorted_indices = indices;
        layer.remove_nodes(sorted_indices);
        health_monitor_.record_change(0, -static_cast<int>(indices.size()));
    }

    // Configuration
    void set_min_layers(size_t min) { min_layers_ = min; }
    void set_max_layers(size_t max) { max_layers_ = max; }
    void set_min_nodes_per_layer(size_t min) { min_nodes_per_layer_ = min; }
    void set_max_nodes_per_layer(size_t max) { max_nodes_per_layer_ = max; }
    void set_efficiency_threshold(double threshold) { efficiency_threshold_ = threshold; }
    void set_saturation_threshold(double threshold) { saturation_threshold_ = threshold; }

private:
    bool should_add_layer() const {
        return should_add_layer_adaptive(saturation_threshold_);
    }

    bool should_add_layer_adaptive(double threshold) const {
        if (!health_monitor_.allow_layer_addition()) {
            return false;
        }
        if (network_.num_layers() >= max_layers_) {
            return false;
        }

        // Count saturated layers using adaptive threshold
        size_t saturated_count = 0;
        for (size_t i = 0; i < network_.num_layers(); ++i) {
            auto metrics = network_.layer(i).compute_metrics();
            if (metrics.avg_node_efficiency > threshold) {
                saturated_count++;
            }
        }

        // If most layers are saturated, we may need more capacity
        double saturation_ratio = static_cast<double>(saturated_count) / network_.num_layers();
        return saturation_ratio > 0.7;
    }

    bool should_remove_layer() const {
        if (!health_monitor_.allow_layer_removal()) {
            return false;
        }
        if (network_.num_layers() <= min_layers_) {
            return false;
        }

        // Check for redundant layers
        for (size_t i = 1; i < network_.num_layers() - 1; ++i) {
            auto metrics = network_.layer(i).compute_metrics();

            // Very low efficiency - potentially redundant
            if (metrics.avg_node_efficiency < redundancy_threshold_) {
                return true;
            }

            // Check similarity with adjacent layers
            if (i < network_.num_layers() - 1) {
                double similarity = compute_layer_similarity(i, i + 1);
                if (similarity > 0.95) {
                    return true;
                }
            }
        }

        return false;
    }

    size_t find_best_insertion_point() const {
        // Find layer with highest "information bottleneck"
        // (largest relative size reduction)
        double max_bottleneck = 0.0;
        size_t best_pos = network_.num_layers() / 2;

        for (size_t i = 0; i < network_.num_layers() - 1; ++i) {
            size_t size_before = network_.layer(i).output_size();
            size_t size_after = network_.layer(i + 1).output_size();

            if (size_before > 0 && size_after > 0) {
                double ratio = static_cast<double>(size_before) / size_after;
                if (ratio > 1.0) ratio = 1.0 / ratio;
                double bottleneck = 1.0 - ratio;

                if (bottleneck > max_bottleneck) {
                    max_bottleneck = bottleneck;
                    best_pos = i;
                }
            }
        }

        return best_pos;
    }

    size_t find_least_efficient_layer() const {
        double min_efficiency = 1.0;
        size_t least_efficient = 1;  // Skip first layer

        for (size_t i = 1; i < network_.num_layers() - 1; ++i) {  // Skip first and last
            auto metrics = network_.layer(i).compute_metrics();
            if (metrics.avg_node_efficiency < min_efficiency) {
                min_efficiency = metrics.avg_node_efficiency;
                least_efficient = i;
            }
        }

        return least_efficient;
    }

    size_t compute_new_layer_size(size_t insert_after) const {
        // Size between adjacent layers
        size_t before_size = network_.layer(insert_after).output_size();
        size_t after_size = network_.layer(insert_after + 1).input_size();

        // Geometric mean
        size_t new_size = static_cast<size_t>(std::sqrt(
            static_cast<double>(before_size) * after_size));

        return std::max(min_nodes_per_layer_, new_size);
    }

    double compute_layer_similarity(size_t i, size_t j) const {
        // Simplified similarity based on size and efficiency
        auto metrics_i = network_.layer(i).compute_metrics();
        auto metrics_j = network_.layer(j).compute_metrics();

        double size_sim = 1.0 - std::abs(
            static_cast<double>(network_.layer(i).output_size()) /
            static_cast<double>(network_.layer(j).output_size()) - 1.0);
        size_sim = std::max(0.0, size_sim);

        double eff_sim = 1.0 - std::abs(
            metrics_i.avg_node_efficiency - metrics_j.avg_node_efficiency);

        return (size_sim + eff_sim) / 2.0;
    }

    Network<T>& network_;
    HealthMonitor<T>& health_monitor_;
    size_t min_layers_;
    size_t max_layers_;
    size_t min_nodes_per_layer_;
    size_t max_nodes_per_layer_;
    double efficiency_threshold_;
    double saturation_threshold_;
    double redundancy_threshold_;
    bool adaptive_threshold_enabled_;
    double sigmoid_k_;
    double sigmoid_base_;
    double sigmoid_range_;
};

} // namespace dynamics
} // namespace dnn
