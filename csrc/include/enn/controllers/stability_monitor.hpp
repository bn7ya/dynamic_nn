#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>


namespace enn::controllers {


enum class StabilityState {
    Stable,
    ExcessiveGrowthRisk,
    PathologicalGrowth,
    ExcessivePruningRisk,
    PathologicalPruning,
    Critical,
};


struct StabilityConfig {
    double growth_anomaly_threshold = 0.7;
    double capacity_loss_threshold = 0.7;
    double pathological_score_threshold = 0.9;
    std::size_t history_window = 50;
    std::size_t min_layers = 2;
    double weight_rate = 0.30;
    double weight_consecutive = 0.25;
    double weight_size = 0.25;
    double weight_remainder = 0.20;
    double growth_size_normalizer = 5.0;
    double growth_node_normalizer = 10.0;
    double prune_size_normalizer = 3.0;
};


struct StabilityReport {
    StabilityState state;
    double growth_anomaly_score;
    double capacity_loss_score;
    double overall_stability;
    std::string diagnosis;
    std::size_t current_layers;
    std::size_t current_nodes;
    std::size_t changes_in_window;
};


class StabilityMonitor {
public:
    StabilityMonitor(std::size_t initial_layers, std::size_t initial_nodes,
                     StabilityConfig cfg);

    void record_change(std::int64_t layer_delta, std::int64_t node_delta);
    void update_epoch(std::size_t epoch);
    void update_topology(std::size_t layers, std::size_t nodes);

    StabilityReport diagnose() const;
    StabilityState current_state() const;
    bool allow_layer_addition() const;
    bool allow_layer_removal() const;
    double growth_anomaly_score() const;
    double capacity_loss_score() const;

    const StabilityConfig& config() const noexcept { return config_; }
    void reset();

private:
    struct ChangeEvent {
        std::size_t epoch;
        std::int64_t layer_delta;
        std::int64_t node_delta;
    };

    StabilityConfig config_;
    std::deque<ChangeEvent> events_;
    std::size_t current_epoch_ = 0;
    std::size_t initial_layers_;
    std::size_t initial_nodes_;
    std::size_t current_layers_;
    std::size_t current_nodes_;
};


std::string stability_state_name(StabilityState s);


}  // namespace enn::controllers
