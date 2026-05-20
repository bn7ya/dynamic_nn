#include "enn/controllers/stability_monitor.hpp"

#include <algorithm>


namespace enn::controllers {


StabilityMonitor::StabilityMonitor(std::size_t initial_layers,
                                   std::size_t initial_nodes,
                                   StabilityConfig cfg)
    : config_(cfg),
      initial_layers_(initial_layers),
      initial_nodes_(initial_nodes),
      current_layers_(initial_layers),
      current_nodes_(initial_nodes) {}


void StabilityMonitor::record_change(std::int64_t layer_delta,
                                     std::int64_t node_delta) {
    events_.push_back({current_epoch_, layer_delta, node_delta});
    while (events_.size() > config_.history_window) {
        events_.pop_front();
    }
}


void StabilityMonitor::update_epoch(std::size_t epoch) {
    current_epoch_ = epoch;
}


void StabilityMonitor::update_topology(std::size_t layers,
                                       std::size_t nodes) {
    current_layers_ = layers;
    current_nodes_ = nodes;
}


namespace {

struct EventRef {
    std::int64_t layer_delta;
    std::int64_t node_delta;
};

template <typename Container>
double sign_consecutive_factor(const Container& events, int direction) {
    if (events.empty()) return 0.0;
    std::size_t streak = 0;
    std::size_t best = 0;
    for (const auto& e : events) {
        bool matches = (direction > 0)
            ? (e.layer_delta > 0 || e.node_delta > 0)
            : (e.layer_delta < 0 || e.node_delta < 0);
        if (matches) {
            ++streak;
            best = std::max(best, streak);
        } else {
            streak = 0;
        }
    }
    return std::min(1.0, static_cast<double>(best) /
                          static_cast<double>(events.size()));
}

template <typename Container>
double change_rate(const Container& events, int direction,
                   std::size_t window) {
    if (events.empty() || window == 0) return 0.0;
    std::size_t count = 0;
    for (const auto& e : events) {
        bool matches = (direction > 0)
            ? (e.layer_delta > 0 || e.node_delta > 0)
            : (e.layer_delta < 0 || e.node_delta < 0);
        if (matches) ++count;
    }
    return std::min(1.0, static_cast<double>(count) /
                          static_cast<double>(window));
}

}  // namespace


double StabilityMonitor::growth_anomaly_score() const {
    if (events_.empty()) return 0.0;
    double rate = change_rate(events_, +1, config_.history_window);
    double cons = sign_consecutive_factor(events_, +1);
    double size_ratio = initial_nodes_ > 0
        ? static_cast<double>(current_nodes_) /
              static_cast<double>(initial_nodes_)
        : 1.0;
    double size_factor = std::min(
        1.0, std::max(0.0, (size_ratio - 1.0) /
                                config_.growth_size_normalizer));
    double layer_ratio = initial_layers_ > 0
        ? static_cast<double>(current_layers_) /
              static_cast<double>(initial_layers_)
        : 1.0;
    double layer_factor = std::min(
        1.0, std::max(0.0, (layer_ratio - 1.0) /
                                config_.growth_node_normalizer));
    return config_.weight_rate * rate +
           config_.weight_consecutive * cons +
           config_.weight_size * size_factor +
           config_.weight_remainder * layer_factor;
}


double StabilityMonitor::capacity_loss_score() const {
    if (events_.empty()) return 0.0;
    double rate = change_rate(events_, -1, config_.history_window);
    double cons = sign_consecutive_factor(events_, -1);
    double size_ratio = current_nodes_ > 0 && initial_nodes_ > 0
        ? static_cast<double>(initial_nodes_) /
              static_cast<double>(current_nodes_)
        : 1.0;
    double size_factor = std::min(
        1.0, std::max(0.0, (size_ratio - 1.0) /
                                config_.prune_size_normalizer));
    double min_factor = current_nodes_ <= initial_nodes_ / 4 ? 1.0 : 0.0;
    return config_.weight_rate * rate +
           config_.weight_consecutive * cons +
           config_.weight_size * size_factor +
           config_.weight_remainder * min_factor;
}


StabilityState StabilityMonitor::current_state() const {
    if (current_layers_ < config_.min_layers) return StabilityState::Critical;
    double g = growth_anomaly_score();
    double c = capacity_loss_score();
    if (g >= config_.pathological_score_threshold)
        return StabilityState::PathologicalGrowth;
    if (c >= config_.pathological_score_threshold)
        return StabilityState::PathologicalPruning;
    if (g >= config_.growth_anomaly_threshold)
        return StabilityState::ExcessiveGrowthRisk;
    if (c >= config_.capacity_loss_threshold)
        return StabilityState::ExcessivePruningRisk;
    return StabilityState::Stable;
}


bool StabilityMonitor::allow_layer_addition() const {
    auto s = current_state();
    return s != StabilityState::PathologicalGrowth &&
           s != StabilityState::ExcessiveGrowthRisk;
}


bool StabilityMonitor::allow_layer_removal() const {
    if (current_layers_ <= config_.min_layers) return false;
    auto s = current_state();
    return s != StabilityState::PathologicalPruning &&
           s != StabilityState::ExcessivePruningRisk;
}


StabilityReport StabilityMonitor::diagnose() const {
    double g = growth_anomaly_score();
    double c = capacity_loss_score();
    StabilityState s = current_state();
    return StabilityReport{
        s, g, c, 1.0 - std::max(g, c), stability_state_name(s),
        current_layers_, current_nodes_, events_.size()};
}


void StabilityMonitor::reset() {
    events_.clear();
    current_epoch_ = 0;
    current_layers_ = initial_layers_;
    current_nodes_ = initial_nodes_;
}


std::string stability_state_name(StabilityState s) {
    switch (s) {
        case StabilityState::Stable: return "Stable";
        case StabilityState::ExcessiveGrowthRisk: return "ExcessiveGrowthRisk";
        case StabilityState::PathologicalGrowth: return "PathologicalGrowth";
        case StabilityState::ExcessivePruningRisk:
            return "ExcessivePruningRisk";
        case StabilityState::PathologicalPruning: return "PathologicalPruning";
        case StabilityState::Critical: return "Critical";
    }
    return "Unknown";
}


}  // namespace enn::controllers
