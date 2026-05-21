#include "enn/controllers/layer_manager.hpp"

#include <algorithm>

#include "enn/controllers/plateau_detector.hpp"
#include "enn/controllers/stability_monitor.hpp"
#include "enn/modules/reversible_linear.hpp"
#include "enn/modules/reversible_network.hpp"


namespace enn::controllers {


LayerManager::LayerManager(enn::modules::ReversibleNetworkImpl& net,
                            StabilityMonitor& monitor,
                            LayerManagerConfig cfg)
    : network_(net), monitor_(monitor), config_(cfg) {}


void LayerManager::tick_utilization(double utilization) {
    utilization_window_.push_back(utilization);
    if (utilization_window_.size() > 64) {
        utilization_window_.erase(utilization_window_.begin());
    }
}


bool LayerManager::plateau_detected() const {
    if (utilization_window_.size() < 8) return false;
    double recent =
        std::accumulate(utilization_window_.end() - 4,
                         utilization_window_.end(), 0.0) / 4.0;
    double older = std::accumulate(utilization_window_.end() - 8,
                                     utilization_window_.end() - 4, 0.0) /
                    4.0;
    return std::abs(recent - older) < 1e-3;
}


LayerDecision LayerManager::analyze() {
    if (utilization_window_.empty()) {
        return {LayerDecision::Action::None, 0, 0, {}, "no_data", 0.0};
    }
    return analyze_with_utilization(utilization_window_.back());
}


LayerDecision LayerManager::analyze_with_utilization(double utilization) {
    LayerDecision d;
    if (utilization >= config_.saturation_threshold) {
        if (network_.num_layers() < config_.max_layers &&
            monitor_.allow_layer_addition()) {
            d.action = LayerDecision::Action::AddLayer;
            d.layer_index = network_.num_layers() / 2;
            d.node_count = std::max<std::size_t>(
                config_.min_nodes_per_layer,
                static_cast<std::size_t>(
                    config_.growth_fraction *
                    static_cast<double>(network_.num_hidden_nodes())));
            d.reason = "saturation";
            d.confidence = utilization;
        }
    } else if (utilization < config_.utilization_threshold) {
        if (network_.num_layers() > config_.min_layers &&
            monitor_.allow_layer_removal() &&
            (!config_.prune_requires_plateau || plateau_detected())) {
            d.action = LayerDecision::Action::RemoveNodes;
            d.layer_index = 0;
            std::size_t target = std::max<std::size_t>(
                config_.min_nodes_per_layer,
                static_cast<std::size_t>(
                    config_.growth_fraction *
                    static_cast<double>(
                        network_.layer(0).active_count())));
            for (std::size_t i = 0; i < target; ++i) {
                d.nodes_to_remove.push_back(static_cast<std::int64_t>(i));
            }
            d.reason = "low_utilization_plateau";
            d.confidence = 1.0 - utilization;
        }
    }
    return d;
}


void LayerManager::execute(const LayerDecision& d) {
    switch (d.action) {
        case LayerDecision::Action::None:
            break;
        case LayerDecision::Action::AddLayer:
            network_.insert_layer(static_cast<std::int64_t>(d.layer_index),
                                   static_cast<std::int64_t>(d.node_count));
            monitor_.record_change(+1,
                                    static_cast<std::int64_t>(d.node_count));
            break;
        case LayerDecision::Action::RemoveLayer:
            network_.remove_layer(static_cast<std::int64_t>(d.layer_index));
            monitor_.record_change(-1, 0);
            break;
        case LayerDecision::Action::AddNodes:
            network_.layer(d.layer_index).add_nodes(
                static_cast<std::int64_t>(d.node_count));
            monitor_.record_change(0,
                                    static_cast<std::int64_t>(d.node_count));
            break;
        case LayerDecision::Action::RemoveNodes:
            network_.layer(d.layer_index).prune_nodes(d.nodes_to_remove);
            monitor_.record_change(0, -static_cast<std::int64_t>(
                                            d.nodes_to_remove.size()));
            break;
        case LayerDecision::Action::AddAndRemove:
            network_.layer(d.layer_index).prune_nodes(d.nodes_to_remove);
            network_.layer(d.layer_index).add_nodes(
                static_cast<std::int64_t>(d.node_count));
            break;
    }
    monitor_.update_topology(network_.num_layers(),
                              network_.num_hidden_nodes());
}


LayerDecision LayerManager::auto_adjust() {
    auto d = analyze();
    execute(d);
    return d;
}


torch::Tensor LayerManager::linear_cka(const torch::Tensor& x,
                                        const torch::Tensor& y) {
    auto xc = x - x.mean(0, /*keepdim=*/true);
    auto yc = y - y.mean(0, /*keepdim=*/true);
    auto xx = (xc.transpose(0, 1).matmul(xc)).pow(2).sum();
    auto yy = (yc.transpose(0, 1).matmul(yc)).pow(2).sum();
    auto xy = (xc.transpose(0, 1).matmul(yc)).pow(2).sum();
    auto denom = (xx * yy).sqrt().clamp_min(1e-12);
    return xy / denom;
}


}  // namespace enn::controllers
