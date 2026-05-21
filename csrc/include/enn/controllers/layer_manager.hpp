#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <string>
#include <vector>


namespace enn::modules {
class ReversibleNetworkImpl;
}


namespace enn::controllers {


class StabilityMonitor;
class PlateauDetector;


struct LayerManagerConfig {
    double utilization_threshold = 0.5;
    double saturation_threshold = 0.7;
    double redundancy_threshold = 0.01;
    double cka_redundancy_threshold = 0.95;
    bool prune_requires_plateau = true;
    std::size_t min_layers = 2;
    std::size_t max_layers = 100;
    std::size_t min_nodes_per_layer = 4;
    std::size_t max_nodes_per_layer = 10000;
    double growth_fraction = 0.25;
    double saturation_layer_ratio = 0.7;
    std::int64_t probe_batch = 16;
    std::int64_t probe_seed = 0xC0FFEE;
};


struct LayerDecision {
    enum class Action {
        None,
        AddLayer,
        RemoveLayer,
        AddNodes,
        RemoveNodes,
        AddAndRemove,
    } action = Action::None;

    std::size_t layer_index = 0;
    std::size_t node_count = 0;
    std::vector<std::int64_t> nodes_to_remove;
    std::string reason;
    double confidence = 0.0;
};


class LayerManager {
public:
    LayerManager(enn::modules::ReversibleNetworkImpl& net,
                 StabilityMonitor& monitor, LayerManagerConfig cfg);

    void tick_utilization(double utilization);
    bool plateau_detected() const;

    LayerDecision analyze();
    LayerDecision analyze_with_utilization(double utilization);
    void execute(const LayerDecision& d);
    LayerDecision auto_adjust();

    static torch::Tensor linear_cka(const torch::Tensor& x,
                                     const torch::Tensor& y);

    const LayerManagerConfig& config() const noexcept { return config_; }

private:
    enn::modules::ReversibleNetworkImpl& network_;
    StabilityMonitor& monitor_;
    LayerManagerConfig config_;
    std::vector<double> utilization_window_;
};


}  // namespace enn::controllers
