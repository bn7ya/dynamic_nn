#pragma once

#include <torch/torch.h>

#include <cstdint>


namespace enn::modules {


struct NodeMetricsConfig {
    std::int64_t min_samples_for_utilization = 16;
    double ema_alpha = 0.2;
    double variance_normalizer = 0.25;
    double gradient_normalizer = 0.1;
    double contribution_normalizer = 0.1;
    double weight_variance = 0.25;
    double weight_gradient = 0.30;
    double weight_alive = 0.25;
    double weight_contribution = 0.20;
};


class NodeMetrics {
public:
    NodeMetrics(std::int64_t num_nodes, NodeMetricsConfig cfg);

    void resize(std::int64_t num_nodes);
    void reset();

    void observe(const torch::Tensor& pre_act, const torch::Tensor& grad_act);

    torch::Tensor utilization_per_node() const;
    torch::Tensor sample_count() const noexcept { return sample_count_; }

    const NodeMetricsConfig& config() const noexcept { return config_; }

private:
    NodeMetricsConfig config_;
    torch::Tensor activation_mean_;
    torch::Tensor activation_sq_mean_;
    torch::Tensor gradient_abs_mean_;
    torch::Tensor dead_fraction_;
    torch::Tensor contribution_mean_;
    torch::Tensor sample_count_;
};


}  // namespace enn::modules
