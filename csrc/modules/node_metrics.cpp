#include "enn/modules/node_metrics.hpp"


namespace enn::modules {


NodeMetrics::NodeMetrics(std::int64_t num_nodes, NodeMetricsConfig cfg)
    : config_(cfg) {
    resize(num_nodes);
}


void NodeMetrics::resize(std::int64_t num_nodes) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat64);
    activation_mean_ = torch::zeros({num_nodes}, opts);
    activation_sq_mean_ = torch::zeros({num_nodes}, opts);
    gradient_abs_mean_ = torch::zeros({num_nodes}, opts);
    dead_fraction_ = torch::zeros({num_nodes}, opts);
    contribution_mean_ = torch::zeros({num_nodes}, opts);
    sample_count_ = torch::zeros({num_nodes},
                                  torch::TensorOptions().dtype(torch::kInt64));
}


void NodeMetrics::reset() {
    auto n = activation_mean_.numel();
    resize(n);
}


void NodeMetrics::observe(const torch::Tensor& pre_act,
                          const torch::Tensor& grad_act) {
    auto a = pre_act.detach().to(torch::kFloat64);
    auto g = grad_act.detach().to(torch::kFloat64);
    if (a.dim() == 1) a = a.unsqueeze(0);
    if (g.dim() == 1) g = g.unsqueeze(0);
    auto batch = a.size(0);
    auto a_mean = a.mean(0);
    auto a_sq = (a * a).mean(0);
    auto g_abs = g.abs().mean(0);
    auto dead = (a <= 0.0).to(torch::kFloat64).mean(0);
    auto contrib = (a * g).abs().mean(0);

    double alpha = config_.ema_alpha;
    activation_mean_ = (1.0 - alpha) * activation_mean_ + alpha * a_mean;
    activation_sq_mean_ = (1.0 - alpha) * activation_sq_mean_ + alpha * a_sq;
    gradient_abs_mean_ = (1.0 - alpha) * gradient_abs_mean_ + alpha * g_abs;
    dead_fraction_ = (1.0 - alpha) * dead_fraction_ + alpha * dead;
    contribution_mean_ = (1.0 - alpha) * contribution_mean_ + alpha * contrib;
    sample_count_ += batch;
}


torch::Tensor NodeMetrics::utilization_per_node() const {
    auto n = activation_mean_.size(0);
    auto opts = torch::TensorOptions().dtype(torch::kFloat64);
    auto var = (activation_sq_mean_ - activation_mean_ * activation_mean_)
                  .clamp_min(0.0);
    auto var_score = (var / config_.variance_normalizer).clamp(0.0, 1.0);
    auto grad_score = (gradient_abs_mean_ / config_.gradient_normalizer)
                          .clamp(0.0, 1.0);
    auto alive_score = (1.0 - dead_fraction_).clamp(0.0, 1.0);
    auto contrib_score =
        (contribution_mean_ / config_.contribution_normalizer)
            .clamp(0.0, 1.0);
    auto util = config_.weight_variance * var_score +
                config_.weight_gradient * grad_score +
                config_.weight_alive * alive_score +
                config_.weight_contribution * contrib_score;
    auto mask = (sample_count_ >= config_.min_samples_for_utilization)
                    .to(opts);
    auto unknown = 0.5 * torch::ones({n}, opts);
    return mask * util + (1.0 - mask) * unknown;
}


}  // namespace enn::modules
