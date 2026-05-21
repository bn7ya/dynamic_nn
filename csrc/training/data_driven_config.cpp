#include "enn/training/data_driven_config.hpp"

#include <algorithm>
#include <cmath>


namespace enn::training {


namespace {

double sigmoid(double z, double k = 5.0) {
    return 1.0 / (1.0 + std::exp(-k * z));
}


double normalized_label_entropy(const torch::Tensor& y) {
    auto labels = y;
    if (labels.dim() > 1 && labels.size(1) > 1) {
        labels = labels.argmax(1);
    }
    labels = labels.to(torch::kInt64).contiguous();
    auto max_label = labels.max().item<std::int64_t>();
    auto bins = max_label + 1;
    auto counts = torch::zeros({bins}, torch::kFloat64);
    counts.scatter_add_(0, labels,
                         torch::ones_like(labels, torch::kFloat64));
    auto probs = counts / counts.sum().clamp_min(1.0);
    auto safe = probs.clamp_min(1e-12);
    double h = (-(probs * safe.log()).sum()).item<double>();
    double h_max = std::log(static_cast<double>(bins));
    return h_max > 0.0 ? h / h_max : 0.0;
}


double effective_rank(const torch::Tensor& X) {
    auto n = X.size(0);
    auto cap = std::min<std::int64_t>(n, 1024);
    auto sub = X.index({torch::indexing::Slice(0, cap)}).to(torch::kFloat64);
    auto centered = sub - sub.mean(0, /*keepdim=*/true);
    auto svd_result = at::linalg_svd(centered, /*full_matrices=*/false);
    auto sigmas = std::get<1>(svd_result).clamp_min(1e-12);
    auto p = sigmas / sigmas.sum();
    auto entropy = -(p * p.log()).sum().item<double>();
    return std::exp(entropy);
}


double fisher_separability(const torch::Tensor& X, const torch::Tensor& y) {
    auto labels = y;
    if (labels.dim() > 1 && labels.size(1) > 1) labels = labels.argmax(1);
    labels = labels.to(torch::kInt64).contiguous();
    auto x = X.to(torch::kFloat64);
    auto global_mean = x.mean(0);
    auto unique = std::get<0>(torch::_unique(labels));
    double between = 0.0;
    double within = 0.0;
    for (std::int64_t i = 0; i < unique.size(0); ++i) {
        auto cls = unique[i].item<std::int64_t>();
        auto mask = (labels == cls);
        auto sub = x.index({mask});
        if (sub.size(0) <= 1) continue;
        auto mu = sub.mean(0);
        auto diff = mu - global_mean;
        between += diff.pow(2).sum().item<double>() *
                    static_cast<double>(sub.size(0));
        within += (sub - mu).pow(2).sum().item<double>();
    }
    if (within <= 0.0) return 0.0;
    double sep = between / (between + within);
    return std::clamp(sep, 0.0, 1.0);
}

}  // namespace


DatasetStatistics compute_dataset_statistics(const torch::Tensor& X,
                                              const torch::Tensor& y) {
    DatasetStatistics s;
    auto x = X.to(torch::kFloat64);
    auto flat = x.dim() > 2
                    ? x.reshape({x.size(0), -1})
                    : x;
    auto var = flat.var(0, /*unbiased=*/true);
    s.feature_variance_mean = var.mean().item<double>();
    s.feature_variance_spread = var.std().item<double>();
    auto mu = flat.mean(0);
    auto signal = mu.pow(2).mean().item<double>();
    s.signal_to_noise_ratio =
        signal / std::max(s.feature_variance_mean, 1e-12);
    s.num_samples = flat.size(0);
    s.num_features = flat.size(1);
    if (y.dim() == 1 || (y.dim() == 2 && y.size(1) == 1)) {
        auto vals = y.to(torch::kFloat64).flatten();
        auto distinct = std::get<0>(torch::_unique(vals));
        if (distinct.size(0) > 20) {
            s.is_regression = true;
            s.num_classes = 0;
            s.label_entropy = 1.0;
            s.fisher_separability = -1.0;
        } else {
            s.num_classes = distinct.size(0);
            s.label_entropy = normalized_label_entropy(y);
            s.fisher_separability = fisher_separability(flat, y);
        }
    } else {
        s.num_classes = y.size(1);
        s.label_entropy = normalized_label_entropy(y);
        s.fisher_separability = fisher_separability(flat, y);
    }
    s.effective_rank = effective_rank(flat);
    return s;
}


enn::controllers::AdaptiveLRConfig derive_adaptive_lr_config(
    const DatasetStatistics& s) {
    enn::controllers::AdaptiveLRConfig cfg;
    double snr = s.signal_to_noise_ratio;
    cfg.cost_improvement_threshold = 0.5 / std::max(snr + 1.0, 1.0) * 0.01;
    cfg.utilization_improvement_threshold =
        0.02 * (1.0 + sigmoid(s.label_entropy - 0.5));
    cfg.baseline_learning_rate =
        0.1 * sigmoid(1.0 - s.feature_variance_mean, 3.0);
    cfg.max_learning_rate = 5.0 * cfg.baseline_learning_rate;
    cfg.min_learning_rate = 1e-4 * cfg.baseline_learning_rate;
    cfg.trend_window = static_cast<std::size_t>(
        std::max(5.0, std::ceil(std::sqrt(
                         static_cast<double>(std::max<std::int64_t>(
                             100, s.num_samples / 1000))))));
    cfg.critical_imbalance_threshold = 0.5 + 0.25 * s.label_entropy;
    cfg.moderate_imbalance_threshold =
        cfg.critical_imbalance_threshold - 0.1;
    return cfg;
}


enn::controllers::StabilityConfig derive_stability_config(
    const DatasetStatistics& s) {
    enn::controllers::StabilityConfig cfg;
    double complexity =
        0.5 * s.label_entropy + 0.5 * std::min(1.0, s.effective_rank /
                                                       static_cast<double>(
                                                           std::max(s.num_features,
                                                                     std::int64_t{1})));
    cfg.growth_anomaly_threshold = 0.5 + 0.3 * (1.0 - complexity);
    cfg.capacity_loss_threshold = 0.5 + 0.3 * complexity;
    cfg.pathological_score_threshold =
        std::min(0.95, cfg.growth_anomaly_threshold + 0.2);
    cfg.history_window = static_cast<std::size_t>(
        std::max(20.0, std::ceil(std::sqrt(
                          static_cast<double>(std::max<std::int64_t>(
                              100, s.num_samples / 1000)))) *
                          5.0));
    return cfg;
}


PruningConfig derive_pruning_config(const DatasetStatistics& s) {
    PruningConfig cfg;
    cfg.prune_utilization_threshold =
        0.3 + 0.2 * (1.0 - s.label_entropy);
    cfg.growth_utilization_threshold =
        0.6 + 0.2 * s.label_entropy;
    cfg.cka_redundancy_threshold = 0.9 + 0.05 * s.label_entropy;
    cfg.min_epochs_before_action = static_cast<std::int64_t>(
        std::ceil(std::sqrt(static_cast<double>(
            std::max<std::int64_t>(64, s.num_samples / 100)))));
    cfg.cooldown_epochs = std::max<std::int64_t>(
        3, cfg.min_epochs_before_action / 2);
    return cfg;
}


PhaseScheduleConfig derive_phase_schedule(const DatasetStatistics& s) {
    PhaseScheduleConfig cfg;
    double n = static_cast<double>(std::max<std::int64_t>(100, s.num_samples));
    double base = std::ceil(std::sqrt(n));
    cfg.topology_discovery_epochs =
        static_cast<std::int64_t>(std::max(5.0, base * 0.25));
    cfg.convergence_estimation_epochs =
        static_cast<std::int64_t>(std::max(5.0, base * 0.25));
    cfg.adaptive_training_epochs =
        static_cast<std::int64_t>(std::max(20.0, base * 1.5));
    cfg.frozen_finetune_epochs =
        static_cast<std::int64_t>(std::max(10.0, base * 0.5));
    cfg.topology_discovery_lr = 0.1 *
        (1.0 + sigmoid(s.feature_variance_mean - 1.0, 3.0));
    cfg.convergence_estimation_lr =
        cfg.topology_discovery_lr * 0.5;
    cfg.adaptive_training_lr_initial =
        cfg.convergence_estimation_lr;
    cfg.adaptive_training_lr_floor =
        cfg.adaptive_training_lr_initial * 0.01;
    cfg.frozen_finetune_lr = cfg.adaptive_training_lr_floor;
    cfg.plateau_window = static_cast<std::int64_t>(
        std::max(5.0, base * 0.1));
    return cfg;
}


enn::controllers::PlateauConfig derive_plateau_config(
    const DatasetStatistics& s) {
    enn::controllers::PlateauConfig cfg;
    double base = std::sqrt(static_cast<double>(std::max<std::int64_t>(
                                100, s.num_samples / 1000)));
    cfg.window = static_cast<std::size_t>(std::max(5.0, std::ceil(base)));
    cfg.slope_epsilon =
        1e-4 * (1.0 + s.feature_variance_spread);
    cfg.warmup_epochs = cfg.window;
    cfg.cooldown_epochs = std::max<std::size_t>(3, cfg.window / 2);
    return cfg;
}


}  // namespace enn::training
