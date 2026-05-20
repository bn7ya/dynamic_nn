#pragma once

#include <torch/torch.h>

#include "enn/controllers/adaptive_lr_controller.hpp"
#include "enn/controllers/layer_manager.hpp"
#include "enn/controllers/stability_monitor.hpp"
#include "enn/controllers/plateau_detector.hpp"


namespace enn::training {


struct DatasetStatistics {
    double feature_variance_mean = 0.0;
    double feature_variance_spread = 0.0;
    double signal_to_noise_ratio = 0.0;
    double label_entropy = 0.0;
    double effective_rank = 0.0;
    double fisher_separability = -1.0;
    std::int64_t num_samples = 0;
    std::int64_t num_features = 0;
    std::int64_t num_classes = 0;
    bool is_regression = false;
};


struct PhaseScheduleConfig {
    std::int64_t topology_discovery_epochs;
    std::int64_t convergence_estimation_epochs;
    std::int64_t adaptive_training_epochs;
    std::int64_t frozen_finetune_epochs;
    double topology_discovery_lr;
    double convergence_estimation_lr;
    double adaptive_training_lr_initial;
    double adaptive_training_lr_floor;
    double frozen_finetune_lr;
    std::int64_t plateau_window;
};


struct PruningConfig {
    double prune_utilization_threshold;
    double growth_utilization_threshold;
    double cka_redundancy_threshold;
    std::int64_t min_epochs_before_action;
    std::int64_t cooldown_epochs;
};


DatasetStatistics compute_dataset_statistics(const torch::Tensor& X,
                                              const torch::Tensor& y);

enn::controllers::AdaptiveLRConfig derive_adaptive_lr_config(
    const DatasetStatistics& s);

enn::controllers::StabilityConfig derive_stability_config(
    const DatasetStatistics& s);

PruningConfig derive_pruning_config(const DatasetStatistics& s);

PhaseScheduleConfig derive_phase_schedule(const DatasetStatistics& s);

enn::controllers::PlateauConfig derive_plateau_config(
    const DatasetStatistics& s);


}  // namespace enn::training
