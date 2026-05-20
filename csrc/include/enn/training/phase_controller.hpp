#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "enn/controllers/adaptive_lr_controller.hpp"
#include "enn/controllers/layer_manager.hpp"
#include "enn/controllers/plateau_detector.hpp"
#include "enn/controllers/stability_monitor.hpp"
#include "enn/modules/reversible_network.hpp"
#include "enn/training/data_driven_config.hpp"
#include "enn/training/early_stopping.hpp"


namespace enn::training {


struct TrainingResult {
    std::int64_t epochs_completed = 0;
    std::vector<double> cost_trajectory;
    std::vector<double> utilization_trajectory;
    std::vector<std::string> stability_state_trajectory;
    std::int64_t topology_version_final = 0;
    std::int64_t parameter_count_final = 0;
    std::string stopping_reason;
};


struct PhaseRunResult {
    std::int64_t epochs_run = 0;
    double final_cost = 0.0;
    double final_utilization = 0.0;
};


enum class CostFunction { MSE, CrossEntropy };


class PhaseController {
public:
    PhaseController(enn::modules::ReversibleNetworkImpl& net,
                     CostFunction cost = CostFunction::CrossEntropy);

    TrainingResult fit(const torch::Tensor& X, const torch::Tensor& y);

    PhaseRunResult run_topology_discovery(const torch::Tensor& X,
                                            const torch::Tensor& y);
    PhaseRunResult run_convergence_rate_estimation(const torch::Tensor& X,
                                                     const torch::Tensor& y);
    PhaseRunResult run_adaptive_training(const torch::Tensor& X,
                                          const torch::Tensor& y);
    PhaseRunResult run_frozen_architecture_finetuning(
        const torch::Tensor& X, const torch::Tensor& y);

    const DatasetStatistics& statistics() const { return stats_; }
    const TrainingResult& result() const { return result_; }

    torch::Tensor compute_loss(torch::Tensor pred, torch::Tensor target);
    double compute_global_utilization();
    void log_epoch(double cost);
    void freeze_topology();

private:

    enn::modules::ReversibleNetworkImpl& net_;
    CostFunction cost_function_;
    DatasetStatistics stats_;
    PhaseScheduleConfig schedule_;
    PruningConfig pruning_;
    enn::controllers::AdaptiveLRConfig lr_cfg_;
    enn::controllers::StabilityConfig stability_cfg_;
    enn::controllers::PlateauConfig plateau_cfg_;
    std::unique_ptr<enn::controllers::StabilityMonitor> stability_;
    std::unique_ptr<enn::controllers::LayerManager> layer_manager_;
    std::unique_ptr<enn::controllers::PlateauDetector> plateau_;
    enn::controllers::AdaptiveLRState lr_state_;
    TrainingResult result_;
    double current_lr_ = 0.1;
    bool initialized_ = false;
};


}  // namespace enn::training
