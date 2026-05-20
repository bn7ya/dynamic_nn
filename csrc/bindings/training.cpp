#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "enn/training/batch_manager.hpp"
#include "enn/training/data_driven_config.hpp"
#include "enn/training/early_stopping.hpp"
#include "enn/training/phase_controller.hpp"


namespace py = pybind11;
using namespace enn::training;


void register_training(py::module_& m) {
    auto sub = m.def_submodule("training");

    py::class_<DatasetStatistics>(sub, "DatasetStatistics")
        .def(py::init<>())
        .def_readwrite("feature_variance_mean",
                        &DatasetStatistics::feature_variance_mean)
        .def_readwrite("feature_variance_spread",
                        &DatasetStatistics::feature_variance_spread)
        .def_readwrite("signal_to_noise_ratio",
                        &DatasetStatistics::signal_to_noise_ratio)
        .def_readwrite("label_entropy", &DatasetStatistics::label_entropy)
        .def_readwrite("effective_rank", &DatasetStatistics::effective_rank)
        .def_readwrite("fisher_separability",
                        &DatasetStatistics::fisher_separability)
        .def_readwrite("num_samples", &DatasetStatistics::num_samples)
        .def_readwrite("num_features", &DatasetStatistics::num_features)
        .def_readwrite("num_classes", &DatasetStatistics::num_classes)
        .def_readwrite("is_regression", &DatasetStatistics::is_regression);

    sub.def("compute_dataset_statistics", &compute_dataset_statistics);
    sub.def("derive_adaptive_lr_config", &derive_adaptive_lr_config);
    sub.def("derive_stability_config", &derive_stability_config);
    sub.def("derive_pruning_config", &derive_pruning_config);
    sub.def("derive_phase_schedule", &derive_phase_schedule);
    sub.def("derive_plateau_config", &derive_plateau_config);

    py::class_<PhaseScheduleConfig>(sub, "PhaseScheduleConfig")
        .def(py::init<>())
        .def_readwrite("topology_discovery_epochs",
                        &PhaseScheduleConfig::topology_discovery_epochs)
        .def_readwrite(
            "convergence_estimation_epochs",
            &PhaseScheduleConfig::convergence_estimation_epochs)
        .def_readwrite("adaptive_training_epochs",
                        &PhaseScheduleConfig::adaptive_training_epochs)
        .def_readwrite("frozen_finetune_epochs",
                        &PhaseScheduleConfig::frozen_finetune_epochs)
        .def_readwrite("topology_discovery_lr",
                        &PhaseScheduleConfig::topology_discovery_lr)
        .def_readwrite(
            "convergence_estimation_lr",
            &PhaseScheduleConfig::convergence_estimation_lr)
        .def_readwrite(
            "adaptive_training_lr_initial",
            &PhaseScheduleConfig::adaptive_training_lr_initial)
        .def_readwrite("adaptive_training_lr_floor",
                        &PhaseScheduleConfig::adaptive_training_lr_floor)
        .def_readwrite("frozen_finetune_lr",
                        &PhaseScheduleConfig::frozen_finetune_lr)
        .def_readwrite("plateau_window",
                        &PhaseScheduleConfig::plateau_window);

    py::class_<PruningConfig>(sub, "PruningConfig")
        .def(py::init<>())
        .def_readwrite("prune_utilization_threshold",
                        &PruningConfig::prune_utilization_threshold)
        .def_readwrite("growth_utilization_threshold",
                        &PruningConfig::growth_utilization_threshold)
        .def_readwrite("cka_redundancy_threshold",
                        &PruningConfig::cka_redundancy_threshold)
        .def_readwrite("min_epochs_before_action",
                        &PruningConfig::min_epochs_before_action)
        .def_readwrite("cooldown_epochs", &PruningConfig::cooldown_epochs);

    py::class_<BatchConfig>(sub, "BatchConfig")
        .def(py::init<>())
        .def_readwrite("min_batch_size", &BatchConfig::min_batch_size)
        .def_readwrite("max_batch_size", &BatchConfig::max_batch_size)
        .def_readwrite("growth_rate", &BatchConfig::growth_rate)
        .def_readwrite("growth_interval_epochs",
                        &BatchConfig::growth_interval_epochs);

    py::class_<BatchManager>(sub, "BatchManager")
        .def(py::init<std::size_t, BatchConfig>())
        .def("update_epoch", &BatchManager::update_epoch)
        .def("current_batch_size", &BatchManager::current_batch_size)
        .def("shuffled_batches", &BatchManager::shuffled_batches);

    py::class_<EarlyStoppingConfig>(sub, "EarlyStoppingConfig")
        .def(py::init<>())
        .def_readwrite("patience", &EarlyStoppingConfig::patience)
        .def_readwrite("min_improvement",
                        &EarlyStoppingConfig::min_improvement)
        .def_readwrite("min_epochs", &EarlyStoppingConfig::min_epochs);

    py::class_<StoppingDecision>(sub, "StoppingDecision")
        .def_readonly("should_stop", &StoppingDecision::should_stop)
        .def_readonly("reason", &StoppingDecision::reason)
        .def_readonly("epochs_seen", &StoppingDecision::epochs_seen);

    py::class_<EarlyStopping>(sub, "EarlyStopping")
        .def(py::init<EarlyStoppingConfig>())
        .def("evaluate", &EarlyStopping::evaluate)
        .def("reset", &EarlyStopping::reset);

    py::enum_<CostFunction>(sub, "CostFunction")
        .value("MSE", CostFunction::MSE)
        .value("CrossEntropy", CostFunction::CrossEntropy);

    py::class_<TrainingResult>(sub, "TrainingResult")
        .def_readonly("epochs_completed",
                       &TrainingResult::epochs_completed)
        .def_readonly("cost_trajectory",
                       &TrainingResult::cost_trajectory)
        .def_readonly("utilization_trajectory",
                       &TrainingResult::utilization_trajectory)
        .def_readonly("stability_state_trajectory",
                       &TrainingResult::stability_state_trajectory)
        .def_readonly("topology_version_final",
                       &TrainingResult::topology_version_final)
        .def_readonly("parameter_count_final",
                       &TrainingResult::parameter_count_final)
        .def_readonly("stopping_reason", &TrainingResult::stopping_reason);

    py::class_<PhaseController>(sub, "PhaseController")
        .def(py::init<enn::modules::ReversibleNetworkImpl&, CostFunction>(),
              py::arg("net"), py::arg("cost") = CostFunction::CrossEntropy)
        .def("fit",
              [](PhaseController& self, const torch::Tensor& X,
                  const torch::Tensor& y) {
                  py::gil_scoped_release release;
                  return self.fit(X, y);
              })
        .def("statistics", &PhaseController::statistics,
              py::return_value_policy::reference_internal);
}
