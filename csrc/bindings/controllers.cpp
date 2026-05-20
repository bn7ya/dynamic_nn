#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "enn/controllers/adaptive_lr_controller.hpp"
#include "enn/controllers/layer_manager.hpp"
#include "enn/controllers/plateau_detector.hpp"
#include "enn/controllers/stability_monitor.hpp"
#include "enn/controllers/trainable_scheduler.hpp"


namespace py = pybind11;
using namespace enn::controllers;


void register_controllers(py::module_& m) {
    auto sub = m.def_submodule("controllers");

    py::class_<PlateauConfig>(sub, "PlateauConfig")
        .def(py::init<>())
        .def_readwrite("window", &PlateauConfig::window)
        .def_readwrite("slope_epsilon", &PlateauConfig::slope_epsilon)
        .def_readwrite("warmup_epochs", &PlateauConfig::warmup_epochs)
        .def_readwrite("cooldown_epochs", &PlateauConfig::cooldown_epochs);

    py::class_<PlateauDetector>(sub, "PlateauDetector")
        .def(py::init<PlateauConfig>())
        .def("tick", &PlateauDetector::tick)
        .def("notify_event_fired", &PlateauDetector::notify_event_fired)
        .def("slope", &PlateauDetector::slope)
        .def("curvature", &PlateauDetector::curvature)
        .def("is_at_local_max", &PlateauDetector::is_at_local_max)
        .def("epochs_seen", &PlateauDetector::epochs_seen)
        .def("window_size", &PlateauDetector::window_size);

    py::enum_<StabilityState>(sub, "StabilityState")
        .value("Stable", StabilityState::Stable)
        .value("ExcessiveGrowthRisk", StabilityState::ExcessiveGrowthRisk)
        .value("PathologicalGrowth", StabilityState::PathologicalGrowth)
        .value("ExcessivePruningRisk",
                StabilityState::ExcessivePruningRisk)
        .value("PathologicalPruning", StabilityState::PathologicalPruning)
        .value("Critical", StabilityState::Critical);

    py::class_<StabilityConfig>(sub, "StabilityConfig")
        .def(py::init<>())
        .def_readwrite("growth_anomaly_threshold",
                        &StabilityConfig::growth_anomaly_threshold)
        .def_readwrite("capacity_loss_threshold",
                        &StabilityConfig::capacity_loss_threshold)
        .def_readwrite("pathological_score_threshold",
                        &StabilityConfig::pathological_score_threshold)
        .def_readwrite("history_window", &StabilityConfig::history_window)
        .def_readwrite("min_layers", &StabilityConfig::min_layers);

    py::class_<StabilityReport>(sub, "StabilityReport")
        .def_readonly("state", &StabilityReport::state)
        .def_readonly("growth_anomaly_score",
                       &StabilityReport::growth_anomaly_score)
        .def_readonly("capacity_loss_score",
                       &StabilityReport::capacity_loss_score)
        .def_readonly("overall_stability",
                       &StabilityReport::overall_stability)
        .def_readonly("diagnosis", &StabilityReport::diagnosis)
        .def_readonly("current_layers", &StabilityReport::current_layers)
        .def_readonly("current_nodes", &StabilityReport::current_nodes)
        .def_readonly("changes_in_window",
                       &StabilityReport::changes_in_window);

    py::class_<StabilityMonitor>(sub, "StabilityMonitor")
        .def(py::init<std::size_t, std::size_t, StabilityConfig>())
        .def("record_change", &StabilityMonitor::record_change)
        .def("update_epoch", &StabilityMonitor::update_epoch)
        .def("update_topology", &StabilityMonitor::update_topology)
        .def("diagnose", &StabilityMonitor::diagnose)
        .def("current_state", &StabilityMonitor::current_state)
        .def("allow_layer_addition",
              &StabilityMonitor::allow_layer_addition)
        .def("allow_layer_removal", &StabilityMonitor::allow_layer_removal)
        .def("growth_anomaly_score",
              &StabilityMonitor::growth_anomaly_score)
        .def("capacity_loss_score",
              &StabilityMonitor::capacity_loss_score)
        .def("reset", &StabilityMonitor::reset);
    sub.def("stability_state_name", &stability_state_name);

    py::class_<AdaptiveLRConfig>(sub, "AdaptiveLRConfig")
        .def(py::init<>())
        .def_readwrite("cost_improvement_threshold",
                        &AdaptiveLRConfig::cost_improvement_threshold)
        .def_readwrite("utilization_improvement_threshold",
                        &AdaptiveLRConfig::utilization_improvement_threshold)
        .def_readwrite("critical_imbalance_threshold",
                        &AdaptiveLRConfig::critical_imbalance_threshold)
        .def_readwrite("moderate_imbalance_threshold",
                        &AdaptiveLRConfig::moderate_imbalance_threshold)
        .def_readwrite("max_increase_factor",
                        &AdaptiveLRConfig::max_increase_factor)
        .def_readwrite("min_decrease_factor",
                        &AdaptiveLRConfig::min_decrease_factor)
        .def_readwrite("min_learning_rate",
                        &AdaptiveLRConfig::min_learning_rate)
        .def_readwrite("max_learning_rate",
                        &AdaptiveLRConfig::max_learning_rate)
        .def_readwrite("baseline_learning_rate",
                        &AdaptiveLRConfig::baseline_learning_rate)
        .def_readwrite("trend_window", &AdaptiveLRConfig::trend_window)
        .def_readwrite("history_window",
                        &AdaptiveLRConfig::history_window);

    py::class_<AdaptiveLRState>(sub, "AdaptiveLRState")
        .def(py::init<>())
        .def_readwrite("improvement_signals",
                        &AdaptiveLRState::improvement_signals)
        .def_readwrite("regression_signals",
                        &AdaptiveLRState::regression_signals)
        .def_readwrite("lr_reset_count", &AdaptiveLRState::lr_reset_count)
        .def("regression_rate", &AdaptiveLRState::regression_rate)
        .def("improvement_rate", &AdaptiveLRState::improvement_rate);

    sub.def("step_adaptive_lr_controller", &step_adaptive_lr_controller);
    sub.def("sigmoid_threshold", &sigmoid_threshold);

    py::class_<LayerManagerConfig>(sub, "LayerManagerConfig")
        .def(py::init<>())
        .def_readwrite("utilization_threshold",
                        &LayerManagerConfig::utilization_threshold)
        .def_readwrite("saturation_threshold",
                        &LayerManagerConfig::saturation_threshold)
        .def_readwrite("redundancy_threshold",
                        &LayerManagerConfig::redundancy_threshold)
        .def_readwrite("prune_requires_plateau",
                        &LayerManagerConfig::prune_requires_plateau)
        .def_readwrite("min_layers", &LayerManagerConfig::min_layers)
        .def_readwrite("max_layers", &LayerManagerConfig::max_layers);

    py::class_<TrainableConfig>(sub, "TrainableConfig")
        .def(py::init<>())
        .def_readwrite("initial_fraction",
                        &TrainableConfig::initial_fraction)
        .def_readwrite("final_fraction",
                        &TrainableConfig::final_fraction)
        .def_readwrite("transition_epochs",
                        &TrainableConfig::transition_epochs);
}
