#include <torch/extension.h>

#include "enn/controllers/plateau_detector.hpp"
#include "enn/controllers/stability_monitor.hpp"
#include "enn/controllers/adaptive_lr_controller.hpp"
#include "enn/controllers/layer_manager.hpp"
#include "enn/controllers/trainable_scheduler.hpp"
#include "enn/modules/reversible_linear.hpp"
#include "enn/modules/reversible_network.hpp"
#include "enn/modules/adaptive_conv2d.hpp"
#include "enn/training/batch_manager.hpp"
#include "enn/training/early_stopping.hpp"
#include "enn/training/data_driven_config.hpp"
#include "enn/training/phase_controller.hpp"


namespace py = pybind11;

void register_controllers(py::module_& m);
void register_modules(py::module_& m);
void register_training(py::module_& m);


PYBIND11_MODULE(_enn_core, m) {
    m.doc() = "elasticneuralnetwork native extension";
    register_controllers(m);
    register_modules(m);
    register_training(m);
}
