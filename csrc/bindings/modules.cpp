#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "enn/modules/adaptive_conv2d.hpp"
#include "enn/modules/reversible_linear.hpp"
#include "enn/modules/reversible_network.hpp"


namespace py = pybind11;
using namespace enn::modules;


void register_modules(py::module_& m) {
    auto sub = m.def_submodule("modules");

    py::class_<ReversibleLinearImpl,
                std::shared_ptr<ReversibleLinearImpl>>(sub,
                                                         "ReversibleLinear")
        .def(py::init<std::int64_t, std::int64_t>(),
              py::arg("in_features"), py::arg("out_features"))
        .def("forward", &ReversibleLinearImpl::forward)
        .def("__call__", &ReversibleLinearImpl::forward)
        .def("add_nodes", &ReversibleLinearImpl::add_nodes)
        .def("prune_nodes",
              [](ReversibleLinearImpl& self,
                  const std::vector<std::int64_t>& idx) {
                  self.prune_nodes(idx);
              })
        .def("compact", &ReversibleLinearImpl::compact)
        .def("update_node_metrics",
              &ReversibleLinearImpl::update_node_metrics)
        .def_property_readonly("weight",
                                 [](ReversibleLinearImpl& s) {
                                     return s.weight;
                                 })
        .def_property_readonly("bias",
                                 [](ReversibleLinearImpl& s) {
                                     return s.bias;
                                 })
        .def_property_readonly("active_mask",
                                 [](ReversibleLinearImpl& s) {
                                     return s.active_mask;
                                 })
        .def("in_features", &ReversibleLinearImpl::in_features)
        .def("out_features", &ReversibleLinearImpl::out_features)
        .def("active_count", &ReversibleLinearImpl::active_count)
        .def("topology_version", &ReversibleLinearImpl::topology_version)
        .def("parameters",
              [](ReversibleLinearImpl& s) { return s.parameters(); })
        .def("buffers",
              [](ReversibleLinearImpl& s) { return s.buffers(); });

    py::enum_<HiddenActivation>(sub, "HiddenActivation")
        .value("ReLU", HiddenActivation::ReLU)
        .value("GELU", HiddenActivation::GELU)
        .value("Tanh", HiddenActivation::Tanh)
        .value("Identity", HiddenActivation::Identity);

    py::class_<ReversibleNetworkConfig>(sub, "ReversibleNetworkConfig")
        .def(py::init<>())
        .def_readwrite("input_features",
                        &ReversibleNetworkConfig::input_features)
        .def_readwrite("output_features",
                        &ReversibleNetworkConfig::output_features)
        .def_readwrite("hidden_sizes",
                        &ReversibleNetworkConfig::hidden_sizes)
        .def_readwrite("hidden_activation",
                        &ReversibleNetworkConfig::hidden_activation);

    py::class_<ReversibleNetworkImpl,
                std::shared_ptr<ReversibleNetworkImpl>>(sub,
                                                         "ReversibleNetwork")
        .def(py::init<ReversibleNetworkConfig>())
        .def("forward", &ReversibleNetworkImpl::forward)
        .def("__call__", &ReversibleNetworkImpl::forward)
        .def("insert_layer", &ReversibleNetworkImpl::insert_layer)
        .def("remove_layer", &ReversibleNetworkImpl::remove_layer)
        .def("compact", &ReversibleNetworkImpl::compact)
        .def("num_layers", &ReversibleNetworkImpl::num_layers)
        .def("num_active_layers", &ReversibleNetworkImpl::num_active_layers)
        .def("num_hidden_nodes", &ReversibleNetworkImpl::num_hidden_nodes)
        .def("topology_version", &ReversibleNetworkImpl::topology_version)
        .def("layer",
              [](ReversibleNetworkImpl& self, std::size_t i)
                  -> std::shared_ptr<ReversibleLinearImpl> {
                  return self.layer_vec_[i].ptr();
              })
        .def("layer_active", &ReversibleNetworkImpl::layer_active)
        .def("parameters",
              [](ReversibleNetworkImpl& s) { return s.parameters(); });

    py::class_<AdaptiveConv2dImpl,
                std::shared_ptr<AdaptiveConv2dImpl>>(sub, "AdaptiveConv2d")
        .def(py::init<std::int64_t, std::int64_t,
                       std::vector<std::int64_t>>(),
              py::arg("in_channels"), py::arg("out_channels"),
              py::arg("candidate_kernel_sizes"))
        .def("forward", &AdaptiveConv2dImpl::forward)
        .def("__call__", &AdaptiveConv2dImpl::forward)
        .def("prune_smallest_candidate",
              &AdaptiveConv2dImpl::prune_smallest_candidate)
        .def("grow_active_kernel",
              &AdaptiveConv2dImpl::grow_active_kernel)
        .def("active_kernel_size",
              &AdaptiveConv2dImpl::active_kernel_size)
        .def("topology_version", &AdaptiveConv2dImpl::topology_version)
        .def_property_readonly("candidate_mask",
                                 &AdaptiveConv2dImpl::candidate_mask)
        .def_property_readonly("architecture_logits",
                                 &AdaptiveConv2dImpl::architecture_logits)
        .def("parameters",
              [](AdaptiveConv2dImpl& s) { return s.parameters(); });
}
