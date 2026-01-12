/**
 * Python bindings for Dynamic Neural Network library.
 * Uses pybind11 for zero-copy NumPy integration.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include "dnn/core/tensor.hpp"
#include "dnn/core/network.hpp"
#include "dnn/core/activations.hpp"
#include "dnn/core/device.hpp"
#include "dnn/training/trainer.hpp"
#include "dnn/training/cost_functions.hpp"
#include "dnn/dynamics/health_monitor.hpp"
#include "dnn/io/model_serializer.hpp"

namespace py = pybind11;
using namespace dnn;

// Helper to convert NumPy array to Tensor
template<typename T>
core::Tensor<T> numpy_to_tensor(py::array_t<T> arr) {
    py::buffer_info buf = arr.request();

    std::vector<size_t> shape;
    for (auto dim : buf.shape) {
        shape.push_back(static_cast<size_t>(dim));
    }

    core::Tensor<T> tensor(shape);

    // Copy data
    T* src = static_cast<T*>(buf.ptr);
    T* dst = tensor.data();
    std::copy(src, src + tensor.size(), dst);

    return tensor;
}

// Helper to convert Tensor to NumPy array
template<typename T>
py::array_t<T> tensor_to_numpy(const core::Tensor<T>& tensor) {
    std::vector<py::ssize_t> shape;
    for (auto dim : tensor.shape()) {
        shape.push_back(static_cast<py::ssize_t>(dim));
    }

    py::array_t<T> arr(shape);
    py::buffer_info buf = arr.request();

    T* dst = static_cast<T*>(buf.ptr);
    const T* src = tensor.data();
    std::copy(src, src + tensor.size(), dst);

    return arr;
}

// Wrapper class for Python-friendly Network interface
template<typename T>
class PyNetwork {
public:
    PyNetwork(const core::NetworkConfig& config)
        : network_(config) {}

    py::array_t<T> forward(py::array_t<T> input) {
        auto tensor = numpy_to_tensor(input);
        auto output = network_.forward(tensor);
        return tensor_to_numpy(output);
    }

    py::array_t<T> predict(py::array_t<T> input) {
        return forward(input);
    }

    size_t num_layers() const { return network_.num_layers(); }
    size_t num_parameters() const { return network_.num_parameters(); }

    void to(core::Device device) { network_.to(device); }
    core::Device device() const { return network_.device(); }

    py::dict efficiency_report(bool include_nodes = false) {
        py::dict result;

        double total_efficiency = 0.0;
        size_t total_layers = network_.num_layers();
        size_t total_nodes = 0;
        size_t total_params = network_.num_parameters();

        py::list layer_efficiencies;
        for (size_t i = 0; i < total_layers; ++i) {
            const auto& layer = network_.layer(i);
            auto metrics = layer.compute_metrics();
            double eff = metrics.avg_node_efficiency;
            layer_efficiencies.append(eff);
            total_efficiency += eff;
            total_nodes += layer.output_size();
        }

        result["avg_layer_efficiency"] = total_layers > 0 ? total_efficiency / total_layers : 0.0;
        result["total_layers"] = total_layers;
        result["total_nodes"] = total_nodes;
        result["total_parameters"] = total_params;
        result["layer_efficiencies"] = layer_efficiencies;

        return result;
    }

    core::Network<T>& network() { return network_; }
    const core::Network<T>& network() const { return network_; }

private:
    core::Network<T> network_;
};

// Wrapper for Trainer
template<typename T>
class PyTrainer {
public:
    PyTrainer(PyNetwork<T>& network,
              training::CostFunctionType cost_type = training::CostFunctionType::MeanSquaredError,
              const training::TrainerConfig& config = training::TrainerConfig())
        : trainer_(network.network(), cost_type, config) {}

    training::TrainingResult train(py::array_t<T> inputs, py::array_t<T> targets) {
        py::buffer_info input_buf = inputs.request();
        py::buffer_info target_buf = targets.request();

        if (input_buf.ndim < 2 || target_buf.ndim < 2) {
            throw std::runtime_error("Inputs and targets must have at least 2 dimensions");
        }

        size_t n_samples = input_buf.shape[0];
        size_t input_size = 1;
        for (size_t i = 1; i < input_buf.ndim; ++i) {
            input_size *= input_buf.shape[i];
        }
        size_t target_size = target_buf.shape[1];

        std::vector<core::Tensor<T>> input_tensors;
        std::vector<core::Tensor<T>> target_tensors;

        T* input_ptr = static_cast<T*>(input_buf.ptr);
        T* target_ptr = static_cast<T*>(target_buf.ptr);

        for (size_t i = 0; i < n_samples; ++i) {
            core::Tensor<T> inp(std::vector<size_t>{input_size});
            std::copy(input_ptr + i * input_size,
                     input_ptr + (i + 1) * input_size,
                     inp.data());
            input_tensors.push_back(std::move(inp));

            core::Tensor<T> tgt(std::vector<size_t>{target_size});
            std::copy(target_ptr + i * target_size,
                     target_ptr + (i + 1) * target_size,
                     tgt.data());
            target_tensors.push_back(std::move(tgt));
        }

        // Release GIL during training
        py::gil_scoped_release release;
        return trainer_.train(input_tensors, target_tensors);
    }

    void set_epoch_callback(std::function<void(uint64_t, double, double)> callback) {
        trainer_.set_epoch_callback(
            [callback](uint64_t epoch, double cost, double efficiency,
                      const core::Network<T>&) {
                py::gil_scoped_acquire acquire;
                callback(epoch, cost, efficiency);
            }
        );
    }

    double learning_rate() const { return trainer_.learning_rate(); }
    size_t batch_size() const { return trainer_.batch_size(); }

    dynamics::HealthReport health_report() const {
        return trainer_.health_report();
    }

    py::array_t<T> normalize_input(py::array_t<T> input) {
        auto tensor = numpy_to_tensor(input);
        auto normalized = trainer_.normalize_input(tensor);
        return tensor_to_numpy(normalized);
    }

    py::array_t<T> denormalize_output(py::array_t<T> output) {
        auto tensor = numpy_to_tensor(output);
        auto denormalized = trainer_.denormalize_output(tensor);
        return tensor_to_numpy(denormalized);
    }

    py::dict get_normalization_params() const {
        py::dict result;
        const auto& params = trainer_.normalization_params();

        result["input_normalized"] = params.input_normalized;
        result["output_normalized"] = params.output_normalized;
        result["method"] = params.method == training::NormalizationMethod::ZScore ? "zscore" : "minmax";

        if (params.input_normalized) {
            result["input_mean"] = py::array_t<T>(params.input_mean.size(), params.input_mean.data());
            result["input_std"] = py::array_t<T>(params.input_std.size(), params.input_std.data());
            result["input_min"] = py::array_t<T>(params.input_min.size(), params.input_min.data());
            result["input_max"] = py::array_t<T>(params.input_max.size(), params.input_max.data());
        }

        if (params.output_normalized) {
            result["output_mean"] = py::array_t<T>(params.output_mean.size(), params.output_mean.data());
            result["output_std"] = py::array_t<T>(params.output_std.size(), params.output_std.data());
            result["output_min"] = py::array_t<T>(params.output_min.size(), params.output_min.data());
            result["output_max"] = py::array_t<T>(params.output_max.size(), params.output_max.data());
        }

        return result;
    }

private:
    training::Trainer<T> trainer_;
};

// Model saving helper
template<typename T>
void save_model(const PyNetwork<T>& network, const std::string& path, const std::string& name = "model") {
    io::ModelSerializer<T>::save(network.network(), path, name);
}

template<typename T>
std::unique_ptr<core::Network<T>> load_model(const std::string& path) {
    return io::ModelSerializer<T>::load(path);
}

PYBIND11_MODULE(_dnn_core, m) {
    m.doc() = "Dynamic Neural Network C++ Core";

    // Normalization method enum
    py::enum_<training::NormalizationMethod>(m, "NormalizationMethod")
        .value("ZScore", training::NormalizationMethod::ZScore)
        .value("MinMax", training::NormalizationMethod::MinMax)
        .export_values();

    // Cost function enum
    py::enum_<training::CostFunctionType>(m, "CostFunction")
        .value("MSE", training::CostFunctionType::MeanSquaredError)
        .value("MAE", training::CostFunctionType::MeanAbsoluteError)
        .value("CrossEntropy", training::CostFunctionType::CrossEntropy)
        .value("BinaryCrossEntropy", training::CostFunctionType::BinaryCrossEntropy)
        .value("Huber", training::CostFunctionType::HuberLoss)
        .value("LogCosh", training::CostFunctionType::LogCosh)
        .value("KLDivergence", training::CostFunctionType::KLDivergence)
        .value("CosineSimilarity", training::CostFunctionType::CosineSimilarity)
        .export_values();

    // Activation enum
    py::enum_<core::ActivationType>(m, "Activation")
        .value("Linear", core::ActivationType::Linear)
        .value("ReLU", core::ActivationType::ReLU)
        .value("LeakyReLU", core::ActivationType::LeakyReLU)
        .value("ELU", core::ActivationType::ELU)
        .value("SELU", core::ActivationType::SELU)
        .value("Sigmoid", core::ActivationType::Sigmoid)
        .value("Tanh", core::ActivationType::Tanh)
        .value("Softmax", core::ActivationType::Softmax)
        .value("Swish", core::ActivationType::Swish)
        .value("GELU", core::ActivationType::GELU)
        .export_values();

    // Health state enum
    py::enum_<dynamics::HealthState>(m, "HealthState")
        .value("Healthy", dynamics::HealthState::Healthy)
        .value("CancerRisk", dynamics::HealthState::CancerRisk)
        .value("Cancer", dynamics::HealthState::Cancer)
        .value("AlzheimerRisk", dynamics::HealthState::AlzheimerRisk)
        .value("Alzheimer", dynamics::HealthState::Alzheimer)
        .value("Critical", dynamics::HealthState::Critical)
        .export_values();

    // Device enum
    py::enum_<core::Device>(m, "Device")
        .value("CPU", core::Device::CPU)
        .value("CUDA", core::Device::CUDA)
        .export_values();

    // CUDA availability check
    m.def("cuda_available", []() {
#ifdef DNN_ENABLE_CUDA
        return true;  // Runtime check would go here
#else
        return false;
#endif
    }, "Check if CUDA is available");

    // Tensor class (float)
    py::class_<core::Tensor<float>>(m, "Tensor")
        .def(py::init<const std::vector<size_t>&>())
        .def(py::init([](py::array_t<float> arr) {
            return numpy_to_tensor(arr);
        }))
        .def("numpy", [](const core::Tensor<float>& t) {
            return tensor_to_numpy(t);
        })
        .def("shape", [](const core::Tensor<float>& t) { return t.shape(); })
        .def("size", [](const core::Tensor<float>& t) { return t.size(); })
        .def("rank", [](const core::Tensor<float>& t) { return t.rank(); })
        .def("fill", [](core::Tensor<float>& t, float v) { t.fill(v); })
        .def_static("zeros", [](const std::vector<size_t>& shape) {
            return core::Tensor<float>::zeros(shape);
        })
        .def_static("ones", [](const std::vector<size_t>& shape) {
            return core::Tensor<float>::ones(shape);
        })
        .def("sum", [](const core::Tensor<float>& t) { return t.sum(); })
        .def("mean", [](const core::Tensor<float>& t) { return t.mean(); })
        .def("max", [](const core::Tensor<float>& t) { return t.max(); })
        .def("min", [](const core::Tensor<float>& t) { return t.min(); });

    // Network config
    py::class_<core::NetworkConfig>(m, "NetworkConfig")
        .def(py::init<>())
        .def_readwrite("seed", &core::NetworkConfig::seed)
        .def_readwrite("input_shape", &core::NetworkConfig::input_shape)
        .def_readwrite("output_size", &core::NetworkConfig::output_size)
        .def_readwrite("cost_function", &core::NetworkConfig::cost_function)
        .def_readwrite("output_activation", &core::NetworkConfig::output_activation)
        .def_readwrite("hidden_activation", &core::NetworkConfig::hidden_activation)
        .def_readwrite("device", &core::NetworkConfig::device);

    // Normalization config
    py::class_<training::NormalizationConfig>(m, "NormalizationConfig")
        .def(py::init<>())
        .def_readwrite("normalize_input", &training::NormalizationConfig::normalize_input)
        .def_readwrite("normalize_output", &training::NormalizationConfig::normalize_output)
        .def_readwrite("method", &training::NormalizationConfig::method)
        .def_readwrite("epsilon", &training::NormalizationConfig::epsilon);

    // Trainer config
    py::class_<training::TrainerConfig>(m, "TrainerConfig")
        .def(py::init<>())
        .def_readwrite("initial_learning_rate", &training::TrainerConfig::initial_learning_rate)
        .def_readwrite("min_learning_rate", &training::TrainerConfig::min_learning_rate)
        .def_readwrite("learning_rate_decay", &training::TrainerConfig::learning_rate_decay)
        .def_readwrite("max_epochs", &training::TrainerConfig::max_epochs)
        .def_readwrite("enable_dynamic_layers", &training::TrainerConfig::enable_dynamic_layers)
        .def_readwrite("enable_trainable_scheduling", &training::TrainerConfig::enable_trainable_scheduling)
        .def_readwrite("enable_early_stopping", &training::TrainerConfig::enable_early_stopping)
        .def_readwrite("patience", &training::TrainerConfig::patience)
        .def_readwrite("cancer_threshold", &training::TrainerConfig::cancer_threshold)
        .def_readwrite("alzheimer_threshold", &training::TrainerConfig::alzheimer_threshold)
        .def_readwrite("enable_gradient_clipping", &training::TrainerConfig::enable_gradient_clipping)
        .def_readwrite("gradient_clip_value", &training::TrainerConfig::gradient_clip_value)
        .def_readwrite("normalization", &training::TrainerConfig::normalization)
        .def_readwrite("phase4_patience", &training::TrainerConfig::phase4_patience);

    // Training result
    py::class_<training::TrainingResult>(m, "TrainingResult")
        .def(py::init<>())
        .def_readonly("success", &training::TrainingResult::success)
        .def_readonly("epochs_completed", &training::TrainingResult::epochs_completed)
        .def_readonly("final_cost", &training::TrainingResult::final_cost)
        .def_readonly("final_efficiency", &training::TrainingResult::final_efficiency)
        .def_readonly("best_cost", &training::TrainingResult::best_cost)
        .def_readonly("best_efficiency", &training::TrainingResult::best_efficiency)
        .def_readonly("stopping_reason", &training::TrainingResult::stopping_reason)
        .def_readonly("cost_history", &training::TrainingResult::cost_history)
        .def_readonly("efficiency_history", &training::TrainingResult::efficiency_history)
        .def_property_readonly("training_time", [](const training::TrainingResult& r) {
            return r.training_time.count();
        });

    // Health report
    py::class_<dynamics::HealthReport>(m, "HealthReport")
        .def(py::init<>())
        .def_readonly("state", &dynamics::HealthReport::state)
        .def_readonly("cancer_score", &dynamics::HealthReport::cancer_score)
        .def_readonly("alzheimer_score", &dynamics::HealthReport::alzheimer_score)
        .def_readonly("overall_health", &dynamics::HealthReport::overall_health)
        .def_readonly("diagnosis", &dynamics::HealthReport::diagnosis)
        .def_readonly("recommendations", &dynamics::HealthReport::recommendations)
        .def_readonly("current_layers", &dynamics::HealthReport::current_layers)
        .def_readonly("current_nodes", &dynamics::HealthReport::current_nodes);

    // Network class
    py::class_<PyNetwork<float>>(m, "Network")
        .def(py::init<const core::NetworkConfig&>())
        .def("forward", &PyNetwork<float>::forward)
        .def("predict", &PyNetwork<float>::predict)
        .def("num_layers", &PyNetwork<float>::num_layers)
        .def("num_parameters", &PyNetwork<float>::num_parameters)
        .def("efficiency_report", &PyNetwork<float>::efficiency_report,
             py::arg("include_nodes") = false)
        .def("to", &PyNetwork<float>::to, py::arg("device"),
             "Move network to specified device (CPU or CUDA)")
        .def("device", &PyNetwork<float>::device,
             "Get current device");

    // Trainer class
    py::class_<PyTrainer<float>>(m, "Trainer")
        .def(py::init<PyNetwork<float>&, training::CostFunctionType, const training::TrainerConfig&>(),
             py::arg("network"),
             py::arg("cost_type") = training::CostFunctionType::MeanSquaredError,
             py::arg("config") = training::TrainerConfig())
        .def("train", &PyTrainer<float>::train)
        .def("set_epoch_callback", &PyTrainer<float>::set_epoch_callback)
        .def("learning_rate", &PyTrainer<float>::learning_rate)
        .def("batch_size", &PyTrainer<float>::batch_size)
        .def("health_report", &PyTrainer<float>::health_report)
        .def("normalize_input", &PyTrainer<float>::normalize_input)
        .def("denormalize_output", &PyTrainer<float>::denormalize_output)
        .def("get_normalization_params", &PyTrainer<float>::get_normalization_params);

    // Model I/O functions
    m.def("save_model", &save_model<float>,
          py::arg("network"), py::arg("path"), py::arg("name") = "model",
          "Save model to files");

    m.def("load_model", &load_model<float>,
          py::arg("path"),
          "Load model from file");

    // Version info
    m.attr("__version__") = "1.0.0";
    m.attr("__author__") = "Dynamic Neural Network Library";
}
