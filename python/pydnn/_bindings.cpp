/**
 * Python bindings for Dynamic Neural Network library.
 * Uses pybind11 for zero-copy NumPy integration.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <cstring>
#include <cctype>

#ifdef DNN_HAS_OPENMP
#include <omp.h>
#endif

#include "dnn/core/tensor.hpp"
#include "dnn/core/network.hpp"
#include "dnn/core/activations.hpp"
#include "dnn/core/device.hpp"
#include "dnn/training/trainer.hpp"
#include "dnn/training/cost_functions.hpp"
#include "dnn/dynamics/health_monitor.hpp"
#include "dnn/io/model_serializer.hpp"
#ifdef DNN_ENABLE_CUDA
#include "dnn/cuda/cuda_memory_pool.hpp"
#endif

namespace py = pybind11;
using namespace dnn;

// Defined in _bindings_transformer.cpp — registers the
// `_dnn_core.transformer_ops` submodule.
void register_transformer_ops(py::module_& m);

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

    /**
     * Vectorised predict over a batch of N samples.
     *
     * Accepts a 2-D `(N, F)` numpy array (or any contiguous-castable view),
     * runs `network_.forward` on each row inside C++ with the GIL released,
     * and returns a single `(N, output)` numpy array. This bypasses the
     * Python per-sample loop in `DynamicNetwork.predict()` and the per-call
     * binding overhead that came with it.
     */
    py::array_t<T> predict_batch(py::array_t<T> input) {
        auto input_c = py::array::ensure(input,
            py::array::c_style | py::array::forcecast);
        if (!input_c) {
            throw std::runtime_error("predict_batch: input must be convertible "
                                     "to a C-contiguous float array");
        }
        py::buffer_info buf = input_c.request();
        if (buf.ndim < 2) {
            throw std::runtime_error("predict_batch: expected at least a 2-D "
                                     "array shaped (N, F)");
        }

        size_t n_samples = static_cast<size_t>(buf.shape[0]);
        size_t input_size = 1;
        for (py::ssize_t d = 1; d < buf.ndim; ++d) {
            input_size *= static_cast<size_t>(buf.shape[d]);
        }

        const T* src = static_cast<const T*>(buf.ptr);

        // Pre-allocate the output buffer once we know the network's output
        // size. Run the first sample under the GIL so we can size the array
        // from its result, then release the GIL for the remaining rows.
        size_t output_size = 0;
        std::vector<T> flat;
        {
            core::Tensor<T> first_in(std::vector<size_t>{input_size});
            std::memcpy(first_in.data(), src, input_size * sizeof(T));
            core::Tensor<T> first_out = network_.forward(first_in);
            output_size = first_out.size();
            flat.resize(n_samples * output_size);
            std::memcpy(flat.data(), first_out.data(), output_size * sizeof(T));
        }

        if (n_samples > 1) {
            py::gil_scoped_release release;
            for (size_t i = 1; i < n_samples; ++i) {
                core::Tensor<T> inp(std::vector<size_t>{input_size});
                std::memcpy(inp.data(), src + i * input_size,
                            input_size * sizeof(T));
                core::Tensor<T> out = network_.forward(inp);
                if (out.size() != output_size) {
                    // Should not happen unless the network mutates between
                    // rows; surface it loudly rather than corrupting output.
                    throw std::runtime_error(
                        "predict_batch: output size changed between samples");
                }
                std::memcpy(flat.data() + i * output_size,
                            out.data(), output_size * sizeof(T));
            }
        }

        std::vector<py::ssize_t> shape{
            static_cast<py::ssize_t>(n_samples),
            static_cast<py::ssize_t>(output_size)
        };
        return py::array_t<T>(shape, flat.data());
    }

    size_t num_layers() const { return network_.num_layers(); }
    size_t num_parameters() const { return network_.num_parameters(); }

    /**
     * Hard-remove all soft-pruned (dormant) nodes and layers from the
     * underlying network. Returns the number of nodes physically erased.
     * Intended to be called once after fit() finishes so the inference
     * model is the lean compacted form.
     */
    size_t compact() { return network_.compact(); }

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
        // Force C-contiguous, casting if needed. Without this, a strided or
        // transposed numpy view would silently copy garbage into Tensor::data().
        auto inputs_c  = py::array::ensure(inputs,
            py::array::c_style | py::array::forcecast);
        auto targets_c = py::array::ensure(targets,
            py::array::c_style | py::array::forcecast);
        if (!inputs_c || !targets_c) {
            throw std::runtime_error("Inputs and targets must be convertible to "
                                     "a C-contiguous float array");
        }
        py::buffer_info input_buf  = inputs_c.request();
        py::buffer_info target_buf = targets_c.request();

        if (input_buf.ndim < 2 || target_buf.ndim < 2) {
            throw std::runtime_error("Inputs and targets must have at least 2 dimensions");
        }
        if (input_buf.shape[0] != target_buf.shape[0]) {
            throw std::runtime_error("Inputs and targets must have the same n_samples");
        }

        size_t n_samples = static_cast<size_t>(input_buf.shape[0]);
        size_t input_size = 1;
        for (py::ssize_t i = 1; i < input_buf.ndim; ++i) {
            input_size *= static_cast<size_t>(input_buf.shape[i]);
        }
        size_t target_size = static_cast<size_t>(target_buf.shape[1]);

        std::vector<core::Tensor<T>> input_tensors;
        std::vector<core::Tensor<T>> target_tensors;
        input_tensors.reserve(n_samples);
        target_tensors.reserve(n_samples);

        const T* input_ptr  = static_cast<const T*>(input_buf.ptr);
        const T* target_ptr = static_cast<const T*>(target_buf.ptr);

        // Bulk-copy each sample's contiguous row into a fresh Tensor; cheaper
        // than std::copy element-by-element and faithful to Tensor's owning
        // semantics (Trainer reorders / stores samples).
        for (size_t i = 0; i < n_samples; ++i) {
            core::Tensor<T> inp(std::vector<size_t>{input_size});
            std::memcpy(inp.data(), input_ptr + i * input_size,
                        input_size * sizeof(T));
            input_tensors.push_back(std::move(inp));

            core::Tensor<T> tgt(std::vector<size_t>{target_size});
            std::memcpy(tgt.data(), target_ptr + i * target_size,
                        target_size * sizeof(T));
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

    // Transformer ops submodule (matmul, softmax, layernorm, gelu, ...)
    register_transformer_ops(m);

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

    // CudaMemoryPool allocation mode. "device" = cudaMalloc (default,
    // VRAM-only, hard OOM). "managed" = cudaMallocManaged (Unified
    // Memory; pages between VRAM, host RAM and OS swap as needed).
    // No-op when the binding wasn't built with DNN_ENABLE_CUDA.
    m.def("set_cuda_memory_mode", [](const std::string& mode) {
#ifdef DNN_ENABLE_CUDA
        std::string m_lower;
        m_lower.reserve(mode.size());
        for (char c : mode) m_lower.push_back(static_cast<char>(std::tolower(c)));
        if (m_lower == "device") {
            cuda::CudaMemoryPool::instance().set_alloc_mode(
                cuda::CudaMemoryPool::AllocMode::Device);
        } else if (m_lower == "managed") {
            cuda::CudaMemoryPool::instance().set_alloc_mode(
                cuda::CudaMemoryPool::AllocMode::Managed);
        } else {
            throw std::invalid_argument(
                "set_cuda_memory_mode: expected 'device' or 'managed', got '"
                + mode + "'");
        }
#else
        (void)mode;
#endif
    }, "Configure CudaMemoryPool: 'device' (cudaMalloc, VRAM only) or "
       "'managed' (cudaMallocManaged, transparent VRAM<->RAM<->disk paging). "
       "Set this before constructing CUDA networks; affects future allocations only.");

    m.def("get_cuda_memory_mode", []() -> std::string {
#ifdef DNN_ENABLE_CUDA
        auto mode = cuda::CudaMemoryPool::instance().alloc_mode();
        return mode == cuda::CudaMemoryPool::AllocMode::Managed ? "managed" : "device";
#else
        return "device";
#endif
    }, "Return the current CudaMemoryPool allocation mode.");

    // OpenMP thread-count control. libgomp already honours OMP_NUM_THREADS at
    // startup; these helpers let Python callers override it at runtime, e.g.
    //   pydnn.set_num_threads(os.cpu_count())
    // before fit() to ensure all cores are engaged. No-op when built without
    // OpenMP.
    m.def("set_num_threads", [](int n) {
#ifdef DNN_HAS_OPENMP
        if (n > 0) omp_set_num_threads(n);
#else
        (void)n;
#endif
    }, "Set the OpenMP thread count for CPU training. Pass a positive int.");

    m.def("get_num_threads", []() -> int {
#ifdef DNN_HAS_OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }, "Return the current OpenMP max thread count (1 when OpenMP is disabled).");

    m.def("openmp_available", []() -> bool {
#ifdef DNN_HAS_OPENMP
        return true;
#else
        return false;
#endif
    }, "True iff the loaded _dnn_core was built with OpenMP.");

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

    // Layer-mutation thresholds (was hardcoded inside LayerManager)
    py::class_<dynamics::LayerManagerConfig>(m, "LayerManagerConfig")
        .def(py::init<>())
        .def_readwrite("efficiency_threshold", &dynamics::LayerManagerConfig::efficiency_threshold)
        .def_readwrite("saturation_threshold", &dynamics::LayerManagerConfig::saturation_threshold)
        .def_readwrite("redundancy_threshold", &dynamics::LayerManagerConfig::redundancy_threshold)
        // Local-maximum efficiency gate (default-on; opt out via
        // shrink_requires_plateau=False for legacy reproducibility).
        .def_readwrite("shrink_requires_plateau",
                       &dynamics::LayerManagerConfig::shrink_requires_plateau)
        .def_readwrite("plateau_window",
                       &dynamics::LayerManagerConfig::plateau_window)
        .def_readwrite("plateau_slope_epsilon",
                       &dynamics::LayerManagerConfig::plateau_slope_epsilon)
        .def_readwrite("min_epochs_before_shrink",
                       &dynamics::LayerManagerConfig::min_epochs_before_shrink)
        .def_readwrite("shrink_cooldown_epochs",
                       &dynamics::LayerManagerConfig::shrink_cooldown_epochs);

    // Reward / penalty + emotional-state thresholds (was constructed locally in Phase 3)
    py::class_<training::RewardPenaltyConfig>(m, "RewardPenaltyConfig")
        .def(py::init<>())
        .def_readwrite("cost_improvement_threshold", &training::RewardPenaltyConfig::cost_improvement_threshold)
        .def_readwrite("efficiency_improvement_threshold", &training::RewardPenaltyConfig::efficiency_improvement_threshold)
        .def_readwrite("min_learning_rate", &training::RewardPenaltyConfig::min_learning_rate)
        .def_readwrite("max_learning_rate", &training::RewardPenaltyConfig::max_learning_rate)
        .def_readwrite("baseline_learning_rate", &training::RewardPenaltyConfig::baseline_learning_rate)
        .def_readwrite("max_adjustment_factor", &training::RewardPenaltyConfig::max_adjustment_factor)
        .def_readwrite("min_adjustment_factor", &training::RewardPenaltyConfig::min_adjustment_factor)
        .def_readwrite("extreme_threshold", &training::RewardPenaltyConfig::extreme_threshold)
        .def_readwrite("moderate_threshold", &training::RewardPenaltyConfig::moderate_threshold)
        .def_readwrite("window_size", &training::RewardPenaltyConfig::window_size);

    // Adaptive batch sizing
    py::class_<training::BatchConfig>(m, "BatchConfig")
        .def(py::init<>())
        .def_readwrite("min_batch_size", &training::BatchConfig::min_batch_size)
        .def_readwrite("max_batch_size", &training::BatchConfig::max_batch_size)
        .def_readwrite("growth_rate", &training::BatchConfig::growth_rate)
        .def_readwrite("growth_interval_epochs", &training::BatchConfig::growth_interval_epochs)
        .def_readwrite("shuffle", &training::BatchConfig::shuffle)
        .def_readwrite("power_of_two", &training::BatchConfig::power_of_two);

    // Trainer config
    py::class_<training::TrainerConfig>(m, "TrainerConfig")
        .def(py::init<>())
        .def_readwrite("initial_learning_rate", &training::TrainerConfig::initial_learning_rate)
        .def_readwrite("min_learning_rate", &training::TrainerConfig::min_learning_rate)
        .def_readwrite("learning_rate_decay", &training::TrainerConfig::learning_rate_decay)
        .def_readwrite("decay_interval", &training::TrainerConfig::decay_interval)
        .def_readwrite("max_epochs", &training::TrainerConfig::max_epochs)
        .def_readwrite("enable_dynamic_layers", &training::TrainerConfig::enable_dynamic_layers)
        .def_readwrite("layer_adjustment_interval", &training::TrainerConfig::layer_adjustment_interval)
        .def_readwrite("enable_trainable_scheduling", &training::TrainerConfig::enable_trainable_scheduling)
        .def_readwrite("enable_early_stopping", &training::TrainerConfig::enable_early_stopping)
        .def_readwrite("patience", &training::TrainerConfig::patience)
        .def_readwrite("min_improvement", &training::TrainerConfig::min_improvement)
        .def_readwrite("min_epochs_for_early_stop", &training::TrainerConfig::min_epochs_for_early_stop)
        .def_readwrite("batch_config", &training::TrainerConfig::batch_config)
        .def_readwrite("layer_manager_config", &training::TrainerConfig::layer_manager_config)
        .def_readwrite("reward_penalty_config", &training::TrainerConfig::reward_penalty_config)
        .def_readwrite("cancer_threshold", &training::TrainerConfig::cancer_threshold)
        .def_readwrite("alzheimer_threshold", &training::TrainerConfig::alzheimer_threshold)
        .def_readwrite("enable_gradient_clipping", &training::TrainerConfig::enable_gradient_clipping)
        .def_readwrite("gradient_clip_value", &training::TrainerConfig::gradient_clip_value)
        .def_readwrite("normalization", &training::TrainerConfig::normalization)
        .def_readwrite("phase4_patience", &training::TrainerConfig::phase4_patience)
        .def_readwrite("phase4_min_improvement", &training::TrainerConfig::phase4_min_improvement)
        .def_readwrite("runtime_enabled", &training::TrainerConfig::runtime_enabled,
                       "When true, route train_phased() through the concurrent "
                       "StageController (parallel Estimation observer + soft "
                       "topology + adaptive scalars). Default false.")
        .def_readwrite("batched_train_forward", &training::TrainerConfig::batched_train_forward,
                       "When true, train_epoch issues one rank-2 forward and "
                       "one rank-2 backward per batch instead of B rank-1 "
                       "calls. On CUDA this collapses per-sample CPU<->GPU "
                       "round-trips into one cuda_gemm per layer. Numerically "
                       "equivalent on CPU modulo float summation order. "
                       "Default true. Set false to fall back to the per-sample "
                       "loop for benchmarking.");

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
        .def("predict_batch", &PyNetwork<float>::predict_batch,
             py::arg("input"),
             "Vectorised predict over a 2-D (N, F) batch; returns (N, output).")
        .def("num_layers", &PyNetwork<float>::num_layers)
        .def("num_parameters", &PyNetwork<float>::num_parameters)
        .def("compact", &PyNetwork<float>::compact,
             "Hard-remove dormant (soft-pruned) nodes and layers. "
             "Call once after fit() to produce a lean inference model.")
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
    m.attr("__version__") = "0.0.1";
    m.attr("__author__") = "Dynamic Neural Network Library";
}
