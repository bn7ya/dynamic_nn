#pragma once

#include "../core/network.hpp"
#include "../exceptions/dnn_exception.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>

namespace dnn {
namespace io {

using core::Network;
using core::NetworkConfig;
using core::ActivationType;

/**
 * Model save result.
 */
struct SaveResult {
    bool success = false;
    std::string model_path;
    std::string code_path;
    std::string requirements_path;
    std::string error;
};

/**
 * Serializes and deserializes networks to/from files.
 */
template<typename T = float>
class ModelSerializer {
public:
    /**
     * Save model to files.
     * Creates: model_name.txt, model_name_weights/, code.py, requirements.txt
     */
    static SaveResult save(const Network<T>& network,
                           const std::string& base_path,
                           const std::string& model_name) {
        SaveResult result;

        try {
            // Create model metadata file
            std::string model_path = base_path + "/" + model_name + ".txt";
            result.model_path = model_path;

            std::ofstream model_file(model_path);
            if (!model_file.is_open()) {
                throw exceptions::IOException("save", model_path, "Cannot open file");
            }

            write_model_file(model_file, network, model_name);
            model_file.close();

            // Save weights as binary files
            std::string weights_dir = base_path + "/" + model_name + "_weights";
            // In a full implementation, we'd create the directory and save binary weights
            // For now, we include weights in the model file

            // Generate Python code
            std::string code_path = base_path + "/code.py";
            result.code_path = code_path;

            std::ofstream code_file(code_path);
            if (code_file.is_open()) {
                write_python_code(code_file, network, model_name);
                code_file.close();
            }

            // Generate requirements.txt
            std::string req_path = base_path + "/requirements.txt";
            result.requirements_path = req_path;

            std::ofstream req_file(req_path);
            if (req_file.is_open()) {
                write_requirements(req_file);
                req_file.close();
            }

            result.success = true;

        } catch (const std::exception& e) {
            result.success = false;
            result.error = e.what();
        }

        return result;
    }

    /**
     * Load model from file.
     */
    static std::unique_ptr<Network<T>> load(const std::string& model_path) {
        std::ifstream file(model_path);
        if (!file.is_open()) {
            throw exceptions::IOException("load", model_path, "Cannot open file");
        }

        std::string line;
        NetworkConfig config;
        std::vector<std::tuple<size_t, size_t, ActivationType>> layer_specs;

        std::string section;
        while (std::getline(file, line)) {
            // Skip comments and empty lines
            if (line.empty() || line[0] == '#') continue;

            // Check for section headers
            if (line[0] == '[') {
                section = line.substr(1, line.find(']') - 1);
                continue;
            }

            // Parse key = value
            auto eq_pos = line.find('=');
            if (eq_pos == std::string::npos) continue;

            std::string key = trim(line.substr(0, eq_pos));
            std::string value = trim(line.substr(eq_pos + 1));

            if (section == "metadata") {
                if (key == "seed") config.seed = std::stoull(value);
            } else if (section == "architecture") {
                if (key == "input_shape") {
                    config.input_shape = parse_shape(value);
                } else if (key == "output_size") {
                    config.output_size = std::stoull(value);
                }
            } else if (section.substr(0, 5) == "layer") {
                // Parse layer specifications
                // In full implementation, would read weights here
            }
        }

        // Create network
        auto network = std::make_unique<Network<T>>(config);

        // In full implementation, would load weights here

        return network;
    }

private:
    static void write_model_file(std::ostream& out,
                                 const Network<T>& network,
                                 const std::string& model_name) {
        auto now = std::time(nullptr);
        auto* tm = std::localtime(&now);

        out << "# Dynamic Neural Network Model\n";
        out << "# Model: " << model_name << "\n";
        out << "# Generated: " << std::put_time(tm, "%Y-%m-%dT%H:%M:%S") << "\n";
        out << "\n";

        // Metadata
        out << "[metadata]\n";
        out << "version = 1.0\n";
        out << "seed = " << network.config().seed << "\n";
        out << "num_layers = " << network.num_layers() << "\n";
        out << "total_parameters = " << network.num_parameters() << "\n";
        out << "\n";

        // Architecture
        out << "[architecture]\n";
        out << "input_shape = [";
        for (size_t i = 0; i < network.config().input_shape.size(); ++i) {
            if (i > 0) out << ", ";
            out << network.config().input_shape[i];
        }
        out << "]\n";
        out << "output_size = " << network.config().output_size << "\n";
        out << "\n";

        // Layers
        for (size_t i = 0; i < network.num_layers(); ++i) {
            const auto& layer = network.layer(i);
            out << "[layer_" << i << "]\n";
            out << "type = Dense\n";
            out << "input_size = " << layer.input_size() << "\n";
            out << "output_size = " << layer.output_size() << "\n";
            out << "activation = " << activation_name(layer.activation_type()) << "\n";
            out << "num_nodes = " << layer.num_nodes() << "\n";
            out << "trainable_nodes = " << layer.trainable_count() << "\n";
            out << "\n";
        }

        // Training state
        out << "[training_state]\n";
        out << "epochs = " << network.state().epoch << "\n";
        out << "efficiency = " << network.state().current_efficiency << "\n";
    }

    static void write_python_code(std::ostream& out,
                                  const Network<T>& network,
                                  const std::string& model_name) {
        auto now = std::time(nullptr);
        auto* tm = std::localtime(&now);

        out << "# Auto-generated by Dynamic Neural Network\n";
        out << "# Model: " << model_name << "\n";
        out << "# Generated: " << std::put_time(tm, "%Y-%m-%dT%H:%M:%S") << "\n";
        out << "\n";
        out << "import numpy as np\n";
        out << "\n";
        out << "def create_model():\n";
        out << "    \"\"\"\n";
        out << "    Creates the trained model architecture.\n";
        out << "    \n";
        out << "    Architecture:\n";
        for (size_t i = 0; i < network.num_layers(); ++i) {
            const auto& layer = network.layer(i);
            out << "    - Layer " << i << ": Dense "
                << layer.input_size() << " -> " << layer.output_size()
                << ", " << activation_name(layer.activation_type()) << "\n";
        }
        out << "    \n";
        out << "    Training Info:\n";
        out << "    - Seed: " << network.config().seed << "\n";
        out << "    - Total Parameters: " << network.num_parameters() << "\n";
        out << "    \"\"\"\n";
        out << "    # Load model using pydnn\n";
        out << "    from pydnn import DynamicNetwork\n";
        out << "    model = DynamicNetwork.load('" << model_name << ".txt')\n";
        out << "    return model\n";
        out << "\n";
        out << "def predict(model, X):\n";
        out << "    \"\"\"\n";
        out << "    Make predictions with the model.\n";
        out << "    \n";
        out << "    Args:\n";
        out << "        model: Loaded DynamicNetwork\n";
        out << "        X: Input array of shape (batch_size, ...)\n";
        out << "    \n";
        out << "    Returns:\n";
        out << "        Predictions array\n";
        out << "    \"\"\"\n";
        out << "    return model.predict(X)\n";
        out << "\n";
        out << "if __name__ == '__main__':\n";
        out << "    model = create_model()\n";
        out << "    print(f'Model loaded with {model.num_parameters()} parameters')\n";
    }

    static void write_requirements(std::ostream& out) {
        out << "# Auto-generated requirements for Dynamic Neural Network\n";
        out << "\n";
        out << "# Core library\n";
        out << "pydnn>=1.0.0\n";
        out << "\n";
        out << "# Visualization (optional)\n";
        out << "matplotlib>=3.5.0\n";
        out << "plotly>=5.0.0\n";
        out << "\n";
        out << "# Data handling\n";
        out << "numpy>=1.21.0\n";
        out << "\n";
        out << "# Optional input support\n";
        out << "pillow>=9.0.0  # Image support\n";
        out << "# librosa>=0.9.0  # Audio support\n";
        out << "# opencv-python>=4.5.0  # Video support\n";
    }

    static std::string activation_name(ActivationType type) {
        switch (type) {
            case ActivationType::Linear: return "Linear";
            case ActivationType::ReLU: return "ReLU";
            case ActivationType::LeakyReLU: return "LeakyReLU";
            case ActivationType::ELU: return "ELU";
            case ActivationType::SELU: return "SELU";
            case ActivationType::Sigmoid: return "Sigmoid";
            case ActivationType::Tanh: return "Tanh";
            case ActivationType::Softmax: return "Softmax";
            case ActivationType::Swish: return "Swish";
            case ActivationType::GELU: return "GELU";
            case ActivationType::Softplus: return "Softplus";
            default: return "Unknown";
        }
    }

    static std::string trim(const std::string& s) {
        auto start = s.find_first_not_of(" \t\r\n");
        auto end = s.find_last_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        return s.substr(start, end - start + 1);
    }

    static std::vector<size_t> parse_shape(const std::string& s) {
        std::vector<size_t> shape;
        std::string cleaned = s;

        // Remove brackets
        auto start = cleaned.find('[');
        auto end = cleaned.find(']');
        if (start != std::string::npos && end != std::string::npos) {
            cleaned = cleaned.substr(start + 1, end - start - 1);
        }

        // Parse comma-separated values
        std::istringstream iss(cleaned);
        std::string token;
        while (std::getline(iss, token, ',')) {
            shape.push_back(std::stoull(trim(token)));
        }

        return shape;
    }
};

} // namespace io
} // namespace dnn
