// basic_network.cpp - Basic example of using the Dynamic Neural Network
//
// This example demonstrates:
// - Creating a network with automatic architecture
// - Training on simple XOR problem
// - Using the trainer with phased training

#include "dnn/core/network.hpp"
#include "dnn/core/tensor.hpp"
#include "dnn/training/trainer.hpp"
#include "dnn/training/cost_functions.hpp"

#include <iostream>
#include <vector>

using namespace dnn;
using namespace dnn::core;
using namespace dnn::training;

int main() {
    std::cout << "=== Dynamic Neural Network Basic Example ===\n\n";

    // Configure network
    NetworkConfig config;
    config.seed = 42;
    config.input_shape = {2};  // 2 input features
    config.output_size = 1;    // 1 output
    config.hidden_activation = ActivationType::ReLU;
    config.output_activation = ActivationType::Sigmoid;

    // Create network
    Network<float> network(config);

    std::cout << "Network created with:\n";
    std::cout << "  - Layers: " << network.num_layers() << "\n";
    std::cout << "  - Total parameters: " << network.num_parameters() << "\n\n";

    // Create XOR training data
    std::vector<Tensor<float>> inputs = {
        Tensor<float>({0.0f, 0.0f}),
        Tensor<float>({0.0f, 1.0f}),
        Tensor<float>({1.0f, 0.0f}),
        Tensor<float>({1.0f, 1.0f})
    };

    std::vector<Tensor<float>> targets = {
        Tensor<float>({0.0f}),  // 0 XOR 0 = 0
        Tensor<float>({1.0f}),  // 0 XOR 1 = 1
        Tensor<float>({1.0f}),  // 1 XOR 0 = 1
        Tensor<float>({0.0f})   // 1 XOR 1 = 0
    };

    // Configure trainer
    TrainerConfig trainer_config;
    trainer_config.initial_learning_rate = 0.1;
    trainer_config.max_epochs = 100;
    trainer_config.enable_early_stopping = true;
    trainer_config.patience = 20;

    // Create trainer
    Trainer<float> trainer(network, training::CostFunctionType::BinaryCrossEntropy, trainer_config);

    // Set up epoch callback for progress reporting
    trainer.set_epoch_callback([](uint64_t epoch, double cost, double efficiency,
                                  const Network<float>& net) {
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch
                      << " - Cost: " << cost
                      << " - Efficiency: " << efficiency
                      << " - Layers: " << net.num_layers()
                      << "\n";
        }
    });

    std::cout << "Training XOR problem...\n\n";

    // Train the network
    auto result = trainer.train(inputs, targets);

    std::cout << "\n=== Training Complete ===\n";
    std::cout << "  - Epochs: " << result.epochs_completed << "\n";
    std::cout << "  - Final cost: " << result.final_cost << "\n";
    std::cout << "  - Final efficiency: " << result.final_efficiency << "\n";
    std::cout << "  - Best cost: " << result.best_cost << "\n";
    std::cout << "  - Training time: " << result.training_time.count() << "ms\n";
    std::cout << "  - Stopping reason: " << result.stopping_reason << "\n\n";

    // Test predictions
    std::cout << "=== Predictions ===\n";
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = network.forward(inputs[i]);
        std::cout << "  Input: [" << inputs[i][0] << ", " << inputs[i][1] << "]"
                  << " -> Output: " << output[0]
                  << " (Expected: " << targets[i][0] << ")\n";
    }

    std::cout << "\n=== Final Network Architecture ===\n";
    std::cout << "  - Layers: " << network.num_layers() << "\n";
    std::cout << "  - Total nodes: " << network.num_nodes() << "\n";
    std::cout << "  - Total parameters: " << network.num_parameters() << "\n";

    return 0;
}
