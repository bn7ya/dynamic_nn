/**
 * Simple C++ Example for Dynamic Neural Network
 *
 * Demonstrates basic usage of the DNN library in C++.
 */

#include <iostream>
#include <vector>
#include <cmath>

#include "dnn/core/network.hpp"
#include "dnn/core/tensor.hpp"
#include "dnn/training/trainer.hpp"
#include "dnn/dynamics/health_monitor.hpp"
#include "dnn/io/model_serializer.hpp"

using namespace dnn;

/**
 * Generate synthetic XOR-like training data.
 */
void generate_xor_data(std::vector<core::Tensor<float>>& inputs,
                       std::vector<core::Tensor<float>>& targets,
                       size_t n_samples) {
    std::srand(42);

    for (size_t i = 0; i < n_samples; ++i) {
        // Random 2D input
        float x1 = static_cast<float>(std::rand()) / RAND_MAX;
        float x2 = static_cast<float>(std::rand()) / RAND_MAX;

        // XOR-like target
        float y = (x1 > 0.5f) != (x2 > 0.5f) ? 1.0f : 0.0f;

        core::Tensor<float> input({2});
        input.data()[0] = x1;
        input.data()[1] = x2;
        inputs.push_back(std::move(input));

        core::Tensor<float> target({1});
        target.data()[0] = y;
        targets.push_back(std::move(target));
    }
}

int main() {
    std::cout << "=================================================\n";
    std::cout << "Dynamic Neural Network - C++ Example\n";
    std::cout << "=================================================\n\n";

    // Generate training data
    std::cout << "Generating training data...\n";
    std::vector<core::Tensor<float>> train_inputs;
    std::vector<core::Tensor<float>> train_targets;
    generate_xor_data(train_inputs, train_targets, 1000);
    std::cout << "  Samples: " << train_inputs.size() << "\n\n";

    // Create network configuration
    core::NetworkConfig config;
    config.seed = 42;
    config.input_shape = {2};
    config.output_size = 1;
    config.default_activation = core::ActivationType::ReLU;

    // Create network
    std::cout << "Creating network...\n";
    core::Network<float> network(config);
    std::cout << "  Initial layers: " << network.num_layers() << "\n";
    std::cout << "  Initial parameters: " << network.num_parameters() << "\n\n";

    // Configure trainer
    training::TrainerConfig trainer_config;
    trainer_config.max_epochs = 100;
    trainer_config.enable_dynamic_layers = true;
    trainer_config.enable_early_stopping = true;
    trainer_config.patience = 10;

    // Create trainer
    training::Trainer<float> trainer(
        network,
        training::CostFunctionType::BinaryCrossEntropy,
        trainer_config
    );

    // Set epoch callback
    trainer.set_epoch_callback([](uint64_t epoch, double cost, double efficiency,
                                  const core::Network<float>& net) {
        if (epoch % 10 == 0) {
            std::cout << "  Epoch " << epoch << ": cost=" << cost
                      << ", efficiency=" << efficiency
                      << ", layers=" << net.num_layers() << "\n";
        }
    });

    // Train
    std::cout << "Training...\n";
    auto result = trainer.train(train_inputs, train_targets);

    std::cout << "\nTraining complete!\n";
    std::cout << "  Epochs: " << result.epochs_completed << "\n";
    std::cout << "  Final cost: " << result.final_cost << "\n";
    std::cout << "  Final efficiency: " << result.final_efficiency << "\n";
    std::cout << "  Best cost: " << result.best_cost << "\n";
    std::cout << "  Stopping reason: " << result.stopping_reason << "\n";

    // Check network health
    std::cout << "\nNetwork Health:\n";
    auto health = trainer.health_report();
    std::cout << "  State: " << static_cast<int>(health.state) << "\n";
    std::cout << "  Cancer score: " << health.cancer_score << "\n";
    std::cout << "  Alzheimer score: " << health.alzheimer_score << "\n";
    std::cout << "  Overall health: " << health.overall_health << "\n";
    std::cout << "  Current layers: " << health.current_layers << "\n";
    std::cout << "  Current nodes: " << health.current_nodes << "\n";

    // Test prediction
    std::cout << "\nTest predictions:\n";
    std::vector<std::pair<float, float>> test_cases = {
        {0.2f, 0.2f},  // Expected: 0
        {0.8f, 0.2f},  // Expected: 1
        {0.2f, 0.8f},  // Expected: 1
        {0.8f, 0.8f},  // Expected: 0
    };

    for (const auto& [x1, x2] : test_cases) {
        core::Tensor<float> input({2});
        input.data()[0] = x1;
        input.data()[1] = x2;

        auto output = network.forward(input);
        float prediction = output.data()[0];
        int expected = (x1 > 0.5f) != (x2 > 0.5f) ? 1 : 0;

        std::cout << "  Input: (" << x1 << ", " << x2 << ") "
                  << "-> Prediction: " << prediction
                  << " (Expected: " << expected << ")\n";
    }

    // Save model
    std::cout << "\nSaving model...\n";
    io::ModelSerializer<float>::save(network, "./xor_model", "xor_classifier");
    std::cout << "  Model saved to: ./xor_model\n";

    std::cout << "\nExample complete!\n";

    return 0;
}
