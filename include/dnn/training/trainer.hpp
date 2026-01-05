#pragma once

#include "../core/network.hpp"
#include "../core/tensor.hpp"
#include "cost_functions.hpp"
#include "batch_manager.hpp"
#include "early_stopping.hpp"
#include "../dynamics/layer_manager.hpp"
#include "../dynamics/health_monitor.hpp"
#include "../dynamics/trainable_scheduler.hpp"
#include <functional>
#include <chrono>
#include <memory>

namespace dnn {
namespace training {

using core::Network;
using core::Tensor;
using dynamics::LayerManager;
using dynamics::HealthMonitor;
using dynamics::TrainableScheduler;

/**
 * Training result.
 */
struct TrainingResult {
    bool success = false;
    uint64_t epochs_completed = 0;
    double final_cost = 0.0;
    double final_efficiency = 0.0;
    double best_cost = std::numeric_limits<double>::max();
    double best_efficiency = 0.0;
    std::string stopping_reason;
    std::vector<double> cost_history;
    std::vector<double> efficiency_history;
    std::chrono::milliseconds training_time{0};
};

/**
 * Trainer configuration.
 */
struct TrainerConfig {
    // Learning rate (auto-adjusted internally)
    double initial_learning_rate = 0.01;
    double min_learning_rate = 0.0001;
    double learning_rate_decay = 0.95;
    size_t decay_interval = 10;

    // Training limits
    size_t max_epochs = 1000;

    // Dynamic architecture
    bool enable_dynamic_layers = true;
    size_t layer_adjustment_interval = 10;  // Epochs between structure changes

    // Trainable node scheduling
    bool enable_trainable_scheduling = true;
    dynamics::TrainableConfig trainable_config;

    // Early stopping
    bool enable_early_stopping = true;
    size_t patience = 20;
    double min_improvement = 0.001;

    // Batch management
    BatchConfig batch_config;

    // Health monitoring
    double cancer_threshold = 0.7;
    double alzheimer_threshold = 0.7;

    // Numerical stability
    double gradient_clip_value = 1.0;
    bool enable_gradient_clipping = true;
};

/**
 * Epoch callback type.
 */
template<typename T>
using EpochCallback = std::function<void(uint64_t epoch, double cost, double efficiency,
                                         const Network<T>& network)>;

/**
 * Main trainer class.
 * Orchestrates training with automatic parameter adjustment.
 */
template<typename T = float>
class Trainer {
public:
    Trainer(Network<T>& network,
            CostFunctionType cost_type = CostFunctionType::MeanSquaredError,
            const TrainerConfig& config = TrainerConfig())
        : network_(network)
        , config_(config)
        , cost_function_(CostFunction<T>::create(cost_type))
        , batch_manager_(config.batch_config)
        , health_monitor_(network, config.cancer_threshold, config.alzheimer_threshold)
        , layer_manager_(network, health_monitor_)
        , trainable_scheduler_(network, config.trainable_config)
        , early_stopping_(config.patience, config.min_improvement, 10)
        , learning_rate_(config.initial_learning_rate) {}

    /**
     * Train the network.
     */
    TrainingResult train(const std::vector<Tensor<T>>& inputs,
                         const std::vector<Tensor<T>>& targets) {
        auto start_time = std::chrono::high_resolution_clock::now();

        TrainingResult result;
        result.cost_history.reserve(config_.max_epochs);
        result.efficiency_history.reserve(config_.max_epochs);

        // Split data for validation (80/10/10)
        size_t train_end = static_cast<size_t>(inputs.size() * 0.8);
        size_t val_end = static_cast<size_t>(inputs.size() * 0.9);

        std::vector<Tensor<T>> train_inputs(inputs.begin(), inputs.begin() + train_end);
        std::vector<Tensor<T>> train_targets(targets.begin(), targets.begin() + train_end);
        std::vector<Tensor<T>> val_inputs(inputs.begin() + train_end, inputs.begin() + val_end);
        std::vector<Tensor<T>> val_targets(targets.begin() + train_end, targets.begin() + val_end);

        double previous_cost = std::numeric_limits<double>::max();

        for (uint64_t epoch = 0; epoch < config_.max_epochs; ++epoch) {
            // Update components
            batch_manager_.update_epoch(epoch);
            health_monitor_.update_epoch(epoch);

            if (config_.enable_trainable_scheduling) {
                trainable_scheduler_.update_epoch(epoch);
            }

            // Train one epoch
            double epoch_cost = train_epoch(train_inputs, train_targets);

            // Validate
            double val_cost = 0.0;
            if (!val_inputs.empty()) {
                val_cost = evaluate(val_inputs, val_targets);
            } else {
                val_cost = epoch_cost;
            }

            // Compute efficiency metrics
            auto efficiency_metrics = early_stopping_.compute_metrics(
                network_, val_cost, previous_cost);

            // Record history
            result.cost_history.push_back(val_cost);
            result.efficiency_history.push_back(efficiency_metrics.overall_efficiency);

            // Track best
            if (val_cost < result.best_cost) {
                result.best_cost = val_cost;
            }
            if (efficiency_metrics.overall_efficiency > result.best_efficiency) {
                result.best_efficiency = efficiency_metrics.overall_efficiency;
            }

            // Early stopping check
            if (config_.enable_early_stopping) {
                early_stopping_.record_epoch(efficiency_metrics, val_cost);
                auto stopping = early_stopping_.should_stop();
                if (stopping.should_stop) {
                    result.stopping_reason = stopping.reason;
                    break;
                }
            }

            // Dynamic layer adjustment
            if (config_.enable_dynamic_layers &&
                epoch > 0 && epoch % config_.layer_adjustment_interval == 0) {
                auto decision = layer_manager_.auto_adjust();
                if (decision.action != dynamics::LayerDecision::Action::None) {
                    // Reset learning rate after structure change
                    learning_rate_ = config_.initial_learning_rate;
                }
            }

            // Adjust learning rate
            if (epoch > 0 && epoch % config_.decay_interval == 0) {
                learning_rate_ = std::max(config_.min_learning_rate,
                                         learning_rate_ * config_.learning_rate_decay);
            }

            // Adjust batch size based on efficiency
            batch_manager_.adjust_for_efficiency(efficiency_metrics.overall_efficiency);

            // Callback
            if (epoch_callback_) {
                epoch_callback_(epoch, val_cost, efficiency_metrics.overall_efficiency, network_);
            }

            previous_cost = val_cost;
            result.epochs_completed = epoch + 1;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        result.training_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);

        result.success = true;
        result.final_cost = result.cost_history.empty() ? 0.0 : result.cost_history.back();
        result.final_efficiency = result.efficiency_history.empty() ? 0.0 :
                                  result.efficiency_history.back();

        if (result.stopping_reason.empty()) {
            result.stopping_reason = "Reached maximum epochs";
        }

        return result;
    }

    /**
     * Train for a single epoch.
     */
    double train_epoch(const std::vector<Tensor<T>>& inputs,
                       const std::vector<Tensor<T>>& targets) {
        double total_cost = 0.0;
        size_t num_batches = 0;

        auto iterator = batch_manager_.create_iterator(inputs, targets);

        while (iterator.has_next()) {
            auto [batch_inputs, batch_targets] = iterator.next();

            // Process batch
            double batch_cost = 0.0;
            network_.zero_gradients();

            for (size_t i = 0; i < batch_inputs.size(); ++i) {
                // Forward pass
                auto output = network_.forward(batch_inputs[i]);

                // Compute cost
                T sample_cost = cost_function_->compute(output, batch_targets[i]);
                batch_cost += static_cast<double>(sample_cost);

                // Compute gradient
                auto grad = cost_function_->gradient(output, batch_targets[i]);

                // Backward pass
                network_.backward(grad);
            }

            batch_cost /= batch_inputs.size();
            total_cost += batch_cost;
            num_batches++;

            // Clip gradients if enabled
            if (config_.enable_gradient_clipping) {
                clip_gradients();
            }

            // Apply gradients
            T lr = static_cast<T>(learning_rate_ / batch_inputs.size());
            network_.apply_gradients(lr);
        }

        return total_cost / num_batches;
    }

    /**
     * Evaluate on a dataset (no training).
     */
    double evaluate(const std::vector<Tensor<T>>& inputs,
                    const std::vector<Tensor<T>>& targets) {
        double total_cost = 0.0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            auto output = network_.forward(inputs[i]);
            T sample_cost = cost_function_->compute(output, targets[i]);
            total_cost += static_cast<double>(sample_cost);
        }

        return total_cost / inputs.size();
    }

    /**
     * Set epoch callback.
     */
    void set_epoch_callback(EpochCallback<T> callback) {
        epoch_callback_ = callback;
    }

    /**
     * Get current training state.
     */
    const core::TrainingState& state() const {
        return network_.state();
    }

    /**
     * Get health report.
     */
    dynamics::HealthReport health_report() const {
        return health_monitor_.diagnose();
    }

    /**
     * Get current learning rate.
     */
    double learning_rate() const { return learning_rate_; }

    /**
     * Get batch size.
     */
    size_t batch_size() const { return batch_manager_.current_batch_size(); }

    /**
     * Get configuration.
     */
    const TrainerConfig& config() const { return config_; }

private:
    void clip_gradients() {
        // Simple gradient clipping by value
        // In a full implementation, we'd access the accumulated gradients
        // For now, this is a placeholder
    }

    Network<T>& network_;
    TrainerConfig config_;
    std::unique_ptr<CostFunction<T>> cost_function_;
    BatchManager<T> batch_manager_;
    HealthMonitor<T> health_monitor_;
    LayerManager<T> layer_manager_;
    TrainableScheduler<T> trainable_scheduler_;
    EarlyStopping<T> early_stopping_;
    double learning_rate_;
    EpochCallback<T> epoch_callback_;
};

} // namespace training
} // namespace dnn
