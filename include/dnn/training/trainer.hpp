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
 * Training result with comprehensive diagnostics.
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

    // New diagnostic fields
    int nodes_added = 0;
    int nodes_removed = 0;
    int layers_added = 0;
    int layers_removed = 0;
    std::vector<double> cancer_score_history;
    std::vector<double> alzheimer_score_history;
    std::vector<std::pair<size_t, size_t>> architecture_history;  // (layers, nodes)
    int perturbations_applied = 0;
    double phase1_time = 0.0;
    double phase2_time = 0.0;
    double phase3_time = 0.0;
    size_t estimated_epochs = 0;

    // Phase 4 fields
    double phase4_time = 0.0;
    size_t phase4_epochs = 0;
    double phase4_initial_cost = 0.0;
    double phase4_final_cost = 0.0;
    double phase4_cost_reduction = 0.0;
    bool phase4_early_stopped = false;
    size_t phase4_estimated_epochs = 0;
};

/**
 * Normalization method enum.
 */
enum class NormalizationMethod {
    ZScore,   // (x - mean) / std
    MinMax    // (x - min) / (max - min)
};

/**
 * Normalization configuration.
 */
struct NormalizationConfig {
    bool normalize_input = true;           // Auto-normalize input data
    bool normalize_output = true;          // Auto-normalize output data (for regression)
    NormalizationMethod method = NormalizationMethod::ZScore;
    double epsilon = 1e-8;                 // Prevent division by zero
};

/**
 * Stored normalization parameters for denormalization.
 */
template<typename T>
struct NormalizationParams {
    // Input normalization params
    std::vector<T> input_mean;
    std::vector<T> input_std;
    std::vector<T> input_min;
    std::vector<T> input_max;

    // Output normalization params
    std::vector<T> output_mean;
    std::vector<T> output_std;
    std::vector<T> output_min;
    std::vector<T> output_max;

    bool input_normalized = false;
    bool output_normalized = false;
    NormalizationMethod method = NormalizationMethod::ZScore;
    T epsilon = static_cast<T>(1e-8);
};

/**
 * Trainer configuration.
 */
struct TrainerConfig {
    // Learning rate (auto-adjusted internally)
    double initial_learning_rate = 0.005;         // Reduced from 0.01 to prevent gradient explosion
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
    size_t patience = 30;                         // Increased from 20 for less aggressive stopping
    double min_improvement = 0.0001;              // Reduced from 0.001 for less sensitivity

    // Batch management
    BatchConfig batch_config;

    // Health monitoring
    double cancer_threshold = 0.7;
    double alzheimer_threshold = 0.7;

    // Numerical stability
    double gradient_clip_value = 1.0;
    bool enable_gradient_clipping = true;

    // Normalization (auto-normalize data even if not pre-normalized)
    NormalizationConfig normalization;

    // Phase 4: Standard Training (frozen architecture, accuracy focus)
    bool phase4_enabled = true;
    double phase4_learning_rate = 0.005;          // Reduced from 0.01 for fine-tuning stability
    double phase4_min_learning_rate = 0.0001;     // Lower bound for LR
    double phase4_lr_decay_rate = 0.95;           // Gentle decay
    size_t phase4_lr_decay_interval = 10;         // Epochs between decay
    size_t phase4_batch_size = 64;                // Fixed batch size
    double phase4_target_cost_reduction = 0.5;    // Target 50% additional cost reduction
    size_t phase4_min_epochs = 50;                // Increased from 20 for more training time
    size_t phase4_max_epochs = 200;               // Maximum training epochs
    size_t phase4_patience = 35;                  // Early stopping patience (increased from 15)
    double phase4_min_improvement = 1e-5;         // Minimum cost improvement threshold
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
        , cost_type_(cost_type)
        , cost_function_(CostFunction<T>::create(cost_type))
        , batch_manager_(config.batch_config)
        , health_monitor_(network, config.cancer_threshold, config.alzheimer_threshold)
        , layer_manager_(network, health_monitor_)
        , trainable_scheduler_(network, config.trainable_config)
        , early_stopping_(config.patience, config.min_improvement, 10)
        , learning_rate_(config.initial_learning_rate) {
        // Initialize normalization params
        norm_params_.method = config.normalization.method;
        norm_params_.epsilon = static_cast<T>(config.normalization.epsilon);
    }

    /**
     * Train the network.
     */
    TrainingResult train(const std::vector<Tensor<T>>& inputs,
                         const std::vector<Tensor<T>>& targets) {
        auto start_time = std::chrono::high_resolution_clock::now();

        TrainingResult result;
        result.cost_history.reserve(config_.max_epochs);
        result.efficiency_history.reserve(config_.max_epochs);

        // Compute normalization parameters from all data first
        compute_normalization_params(inputs, targets);

        // Split data for validation (80/10/10)
        size_t train_end = static_cast<size_t>(inputs.size() * 0.8);
        size_t val_end = static_cast<size_t>(inputs.size() * 0.9);

        std::vector<Tensor<T>> train_inputs(inputs.begin(), inputs.begin() + train_end);
        std::vector<Tensor<T>> train_targets(targets.begin(), targets.begin() + train_end);
        std::vector<Tensor<T>> val_inputs(inputs.begin() + train_end, inputs.begin() + val_end);
        std::vector<Tensor<T>> val_targets(targets.begin() + train_end, targets.begin() + val_end);

        // Apply normalization to training data
        normalize_data(train_inputs, train_targets);
        normalize_data(val_inputs, val_targets);

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
     * Train with 3-phase approach:
     * Phase 1: Exploration (10 epochs) - High LR, aggressive architecture changes
     * Phase 2: Estimation (10 epochs) - Medium LR, estimate epochs needed
     * Phase 3: Main training - Adaptive LR, perturbation, adaptive thresholds
     */
    TrainingResult train_phased(const std::vector<Tensor<T>>& inputs,
                                const std::vector<Tensor<T>>& targets) {
        auto total_start = std::chrono::high_resolution_clock::now();

        TrainingResult result;
        result.cost_history.reserve(500);
        result.efficiency_history.reserve(500);
        result.cancer_score_history.reserve(500);
        result.alzheimer_score_history.reserve(500);
        result.architecture_history.reserve(500);

        // Compute normalization parameters and normalize data
        compute_normalization_params(inputs, targets);
        std::vector<Tensor<T>> norm_inputs = inputs;  // Copy for normalization
        std::vector<Tensor<T>> norm_targets = targets;
        normalize_data(norm_inputs, norm_targets);

        // ============ PHASE 1: EXPLORATION (10 epochs) ============
        auto phase1_start = std::chrono::high_resolution_clock::now();
        double exploration_lr = 0.5;  // High LR for wide exploration

        for (size_t epoch = 0; epoch < 10; ++epoch) {
            health_monitor_.update_epoch(epoch);

            double epoch_cost = train_epoch_with_lr(norm_inputs, norm_targets, exploration_lr);
            result.cost_history.push_back(epoch_cost);

            // Aggressive architecture exploration with low threshold
            auto decision = layer_manager_.analyze_with_efficiency(0.3);  // Low threshold
            if (decision.action != dynamics::LayerDecision::Action::None) {
                layer_manager_.execute(decision);
                if (decision.action == dynamics::LayerDecision::Action::AddNodes) {
                    result.nodes_added += decision.node_count;
                } else if (decision.action == dynamics::LayerDecision::Action::RemoveNodes) {
                    result.nodes_removed += decision.nodes_to_remove.size();
                } else if (decision.action == dynamics::LayerDecision::Action::AddLayer) {
                    result.layers_added++;
                } else if (decision.action == dynamics::LayerDecision::Action::RemoveLayer) {
                    result.layers_removed++;
                }
            }

            // Track metrics
            auto health = health_monitor_.diagnose();
            result.cancer_score_history.push_back(health.cancer_score);
            result.alzheimer_score_history.push_back(health.alzheimer_score);
            result.architecture_history.emplace_back(network_.num_layers(), network_.num_nodes());

            double efficiency = compute_efficiency(result.cost_history);
            result.efficiency_history.push_back(efficiency);

            if (epoch_cost < result.best_cost) {
                result.best_cost = epoch_cost;
            }
        }

        auto phase1_end = std::chrono::high_resolution_clock::now();
        result.phase1_time = std::chrono::duration<double>(phase1_end - phase1_start).count();

        // ============ PHASE 2: ESTIMATION (10 epochs) ============
        auto phase2_start = std::chrono::high_resolution_clock::now();
        double estimation_lr = 0.1;  // Medium LR
        std::vector<double> estimation_costs;

        for (size_t epoch = 0; epoch < 10; ++epoch) {
            health_monitor_.update_epoch(10 + epoch);

            double epoch_cost = train_epoch_with_lr(norm_inputs, norm_targets, estimation_lr);
            estimation_costs.push_back(epoch_cost);
            result.cost_history.push_back(epoch_cost);

            auto health = health_monitor_.diagnose();
            result.cancer_score_history.push_back(health.cancer_score);
            result.alzheimer_score_history.push_back(health.alzheimer_score);
            result.architecture_history.emplace_back(network_.num_layers(), network_.num_nodes());

            double efficiency = compute_efficiency(result.cost_history);
            result.efficiency_history.push_back(efficiency);
        }

        // Estimate epochs needed
        double avg_improvement = (estimation_costs.front() - estimation_costs.back()) / 10.0;
        double current_efficiency = result.efficiency_history.back();
        double efficiency_gap = 0.9 - current_efficiency;
        result.estimated_epochs = static_cast<size_t>(
            std::max(10.0, std::min(500.0, efficiency_gap / (avg_improvement * 0.1 + 1e-8)))
        );

        auto phase2_end = std::chrono::high_resolution_clock::now();
        result.phase2_time = std::chrono::duration<double>(phase2_end - phase2_start).count();

        // ============ PHASE 3: MAIN TRAINING ============
        auto phase3_start = std::chrono::high_resolution_clock::now();
        double main_lr = 0.1;
        size_t perturbation_cutoff = result.estimated_epochs / 5;  // First 20%

        for (size_t epoch = 0; epoch < result.estimated_epochs; ++epoch) {
            health_monitor_.update_epoch(20 + epoch);

            // Apply perturbation in first 20%
            if (epoch < perturbation_cutoff) {
                apply_random_perturbation(0.005);  // 0.5% of nodes
                result.perturbations_applied++;
            }

            // Compute adaptive saturation threshold
            double efficiency = compute_efficiency(result.cost_history);
            double sat_threshold = compute_sigmoid_threshold(efficiency);

            double epoch_cost = train_epoch_with_lr(norm_inputs, norm_targets, main_lr);
            result.cost_history.push_back(epoch_cost);

            // Layer adjustment with adaptive threshold
            auto decision = layer_manager_.analyze_with_efficiency(efficiency);
            if (decision.action != dynamics::LayerDecision::Action::None) {
                layer_manager_.execute(decision);
                if (decision.action == dynamics::LayerDecision::Action::AddNodes) {
                    result.nodes_added += decision.node_count;
                }
            }

            // Track metrics
            auto health = health_monitor_.diagnose();
            result.cancer_score_history.push_back(health.cancer_score);
            result.alzheimer_score_history.push_back(health.alzheimer_score);
            result.architecture_history.emplace_back(network_.num_layers(), network_.num_nodes());

            efficiency = compute_efficiency(result.cost_history);
            result.efficiency_history.push_back(efficiency);

            if (epoch_cost < result.best_cost) {
                result.best_cost = epoch_cost;
            }
            if (efficiency > result.best_efficiency) {
                result.best_efficiency = efficiency;
            }

            // Learning rate decay
            if (epoch > 0 && epoch % 20 == 0) {
                main_lr *= 0.8;
            }

            // Early stopping
            if (result.cost_history.size() > 20) {
                double recent_improvement = result.cost_history[result.cost_history.size() - 20] -
                                           result.cost_history.back();
                if (recent_improvement < 1e-6) {
                    result.stopping_reason = "Early stopping (no improvement)";
                    break;
                }
            }
        }

        auto phase3_end = std::chrono::high_resolution_clock::now();
        result.phase3_time = std::chrono::duration<double>(phase3_end - phase3_start).count();

        // ============ PHASE 4: STANDARD TRAINING ============
        if (config_.phase4_enabled) {
            auto phase4_start = std::chrono::high_resolution_clock::now();

            double phase3_final_cost = result.cost_history.back();
            size_t phase4_estimated = estimate_phase4_epochs(
                phase3_final_cost,
                result.cost_history
            );

            result.phase4_initial_cost = phase3_final_cost;
            result.phase4_estimated_epochs = phase4_estimated;

            double phase4_lr = config_.phase4_learning_rate;
            double phase4_best_cost = phase3_final_cost;
            size_t patience_counter = 0;

            for (size_t epoch = 0; epoch < phase4_estimated; ++epoch) {
                // Train one epoch - NO architecture modifications (frozen)
                double epoch_cost = train_epoch_with_lr(norm_inputs, norm_targets, phase4_lr);
                result.cost_history.push_back(epoch_cost);

                // Track metrics
                double efficiency = compute_efficiency(result.cost_history);
                result.efficiency_history.push_back(efficiency);

                auto health = health_monitor_.diagnose();
                result.cancer_score_history.push_back(health.cancer_score);
                result.alzheimer_score_history.push_back(health.alzheimer_score);
                result.architecture_history.emplace_back(network_.num_layers(), network_.num_nodes());

                // Track best cost
                if (epoch_cost < result.best_cost) {
                    result.best_cost = epoch_cost;
                }
                if (efficiency > result.best_efficiency) {
                    result.best_efficiency = efficiency;
                }

                // Track best cost and patience for early stopping
                if (epoch_cost < phase4_best_cost - config_.phase4_min_improvement) {
                    phase4_best_cost = epoch_cost;
                    patience_counter = 0;
                } else {
                    patience_counter++;
                }

                // Learning rate decay
                if (epoch > 0 && epoch % config_.phase4_lr_decay_interval == 0) {
                    phase4_lr = std::max(config_.phase4_min_learning_rate,
                                        phase4_lr * config_.phase4_lr_decay_rate);
                }

                // Callback
                if (epoch_callback_) {
                    epoch_callback_(20 + result.estimated_epochs + epoch,
                                   epoch_cost, efficiency, network_);
                }

                // Early stopping check
                if (patience_counter >= config_.phase4_patience) {
                    result.phase4_early_stopped = true;
                    result.stopping_reason = "Phase 4 early stopping (no improvement)";
                    result.phase4_epochs = epoch + 1;
                    break;
                }

                // Check if target reduction achieved
                double cost_reduction = (result.phase4_initial_cost - epoch_cost) /
                                       (result.phase4_initial_cost + 1e-8);
                if (cost_reduction >= config_.phase4_target_cost_reduction) {
                    result.stopping_reason = "Phase 4 target cost reduction achieved";
                    result.phase4_epochs = epoch + 1;
                    break;
                }

                result.phase4_epochs = epoch + 1;
            }

            auto phase4_end = std::chrono::high_resolution_clock::now();
            result.phase4_time = std::chrono::duration<double>(phase4_end - phase4_start).count();
            result.phase4_final_cost = result.cost_history.back();
            result.phase4_cost_reduction = (result.phase4_initial_cost - result.phase4_final_cost) /
                                          (result.phase4_initial_cost + 1e-8);
        }

        // Finalize result
        auto total_end = std::chrono::high_resolution_clock::now();
        result.training_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            total_end - total_start);
        result.success = true;
        result.epochs_completed = result.cost_history.size();
        result.final_cost = result.cost_history.empty() ? 0.0 : result.cost_history.back();
        result.final_efficiency = result.efficiency_history.empty() ? 0.5 :
                                  result.efficiency_history.back();

        if (result.stopping_reason.empty()) {
            result.stopping_reason = "Training completed";
        }

        return result;
    }

    /**
     * Train for a single epoch with specified learning rate.
     */
    double train_epoch_with_lr(const std::vector<Tensor<T>>& inputs,
                               const std::vector<Tensor<T>>& targets,
                               double lr) {
        double saved_lr = learning_rate_;
        learning_rate_ = lr;
        double cost = train_epoch(inputs, targets);
        learning_rate_ = saved_lr;
        return cost;
    }

    /**
     * Compute efficiency from cost history.
     */
    double compute_efficiency(const std::vector<double>& cost_history) const {
        if (cost_history.size() < 2) return 0.5;
        double improvement = (cost_history[cost_history.size() - 2] - cost_history.back()) /
                            (cost_history[cost_history.size() - 2] + 1e-8);
        return std::min(1.0, std::max(0.0, 0.5 + improvement * 10.0));
    }

    /**
     * Compute sigmoid-based adaptive saturation threshold.
     */
    double compute_sigmoid_threshold(double efficiency, double k = 5.0,
                                     double base = 0.3, double range = 0.5) const {
        double sigmoid = 1.0 / (1.0 + std::exp(-k * (efficiency - 0.5)));
        return base + range * sigmoid;
    }

    /**
     * Apply random perturbation to fraction of nodes.
     */
    void apply_random_perturbation(double fraction) {
        // This is a placeholder - full implementation would modify network weights
        // The actual perturbation happens at the network level
        size_t total_nodes = network_.num_nodes();
        size_t num_to_perturb = std::max(size_t(1), static_cast<size_t>(total_nodes * fraction));
        // Network-level perturbation would be implemented here
    }

    /**
     * Estimate epochs needed for Phase 4 based on cost reduction rate.
     *
     * Uses the cost reduction rate from recent training history and applies
     * a diminishing returns factor since Phase 4 improvement is typically slower.
     */
    size_t estimate_phase4_epochs(double phase3_final_cost,
                                  const std::vector<double>& cost_history) const {
        size_t window_size = std::min(size_t(20), cost_history.size());
        if (window_size < 2) {
            return config_.phase4_min_epochs;
        }

        size_t start_idx = cost_history.size() - window_size;
        double total_reduction = cost_history[start_idx] - cost_history.back();
        double reduction_per_epoch = total_reduction / (window_size - 1);

        // Diminishing returns factor (Phase 4 improvement is slower)
        double diminishing_factor = 0.3;
        double effective_rate = reduction_per_epoch * diminishing_factor;

        if (effective_rate <= 0) {
            // No improvement or worsening - use minimum epochs
            return config_.phase4_min_epochs;
        }

        // Target cost after Phase 4
        double target_cost = phase3_final_cost * (1.0 - config_.phase4_target_cost_reduction);
        double remaining_reduction = phase3_final_cost - target_cost;

        // Estimate epochs needed
        size_t estimated = static_cast<size_t>(
            remaining_reduction / (effective_rate + 1e-8)
        );

        // Clamp to configured bounds
        return std::max(config_.phase4_min_epochs,
                       std::min(config_.phase4_max_epochs, estimated));
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

    /**
     * Get normalization parameters (for denormalization in prediction).
     */
    const NormalizationParams<T>& normalization_params() const { return norm_params_; }

    /**
     * Normalize input tensor using stored parameters.
     */
    Tensor<T> normalize_input(const Tensor<T>& input) const {
        if (!norm_params_.input_normalized || norm_params_.input_mean.empty()) {
            return input;
        }

        Tensor<T> result = input.clone();
        T* data = result.data();
        size_t size = result.size();

        if (norm_params_.method == NormalizationMethod::ZScore) {
            for (size_t i = 0; i < size; ++i) {
                size_t idx = i % norm_params_.input_mean.size();
                data[i] = (data[i] - norm_params_.input_mean[idx]) / norm_params_.input_std[idx];
            }
        } else {  // MinMax
            for (size_t i = 0; i < size; ++i) {
                size_t idx = i % norm_params_.input_min.size();
                T range = norm_params_.input_max[idx] - norm_params_.input_min[idx] + norm_params_.epsilon;
                data[i] = (data[i] - norm_params_.input_min[idx]) / range;
            }
        }

        return result;
    }

    /**
     * Denormalize output tensor to original scale.
     */
    Tensor<T> denormalize_output(const Tensor<T>& output) const {
        if (!norm_params_.output_normalized || norm_params_.output_mean.empty()) {
            return output;
        }

        Tensor<T> result = output.clone();
        T* data = result.data();
        size_t size = result.size();

        if (norm_params_.method == NormalizationMethod::ZScore) {
            for (size_t i = 0; i < size; ++i) {
                size_t idx = i % norm_params_.output_mean.size();
                data[i] = data[i] * norm_params_.output_std[idx] + norm_params_.output_mean[idx];
            }
        } else {  // MinMax
            for (size_t i = 0; i < size; ++i) {
                size_t idx = i % norm_params_.output_min.size();
                T range = norm_params_.output_max[idx] - norm_params_.output_min[idx] + norm_params_.epsilon;
                data[i] = data[i] * range + norm_params_.output_min[idx];
            }
        }

        return result;
    }

private:
    void clip_gradients() {
        // Simple gradient clipping by value
        // In a full implementation, we'd access the accumulated gradients
        // For now, this is a placeholder
    }

    /**
     * Compute and store normalization parameters from training data.
     */
    void compute_normalization_params(const std::vector<Tensor<T>>& inputs,
                                      const std::vector<Tensor<T>>& targets) {
        if (inputs.empty()) return;

        size_t input_size = inputs[0].size();
        size_t output_size = targets[0].size();
        size_t n_samples = inputs.size();

        // Compute input normalization params
        if (config_.normalization.normalize_input) {
            norm_params_.input_mean.resize(input_size, T(0));
            norm_params_.input_std.resize(input_size, T(0));
            norm_params_.input_min.resize(input_size, std::numeric_limits<T>::max());
            norm_params_.input_max.resize(input_size, std::numeric_limits<T>::lowest());

            // Compute mean, min, max
            for (const auto& input : inputs) {
                const T* data = input.data();
                for (size_t i = 0; i < input_size; ++i) {
                    norm_params_.input_mean[i] += data[i];
                    norm_params_.input_min[i] = std::min(norm_params_.input_min[i], data[i]);
                    norm_params_.input_max[i] = std::max(norm_params_.input_max[i], data[i]);
                }
            }
            for (size_t i = 0; i < input_size; ++i) {
                norm_params_.input_mean[i] /= n_samples;
            }

            // Compute std
            for (const auto& input : inputs) {
                const T* data = input.data();
                for (size_t i = 0; i < input_size; ++i) {
                    T diff = data[i] - norm_params_.input_mean[i];
                    norm_params_.input_std[i] += diff * diff;
                }
            }
            for (size_t i = 0; i < input_size; ++i) {
                norm_params_.input_std[i] = std::sqrt(norm_params_.input_std[i] / n_samples) + norm_params_.epsilon;
            }

            norm_params_.input_normalized = true;
        }

        // Compute output normalization params (only for regression)
        bool is_regression = (cost_type_ == CostFunctionType::MeanSquaredError ||
                             cost_type_ == CostFunctionType::MeanAbsoluteError ||
                             cost_type_ == CostFunctionType::Huber ||
                             cost_type_ == CostFunctionType::LogCosh);

        if (config_.normalization.normalize_output && is_regression) {
            norm_params_.output_mean.resize(output_size, T(0));
            norm_params_.output_std.resize(output_size, T(0));
            norm_params_.output_min.resize(output_size, std::numeric_limits<T>::max());
            norm_params_.output_max.resize(output_size, std::numeric_limits<T>::lowest());

            // Compute mean, min, max
            for (const auto& target : targets) {
                const T* data = target.data();
                for (size_t i = 0; i < output_size; ++i) {
                    norm_params_.output_mean[i] += data[i];
                    norm_params_.output_min[i] = std::min(norm_params_.output_min[i], data[i]);
                    norm_params_.output_max[i] = std::max(norm_params_.output_max[i], data[i]);
                }
            }
            for (size_t i = 0; i < output_size; ++i) {
                norm_params_.output_mean[i] /= n_samples;
            }

            // Compute std
            for (const auto& target : targets) {
                const T* data = target.data();
                for (size_t i = 0; i < output_size; ++i) {
                    T diff = data[i] - norm_params_.output_mean[i];
                    norm_params_.output_std[i] += diff * diff;
                }
            }
            for (size_t i = 0; i < output_size; ++i) {
                norm_params_.output_std[i] = std::sqrt(norm_params_.output_std[i] / n_samples) + norm_params_.epsilon;
            }

            norm_params_.output_normalized = true;
        }
    }

    /**
     * Apply normalization to training data in-place.
     */
    void normalize_data(std::vector<Tensor<T>>& inputs,
                       std::vector<Tensor<T>>& targets) {
        // Normalize inputs
        if (norm_params_.input_normalized) {
            for (auto& input : inputs) {
                T* data = input.data();
                size_t size = input.size();

                if (norm_params_.method == NormalizationMethod::ZScore) {
                    for (size_t i = 0; i < size; ++i) {
                        size_t idx = i % norm_params_.input_mean.size();
                        data[i] = (data[i] - norm_params_.input_mean[idx]) / norm_params_.input_std[idx];
                    }
                } else {  // MinMax
                    for (size_t i = 0; i < size; ++i) {
                        size_t idx = i % norm_params_.input_min.size();
                        T range = norm_params_.input_max[idx] - norm_params_.input_min[idx] + norm_params_.epsilon;
                        data[i] = (data[i] - norm_params_.input_min[idx]) / range;
                    }
                }
            }
        }

        // Normalize targets
        if (norm_params_.output_normalized) {
            for (auto& target : targets) {
                T* data = target.data();
                size_t size = target.size();

                if (norm_params_.method == NormalizationMethod::ZScore) {
                    for (size_t i = 0; i < size; ++i) {
                        size_t idx = i % norm_params_.output_mean.size();
                        data[i] = (data[i] - norm_params_.output_mean[idx]) / norm_params_.output_std[idx];
                    }
                } else {  // MinMax
                    for (size_t i = 0; i < size; ++i) {
                        size_t idx = i % norm_params_.output_min.size();
                        T range = norm_params_.output_max[idx] - norm_params_.output_min[idx] + norm_params_.epsilon;
                        data[i] = (data[i] - norm_params_.output_min[idx]) / range;
                    }
                }
            }
        }
    }

    Network<T>& network_;
    TrainerConfig config_;
    CostFunctionType cost_type_;
    std::unique_ptr<CostFunction<T>> cost_function_;
    BatchManager<T> batch_manager_;
    HealthMonitor<T> health_monitor_;
    LayerManager<T> layer_manager_;
    TrainableScheduler<T> trainable_scheduler_;
    EarlyStopping<T> early_stopping_;
    double learning_rate_;
    EpochCallback<T> epoch_callback_;
    NormalizationParams<T> norm_params_;
};

} // namespace training
} // namespace dnn
