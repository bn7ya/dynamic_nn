#pragma once

#include "../core/network.hpp"
#include "../core/tensor.hpp"
#include "cost_functions.hpp"
#include "batch_manager.hpp"
#include "early_stopping.hpp"
#include "emotional_state.hpp"
#include "../dynamics/layer_manager.hpp"
#include "../dynamics/health_monitor.hpp"
#include "../dynamics/trainable_scheduler.hpp"
#include "runtime/adaptive_config.hpp"
#include "runtime/metrics_bus.hpp"
#include "runtime/stage_controller.hpp"
#include "runtime/topology_lock.hpp"
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

    // Emotional learning fields (for Python parity)
    int total_rewards = 0;
    int total_penalties = 0;
    std::vector<double> depression_history;
    std::vector<double> excitement_history;
    std::vector<double> learning_rate_history;
    std::vector<std::string> emotional_state_history;
    int lr_reset_count = 0;
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
    size_t min_epochs_for_early_stop = 10;        // Was hardcoded as the 3rd EarlyStopping ctor arg

    // Batch management
    BatchConfig batch_config;

    // Layer-mutation thresholds (previously hardcoded inside LayerManager)
    dynamics::LayerManagerConfig layer_manager_config;

    // Reward/penalty machinery for Phase 3 (previously constructed locally)
    RewardPenaltyConfig reward_penalty_config;

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

    // Concurrent runtime (StageController + MetricsBus + observer thread).
    // When true, train_phased() routes the four stages through the
    // controller-driven pipeline in include/dnn/training/runtime/. The
    // controller publishes per-epoch metrics, runs an Estimation observer
    // in parallel with the active training stage, and can raise a rewind
    // signal so the controller wakes an earlier stage from its preserved
    // state. When false (default), the legacy strictly-sequential body
    // runs unchanged. This flag exists so the new path is opt-in until
    // it has stabilised.
    bool runtime_enabled = false;

    // Batch the per-sample forward/backward inside train_epoch into a single
    // rank-2 forward/backward. The rank-2 path is already implemented in
    // Layer::forward/backward (CPU + CUDA), it just wasn't reached from the
    // trainer. On CUDA this collapses B per-sample CPU<->GPU round-trips
    // (one cuda_gemv per sample per layer) into a single cuda_gemm per layer
    // per batch — typically the difference between GPU being slower than CPU
    // and being meaningfully faster. On CPU it's numerically equivalent
    // (modulo floating-point summation order).
    //
    // Default true. Set false to fall back to the legacy per-sample loop
    // for benchmarking or to bisect a regression.
    bool batched_train_forward = true;
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
    /**
     * RAII guard that flips Network<T>::training_in_progress_ to true
     * for the duration of a train_phased{,_runtime} call. Ensures
     * compact() refuses to run mid-training even if the call exits via
     * exception.
     */
    struct TrainingInProgressGuard {
        Network<T>& net;
        explicit TrainingInProgressGuard(Network<T>& n) : net(n) {
            net.set_training_in_progress(true);
        }
        ~TrainingInProgressGuard() { net.set_training_in_progress(false); }
        TrainingInProgressGuard(const TrainingInProgressGuard&) = delete;
        TrainingInProgressGuard& operator=(const TrainingInProgressGuard&) = delete;
    };

    Trainer(Network<T>& network,
            CostFunctionType cost_type = CostFunctionType::MeanSquaredError,
            const TrainerConfig& config = TrainerConfig())
        : network_(network)
        , config_(config)
        , cost_type_(cost_type)
        , cost_function_(CostFunction<T>::create(cost_type))
        , batch_manager_(config.batch_config)
        , health_monitor_(network, config.cancer_threshold, config.alzheimer_threshold)
        , layer_manager_(network, health_monitor_, config.layer_manager_config)
        , trainable_scheduler_(network, config.trainable_config)
        , early_stopping_(config.patience, config.min_improvement,
                          config.min_epochs_for_early_stop)
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
        TrainingInProgressGuard tip_guard(network_);

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
     * Train via the four-stage pipeline.
     *
     * Conceptually: Exploration → Estimation → Main → Standard. With
     * TrainerConfig::runtime_enabled == true the call routes to
     * train_phased_runtime(), which drives the four stages through a
     * StageController. The Estimation **observer** thread runs in
     * parallel for the lifetime of the call, drains the MetricsBus, and
     * can raise rewind_requested() when Main's cost-trend stalls -- the
     * controller responds with a brief Estimation refresh and re-enters
     * Main with the preserved emotional state. Per-stage state (LR,
     * emotional counters, epoch index) is preserved across suspends so
     * rewinds don't lose learning progress.
     *
     * Topology mutations that previously hard-erased weights now go
     * through Layer<T>::remove_nodes / Network<T>::mark_layer_inactive
     * (soft); physical erasure runs only in Network<T>::compact() at
     * end-of-training. Mutations take TopologyLock in write mode; the
     * per-epoch step body holds it in read mode via run_stage().
     *
     * With runtime_enabled == false (the default) the legacy strictly
     * sequential implementation runs unchanged.
     */
    TrainingResult train_phased(const std::vector<Tensor<T>>& inputs,
                                const std::vector<Tensor<T>>& targets) {
        if (config_.runtime_enabled) {
            return train_phased_runtime(inputs, targets);
        }

        TrainingInProgressGuard tip_guard(network_);

        auto total_start = std::chrono::high_resolution_clock::now();

        TrainingResult result;
        reserve_histories(result, kDefaultHistoryReserve);

        // Compute normalization parameters and normalize data
        compute_normalization_params(inputs, targets);
        std::vector<Tensor<T>> norm_inputs = inputs;  // Copy for normalization
        std::vector<Tensor<T>> norm_targets = targets;
        normalize_data(norm_inputs, norm_targets);

        // ============ PRE-TRAINING: ADAPTIVE WEIGHT INITIALIZATION ============
        // Run a forward pass to collect initial activation statistics
        for (size_t i = 0; i < std::min(size_t(100), norm_inputs.size()); ++i) {
            network_.forward(norm_inputs[i]);
        }
        // Compute variance z-scores and adapt variance weights for all layers
        for (size_t l = 0; l < network_.num_layers(); ++l) {
            network_.layer(l).compute_initial_variance_zscores();
        }
        // Reset metrics after initial pass
        for (size_t l = 0; l < network_.num_layers(); ++l) {
            network_.layer(l).reset_node_metrics();
        }

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

        // ============ POST-PHASE 1: COMPUTE GRADIENT THRESHOLDS ============
        // Auto-detect gradient thresholds based on Phase 1 statistics
        for (size_t l = 0; l < network_.num_layers(); ++l) {
            double grad_threshold = network_.layer(l).compute_gradient_threshold();
            network_.layer(l).set_nodes_grad_threshold(grad_threshold);
        }

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

            // Adapt efficiency weights based on gradient statistics
            for (size_t l = 0; l < network_.num_layers(); ++l) {
                network_.layer(l).adapt_node_weights();
            }
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

        // Initialize emotional learning state
        EmotionalState emotional_state;
        RewardPenaltyConfig reward_penalty_config = config_.reward_penalty_config;
        reward_penalty_config.baseline_learning_rate = main_lr;

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

            // Apply reward/penalty system (matches Python implementation)
            auto [new_lr, action] = apply_reward_penalty_system(
                main_lr, result.cost_history, result.efficiency_history,
                reward_penalty_config, emotional_state);
            main_lr = new_lr;

            // Track emotional learning metrics
            result.learning_rate_history.push_back(main_lr);
            result.emotional_state_history.push_back(action);

            // Adapt efficiency weights based on gradient statistics
            for (size_t l = 0; l < network_.num_layers(); ++l) {
                network_.layer(l).adapt_node_weights();
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

        // Copy emotional state to result
        result.total_rewards = emotional_state.total_rewards;
        result.total_penalties = emotional_state.total_penalties;
        result.lr_reset_count = emotional_state.lr_reset_count;
        result.depression_history.assign(
            emotional_state.depression_history.begin(),
            emotional_state.depression_history.end());
        result.excitement_history.assign(
            emotional_state.excitement_history.begin(),
            emotional_state.excitement_history.end());

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

                // Adapt efficiency weights based on gradient statistics
                for (size_t l = 0; l < network_.num_layers(); ++l) {
                    network_.layer(l).adapt_node_weights();
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
     * Concurrent-pipeline implementation of train_phased(), routed to
     * when TrainerConfig::runtime_enabled == true.
     *
     * Each of the four stages runs as a callable handed to a
     * StageController. The Estimation **observer** thread runs in
     * parallel for the lifetime of the call: it drains the MetricsBus
     * and (a) nudges RuntimeAdaptiveConfig scalars, (b) raises
     * rewind_requested() when convergence stalls. When the Main stage
     * sees a rewind it bails out of its inner loop; the controller
     * loops once back through Estimation to refresh estimated_epochs
     * and then re-enters Main with the preserved emotional state.
     *
     * Topology mutations during Main and Exploration take TopologyLock
     * in write mode; the per-epoch step body holds it in read mode (via
     * StageController::run_stage). Combined with Layer<T>'s soft-delete
     * semantics this means in-flight workers never observe a half-applied
     * topology change.
     */
    TrainingResult train_phased_runtime(const std::vector<Tensor<T>>& inputs,
                                        const std::vector<Tensor<T>>& targets) {
        namespace rt = dnn::training::runtime;

        TrainingInProgressGuard tip_guard(network_);

        auto total_start = std::chrono::high_resolution_clock::now();

        TrainingResult result;
        reserve_histories(result, std::max(kDefaultHistoryReserve,
                                           static_cast<size_t>(config_.max_epochs)));

        // Normalize data (same as legacy).
        compute_normalization_params(inputs, targets);
        std::vector<Tensor<T>> norm_inputs = inputs;
        std::vector<Tensor<T>> norm_targets = targets;
        normalize_data(norm_inputs, norm_targets);

        // Pre-training adaptive weight initialisation (same as legacy).
        for (size_t i = 0; i < std::min(size_t(100), norm_inputs.size()); ++i) {
            network_.forward(norm_inputs[i]);
        }
        for (size_t l = 0; l < network_.num_layers(); ++l) {
            network_.layer(l).compute_initial_variance_zscores();
        }
        for (size_t l = 0; l < network_.num_layers(); ++l) {
            network_.layer(l).reset_node_metrics();
        }

        // Concurrent runtime substrate.
        rt::MetricsBus bus(2048);
        rt::TopologyLock topo_lock;
        rt::RuntimeAdaptiveConfig adaptive_cfg;
        // Seed adaptive scalars from caller-set TrainerConfig fields so
        // values derived by the dynamic-thresholds Python layer flow into
        // the runtime path. With unmodified TrainerConfig defaults this
        // produces the same scalar values as reset_to_defaults() did.
        adaptive_cfg.apply_static_config(config_);
        rt::StageController controller(bus, adaptive_cfg, topo_lock);
        controller.start_observer();

        // Helper that records per-epoch metrics into `result`. Common to
        // every stage so the result histories line up regardless of
        // which stage is active.
        auto record_epoch_metrics = [&](double epoch_cost) {
            auto health = health_monitor_.diagnose();
            result.cancer_score_history.push_back(health.cancer_score);
            result.alzheimer_score_history.push_back(health.alzheimer_score);
            result.architecture_history.emplace_back(network_.num_layers(),
                                                     network_.num_nodes());
            double efficiency = compute_efficiency(result.cost_history);
            result.efficiency_history.push_back(efficiency);
            if (epoch_cost < result.best_cost) result.best_cost = epoch_cost;
            if (efficiency > result.best_efficiency) result.best_efficiency = efficiency;
            return efficiency;
        };

        // ============ PHASE 1: EXPLORATION ============
        auto phase1_start = std::chrono::high_resolution_clock::now();
        const size_t phase1_epochs =
            static_cast<size_t>(adaptive_cfg.exploration_epochs.current());

        controller.run_stage(
            rt::StageId::Exploration,
            [&](uint64_t epoch_in_stage) {
                health_monitor_.update_epoch(epoch_in_stage);
                double lr = adaptive_cfg.exploration_lr.current();
                double epoch_cost = train_epoch_with_lr(norm_inputs, norm_targets, lr);
                result.cost_history.push_back(epoch_cost);

                double sat = adaptive_cfg.exploration_saturation_threshold.current();
                auto decision = layer_manager_.analyze_with_efficiency(sat);
                if (decision.action != dynamics::LayerDecision::Action::None) {
                    auto wlock = topo_lock.write_lock();
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

                double efficiency = record_epoch_metrics(epoch_cost);
                controller.publish_metric(rt::StageId::Exploration,
                                          epoch_in_stage, epoch_cost, efficiency,
                                          lr, network_.topology_version());
            },
            [] { return true; },
            phase1_epochs);

        result.phase1_time = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - phase1_start).count();

        // Auto-detect gradient thresholds from Phase 1 statistics.
        for (size_t l = 0; l < network_.num_layers(); ++l) {
            double t = network_.layer(l).compute_gradient_threshold();
            network_.layer(l).set_nodes_grad_threshold(t);
        }

        // ============ PHASE 2: ESTIMATION (observer also active) ============
        auto phase2_start = std::chrono::high_resolution_clock::now();
        const size_t phase2_epochs =
            static_cast<size_t>(adaptive_cfg.estimation_epochs.current());
        std::vector<double> estimation_costs;

        controller.run_stage(
            rt::StageId::Estimation,
            [&](uint64_t epoch_in_stage) {
                health_monitor_.update_epoch(phase1_epochs + epoch_in_stage);
                double lr = adaptive_cfg.estimation_lr.current();
                double epoch_cost = train_epoch_with_lr(norm_inputs, norm_targets, lr);
                estimation_costs.push_back(epoch_cost);
                result.cost_history.push_back(epoch_cost);
                double efficiency = record_epoch_metrics(epoch_cost);
                for (size_t l = 0; l < network_.num_layers(); ++l) {
                    network_.layer(l).adapt_node_weights();
                }
                controller.publish_metric(rt::StageId::Estimation,
                                          epoch_in_stage, epoch_cost, efficiency,
                                          lr, network_.topology_version());
            },
            [] { return true; },
            phase2_epochs);

        // Estimate epochs needed for Main.
        if (estimation_costs.size() >= 2) {
            double avg_improvement =
                (estimation_costs.front() - estimation_costs.back()) / phase2_epochs;
            double current_efficiency = result.efficiency_history.back();
            double gap = adaptive_cfg.target_efficiency.current() - current_efficiency;
            result.estimated_epochs = static_cast<size_t>(std::max(
                adaptive_cfg.main_epoch_min.current(),
                std::min(adaptive_cfg.main_epoch_max.current(),
                         gap / (avg_improvement * 0.1 + 1e-8))));
        } else {
            result.estimated_epochs =
                static_cast<size_t>(adaptive_cfg.main_epoch_min.current());
        }

        result.phase2_time = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - phase2_start).count();

        // ============ PHASE 3: MAIN (with rewind handling) ============
        auto phase3_start = std::chrono::high_resolution_clock::now();
        EmotionalState emotional_state;
        RewardPenaltyConfig rp_config = config_.reward_penalty_config;
        rp_config.baseline_learning_rate = adaptive_cfg.main_lr.current();
        rp_config.min_learning_rate = adaptive_cfg.reward_lr_floor.current();
        rp_config.max_learning_rate = adaptive_cfg.reward_lr_ceiling.current();
        double main_lr = adaptive_cfg.main_lr.current();
        size_t perturbation_cutoff = static_cast<size_t>(
            result.estimated_epochs * adaptive_cfg.perturbation_cutoff_ratio.current());

        // Allow at most one rewind to keep training bounded; observer can
        // raise rewind once during Main and the controller will re-run a
        // brief Estimation pass before re-entering Main.
        size_t rewinds_used = 0;
        const size_t max_rewinds = 1;
        bool keep_running = true;
        size_t main_epochs_remaining = result.estimated_epochs;

        while (keep_running && main_epochs_remaining > 0) {
            controller.clear_rewind();
            uint64_t executed = controller.run_stage(
                rt::StageId::Main,
                [&](uint64_t epoch_in_stage) {
                    health_monitor_.update_epoch(
                        phase1_epochs + phase2_epochs + epoch_in_stage);

                    if (epoch_in_stage < perturbation_cutoff) {
                        apply_random_perturbation(
                            adaptive_cfg.perturbation_fraction.current());
                        result.perturbations_applied++;
                    }

                    double efficiency_pre = compute_efficiency(result.cost_history);
                    double sat_threshold = compute_sigmoid_threshold(efficiency_pre);
                    (void)sat_threshold;  // currently used via layer_manager below

                    double epoch_cost =
                        train_epoch_with_lr(norm_inputs, norm_targets, main_lr);
                    result.cost_history.push_back(epoch_cost);

                    auto decision = layer_manager_.analyze_with_efficiency(efficiency_pre);
                    if (decision.action != dynamics::LayerDecision::Action::None) {
                        auto wlock = topo_lock.write_lock();
                        layer_manager_.execute(decision);
                        if (decision.action ==
                            dynamics::LayerDecision::Action::AddNodes) {
                            result.nodes_added += decision.node_count;
                        }
                    }

                    double efficiency = record_epoch_metrics(epoch_cost);

                    auto [new_lr, action] = apply_reward_penalty_system(
                        main_lr, result.cost_history, result.efficiency_history,
                        rp_config, emotional_state);
                    main_lr = new_lr;
                    result.learning_rate_history.push_back(main_lr);
                    result.emotional_state_history.push_back(action);

                    for (size_t l = 0; l < network_.num_layers(); ++l) {
                        network_.layer(l).adapt_node_weights();
                    }

                    controller.publish_metric(rt::StageId::Main,
                                              epoch_in_stage, epoch_cost,
                                              efficiency, main_lr,
                                              network_.topology_version());
                },
                [&] {
                    if (controller.rewind_requested()) return false;
                    size_t window =
                        static_cast<size_t>(adaptive_cfg.early_stop_window.current());
                    if (result.cost_history.size() > window) {
                        double recent =
                            result.cost_history[result.cost_history.size() - window]
                            - result.cost_history.back();
                        if (recent < 1e-6) {
                            result.stopping_reason =
                                "Early stopping (no improvement)";
                            return false;
                        }
                    }
                    return true;
                },
                main_epochs_remaining);

            main_epochs_remaining = (executed < main_epochs_remaining)
                                    ? main_epochs_remaining - executed
                                    : 0;

            if (controller.rewind_requested() && rewinds_used < max_rewinds) {
                ++rewinds_used;
                // Brief re-estimation pass: a few epochs of Estimation to
                // refresh the cost-trend signal, then back into Main with
                // preserved emotional state.
                controller.clear_rewind();
                size_t mini_phase2 = std::min<size_t>(phase2_epochs, 5);
                std::vector<double> rewind_costs;
                controller.run_stage(
                    rt::StageId::Estimation,
                    [&](uint64_t epoch_in_stage) {
                        double lr = adaptive_cfg.estimation_lr.current();
                        double c = train_epoch_with_lr(norm_inputs, norm_targets, lr);
                        rewind_costs.push_back(c);
                        result.cost_history.push_back(c);
                        double e = record_epoch_metrics(c);
                        controller.publish_metric(rt::StageId::Estimation,
                                                  epoch_in_stage, c, e,
                                                  lr, network_.topology_version());
                    },
                    [] { return true; },
                    mini_phase2);
                // Loop back into Main with whatever epochs remain.
            } else {
                keep_running = false;
            }
        }

        // Copy emotional state into result.
        result.total_rewards = emotional_state.total_rewards;
        result.total_penalties = emotional_state.total_penalties;
        result.lr_reset_count = emotional_state.lr_reset_count;
        result.depression_history.assign(emotional_state.depression_history.begin(),
                                         emotional_state.depression_history.end());
        result.excitement_history.assign(emotional_state.excitement_history.begin(),
                                         emotional_state.excitement_history.end());

        result.phase3_time = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - phase3_start).count();

        // ============ PHASE 4: STANDARD (frozen architecture) ============
        if (config_.phase4_enabled) {
            auto phase4_start = std::chrono::high_resolution_clock::now();
            double phase3_final_cost = result.cost_history.back();
            size_t phase4_estimated =
                estimate_phase4_epochs(phase3_final_cost, result.cost_history);
            result.phase4_initial_cost = phase3_final_cost;
            result.phase4_estimated_epochs = phase4_estimated;

            double phase4_lr = config_.phase4_learning_rate;
            double phase4_best_cost = phase3_final_cost;
            size_t patience_counter = 0;

            controller.run_stage(
                rt::StageId::Standard,
                [&](uint64_t epoch_in_stage) {
                    double epoch_cost =
                        train_epoch_with_lr(norm_inputs, norm_targets, phase4_lr);
                    result.cost_history.push_back(epoch_cost);
                    double efficiency = record_epoch_metrics(epoch_cost);

                    if (epoch_cost < phase4_best_cost - config_.phase4_min_improvement) {
                        phase4_best_cost = epoch_cost;
                        patience_counter = 0;
                    } else {
                        ++patience_counter;
                    }

                    if (epoch_in_stage > 0 &&
                        epoch_in_stage % config_.phase4_lr_decay_interval == 0) {
                        phase4_lr = std::max(config_.phase4_min_learning_rate,
                                             phase4_lr * config_.phase4_lr_decay_rate);
                    }

                    for (size_t l = 0; l < network_.num_layers(); ++l) {
                        network_.layer(l).adapt_node_weights();
                    }

                    controller.publish_metric(rt::StageId::Standard,
                                              epoch_in_stage, epoch_cost,
                                              efficiency, phase4_lr,
                                              network_.topology_version());
                    result.phase4_epochs = static_cast<size_t>(epoch_in_stage + 1);
                },
                [&] {
                    if (patience_counter >= config_.phase4_patience) {
                        result.phase4_early_stopped = true;
                        result.stopping_reason =
                            "Phase 4 early stopping (no improvement)";
                        return false;
                    }
                    if (!result.cost_history.empty()) {
                        double reduction = (result.phase4_initial_cost
                                            - result.cost_history.back())
                                           / (result.phase4_initial_cost + 1e-8);
                        if (reduction >= config_.phase4_target_cost_reduction) {
                            result.stopping_reason =
                                "Phase 4 target cost reduction achieved";
                            return false;
                        }
                    }
                    return true;
                },
                phase4_estimated);

            result.phase4_time = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - phase4_start).count();
            result.phase4_final_cost = result.cost_history.back();
            result.phase4_cost_reduction =
                (result.phase4_initial_cost - result.phase4_final_cost)
                / (result.phase4_initial_cost + 1e-8);
        }

        controller.stop_observer();

        auto total_end = std::chrono::high_resolution_clock::now();
        result.training_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
        result.success = true;
        result.epochs_completed = result.cost_history.size();
        result.final_cost =
            result.cost_history.empty() ? 0.0 : result.cost_history.back();
        result.final_efficiency =
            result.efficiency_history.empty() ? 0.5 : result.efficiency_history.back();
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

            const size_t B = batch_inputs.size();
            if (config_.batched_train_forward && B > 0) {
                // Batched path: one rank-2 forward + one rank-2 backward per
                // batch instead of B rank-1 calls. The rank-2 path is wired
                // in Layer::forward/backward for both CPU and CUDA.
                //
                // We deliberately call cost_function_->{compute,gradient}
                // per row (rank-1) rather than once on the full rank-2
                // tensor. Why: cost-function rank-2 gradients pre-divide by
                // batch_size (see CrossEntropyCost::gradient at
                // cost_functions.hpp:177-186), and the existing apply_gradients
                // call below already divides the LR by B. Using rank-2 cost
                // gradient would double-divide. Per-row cost is cheap (O(B *
                // output_dim)) compared to the per-layer matmul work.
                const size_t input_dim = batch_inputs[0].size();
                const size_t target_dim = batch_targets[0].size();

                Tensor<T> batch_input_2d(std::vector<size_t>{B, input_dim});
                for (size_t b = 0; b < B; ++b) {
                    for (size_t j = 0; j < input_dim; ++j) {
                        batch_input_2d.at(b, j) = batch_inputs[b][j];
                    }
                }

                auto batch_output_2d = network_.forward(batch_input_2d);
                if (batch_output_2d.rank() != 2) {
                    throw exceptions::ShapeException("train_epoch",
                        "batched forward did not return a rank-2 tensor");
                }
                const size_t output_dim = batch_output_2d.shape()[1];

                Tensor<T> batch_grad_2d(std::vector<size_t>{B, output_dim});
                Tensor<T> output_row(std::vector<size_t>{output_dim});
                Tensor<T> target_row(std::vector<size_t>{target_dim});
                for (size_t b = 0; b < B; ++b) {
                    for (size_t j = 0; j < output_dim; ++j) {
                        output_row[j] = batch_output_2d.at(b, j);
                    }
                    for (size_t j = 0; j < target_dim; ++j) {
                        target_row[j] = batch_targets[b][j];
                    }
                    T sample_cost = cost_function_->compute(output_row, target_row);
                    batch_cost += static_cast<double>(sample_cost);
                    auto grad_row = cost_function_->gradient(output_row, target_row);
                    for (size_t j = 0; j < output_dim; ++j) {
                        batch_grad_2d.at(b, j) = grad_row[j];
                    }
                }

                network_.backward(batch_grad_2d);
            } else {
                // Legacy per-sample path. Kept reachable via
                // config_.batched_train_forward = false for benchmarking
                // and regression bisection.
                for (size_t i = 0; i < B; ++i) {
                    auto output = network_.forward(batch_inputs[i]);
                    T sample_cost = cost_function_->compute(output, batch_targets[i]);
                    batch_cost += static_cast<double>(sample_cost);
                    auto grad = cost_function_->gradient(output, batch_targets[i]);
                    network_.backward(grad);
                }
            }

            batch_cost /= B;
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
    // Per-epoch history vectors are reserved up-front so they don't realloc
    // mid-training. 500 covers every default-config run; train_phased_runtime
    // bumps this to max(500, max_epochs) for opt-in long runs.
    static constexpr size_t kDefaultHistoryReserve = 500;

    static void reserve_histories(TrainingResult& result, size_t cap) {
        result.cost_history.reserve(cap);
        result.efficiency_history.reserve(cap);
        result.cancer_score_history.reserve(cap);
        result.alzheimer_score_history.reserve(cap);
        result.architecture_history.reserve(cap);
    }

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
                             cost_type_ == CostFunctionType::HuberLoss ||
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

// Out-of-line definition of RuntimeAdaptiveConfig::apply_static_config.
// Lives here so TrainerConfig is fully visible; the declaration sits in
// runtime/adaptive_config.hpp.
inline void runtime::RuntimeAdaptiveConfig::apply_static_config(
    const TrainerConfig& cfg) {
    reset_to_defaults();

    gradient_clip.set(cfg.gradient_clip_value);

    patience.set(static_cast<double>(cfg.patience));
    min_improvement.set(cfg.min_improvement);
    phase4_patience.set(static_cast<double>(cfg.phase4_patience));
    phase4_min_improvement.set(cfg.phase4_min_improvement);
    phase4_target_reduction.set(cfg.phase4_target_cost_reduction);

    min_batch_size.set(static_cast<double>(cfg.batch_config.min_batch_size));
    max_batch_size.set(static_cast<double>(cfg.batch_config.max_batch_size));
    batch_growth.set(cfg.batch_config.growth_rate);
    batch_growth_int.set(
        static_cast<double>(cfg.batch_config.growth_interval_epochs));

    cancer_threshold.set(cfg.cancer_threshold);
    alzheimer_threshold.set(cfg.alzheimer_threshold);

    const auto& rp = cfg.reward_penalty_config;
    cost_improvement_threshold.set(rp.cost_improvement_threshold);
    efficiency_improvement_threshold.set(rp.efficiency_improvement_threshold);
    reward_lr_floor.set(rp.min_learning_rate);
    reward_lr_ceiling.set(rp.max_learning_rate);
    reward_max_factor.set(rp.max_adjustment_factor);
    reward_min_factor.set(rp.min_adjustment_factor);
    extreme_threshold.set(rp.extreme_threshold);
    moderate_threshold.set(rp.moderate_threshold);
    emotion_window.set(static_cast<double>(rp.window_size));
}

} // namespace training
} // namespace dnn
