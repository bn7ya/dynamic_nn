#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <string>
#include <unordered_map>

namespace dnn {
namespace training {
namespace runtime {

/**
 * A scalar hyperparameter that adapts at runtime in response to training
 * feedback. The static defaults that previously sat in TrainerConfig,
 * EmotionalState, EarlyStopping etc. become AdaptiveScalar instances so the
 * StageController can nudge them per epoch from a single metrics bus.
 *
 * Updates are clamped to [min, max] and are atomic so multiple stage
 * workers can read the current value without locking. Writes are still
 * expected to come from a single controller thread.
 */
class AdaptiveScalar {
public:
    AdaptiveScalar() = default;
    AdaptiveScalar(double initial, double min_v, double max_v)
        : current_(initial), min_(min_v), max_(max_v) {}

    double current() const { return current_.load(std::memory_order_relaxed); }
    double min() const { return min_; }
    double max() const { return max_; }

    void set(double v) {
        current_.store(clamp(v), std::memory_order_relaxed);
    }

    /**
     * Multiplicative update: current *= factor, clamped to [min, max].
     * Useful for LR-style decay/boost driven by reward/penalty signals.
     */
    void scale(double factor) {
        double v = current_.load(std::memory_order_relaxed) * factor;
        current_.store(clamp(v), std::memory_order_relaxed);
    }

    /**
     * Additive nudge: current += delta, clamped to [min, max].
     * Useful for thresholds and integer-like fields (window sizes, patience).
     */
    void nudge(double delta) {
        double v = current_.load(std::memory_order_relaxed) + delta;
        current_.store(clamp(v), std::memory_order_relaxed);
    }

    operator double() const { return current(); }

private:
    double clamp(double v) const {
        return std::min(std::max(v, min_), max_);
    }

    std::atomic<double> current_{0.0};
    double min_{0.0};
    double max_{0.0};
};

/**
 * Aggregate of all formerly-static learning hyperparameters that the
 * StageController is allowed to retune at runtime. Defaults match the
 * historical hardcoded values from the sequential trainer so existing
 * behaviour is preserved when nothing tunes anything.
 *
 * Replaces the constants previously inlined in:
 *   include/dnn/training/optimizer.hpp:33-47
 *   include/dnn/training/trainer.hpp:119-164,358,407,444,457
 *   include/dnn/training/early_stopping.hpp:51-53,78
 *   include/dnn/training/batch_manager.hpp:19-22
 *   include/dnn/training/emotional_state.hpp:65-82
 */
struct RuntimeAdaptiveConfig {
    // Optimizer hyperparameters
    AdaptiveScalar learning_rate{0.01,  1e-6, 1.0};
    AdaptiveScalar momentum     {0.9,   0.0,  0.999};
    AdaptiveScalar beta1        {0.9,   0.0,  0.999};
    AdaptiveScalar beta2        {0.999, 0.0,  0.9999};
    AdaptiveScalar epsilon      {1e-8,  1e-12, 1e-3};
    AdaptiveScalar gradient_clip{1.0,   0.1,  10.0};

    // Per-stage learning rates (formerly hardcoded magic numbers in
    // trainer.hpp's train_phased loops).
    AdaptiveScalar exploration_lr{0.5,  1e-4, 1.0};
    AdaptiveScalar estimation_lr {0.1,  1e-4, 0.5};
    AdaptiveScalar main_lr       {0.1,  1e-5, 0.5};
    AdaptiveScalar standard_lr   {0.005, 1e-6, 0.05};

    // Stage durations (originally hardcoded as `for (epoch < 10)` etc.).
    AdaptiveScalar exploration_epochs{10.0, 1.0,  100.0};
    AdaptiveScalar estimation_epochs {10.0, 1.0,  100.0};
    AdaptiveScalar main_epoch_min    {10.0, 1.0,  100.0};
    AdaptiveScalar main_epoch_max    {500.0, 50.0, 5000.0};
    AdaptiveScalar standard_epoch_min{50.0, 1.0,  500.0};
    AdaptiveScalar standard_epoch_max{200.0, 50.0, 2000.0};

    // Architecture-mutation thresholds.
    AdaptiveScalar exploration_saturation_threshold{0.3, 0.05, 0.95};
    AdaptiveScalar target_efficiency               {0.9, 0.1,  0.99};
    AdaptiveScalar perturbation_cutoff_ratio       {0.2, 0.0,  0.5};
    AdaptiveScalar perturbation_fraction           {0.005, 0.0, 0.1};

    // Early stopping.
    AdaptiveScalar patience              {30.0, 1.0,  500.0};
    AdaptiveScalar min_improvement       {1e-4, 1e-9, 1e-1};
    AdaptiveScalar early_stop_window     {20.0, 2.0,  200.0};
    AdaptiveScalar phase4_patience       {35.0, 1.0,  500.0};
    AdaptiveScalar phase4_min_improvement{1e-4, 1e-9, 1e-1};
    AdaptiveScalar phase4_target_reduction{0.5, 0.0,  0.99};

    // Batch sizing.
    AdaptiveScalar min_batch_size  {4.0,    1.0, 1024.0};
    AdaptiveScalar max_batch_size  {256.0,  4.0, 8192.0};
    AdaptiveScalar batch_growth    {1.5,    1.0, 4.0};
    AdaptiveScalar batch_growth_int{5.0,    1.0, 100.0};

    // Reward / penalty / emotional system.
    AdaptiveScalar cost_improvement_threshold      {0.005, 1e-6, 0.5};
    AdaptiveScalar efficiency_improvement_threshold{0.02,  1e-6, 0.5};
    AdaptiveScalar reward_lr_floor    {1e-5, 1e-9, 1e-2};
    AdaptiveScalar reward_lr_ceiling  {0.5,  1e-3, 1.0};
    AdaptiveScalar reward_max_factor  {1.5,  1.01, 5.0};
    AdaptiveScalar reward_min_factor  {0.7,  0.1,  0.99};
    AdaptiveScalar extreme_threshold  {0.7,  0.5,  0.99};
    AdaptiveScalar moderate_threshold {0.6,  0.3,  0.95};
    AdaptiveScalar emotion_window     {15.0, 2.0,  500.0};

    // Health (cancer / alzheimer) thresholds.
    AdaptiveScalar cancer_threshold   {0.7, 0.1,  0.99};
    AdaptiveScalar alzheimer_threshold{0.7, 0.1,  0.99};

    /**
     * Reset every scalar to its default value. Used when the controller
     * decides to fully restart a stage rather than nudge it.
     *
     * Implemented field-by-field because AdaptiveScalar holds an
     * std::atomic and is intentionally non-copyable.
     */
    void reset_to_defaults() {
        learning_rate.set(0.01);
        momentum.set(0.9);
        beta1.set(0.9);
        beta2.set(0.999);
        epsilon.set(1e-8);
        gradient_clip.set(1.0);

        exploration_lr.set(0.5);
        estimation_lr.set(0.1);
        main_lr.set(0.1);
        standard_lr.set(0.005);

        exploration_epochs.set(10.0);
        estimation_epochs.set(10.0);
        main_epoch_min.set(10.0);
        main_epoch_max.set(500.0);
        standard_epoch_min.set(50.0);
        standard_epoch_max.set(200.0);

        exploration_saturation_threshold.set(0.3);
        target_efficiency.set(0.9);
        perturbation_cutoff_ratio.set(0.2);
        perturbation_fraction.set(0.005);

        patience.set(30.0);
        min_improvement.set(1e-4);
        early_stop_window.set(20.0);
        phase4_patience.set(35.0);
        phase4_min_improvement.set(1e-4);
        phase4_target_reduction.set(0.5);

        min_batch_size.set(4.0);
        max_batch_size.set(256.0);
        batch_growth.set(1.5);
        batch_growth_int.set(5.0);

        cost_improvement_threshold.set(0.005);
        efficiency_improvement_threshold.set(0.02);
        reward_lr_floor.set(1e-5);
        reward_lr_ceiling.set(0.5);
        reward_max_factor.set(1.5);
        reward_min_factor.set(0.7);
        extreme_threshold.set(0.7);
        moderate_threshold.set(0.6);
        emotion_window.set(15.0);

        cancer_threshold.set(0.7);
        alzheimer_threshold.set(0.7);
    }
};

}  // namespace runtime
}  // namespace training
}  // namespace dnn
