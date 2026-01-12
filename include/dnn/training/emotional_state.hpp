#pragma once

#include <deque>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <utility>

namespace dnn {
namespace training {

/**
 * Emotional state tracking for reward/penalty system.
 * Tracks training progress and adapts learning rate based on "emotional" metrics.
 * Matches Python implementation exactly.
 */
struct EmotionalState {
    int total_rewards = 0;
    int total_penalties = 0;
    std::deque<double> reward_history;
    std::deque<double> penalty_history;
    std::deque<double> depression_history;
    std::deque<double> excitement_history;
    int lr_reset_count = 0;

    /**
     * Depression ratio: proportion of penalties to total.
     * High depression indicates consistently poor progress.
     */
    double depression_ratio() const {
        int total = total_rewards + total_penalties;
        return total > 0 ? static_cast<double>(total_penalties) / total : 0.0;
    }

    /**
     * Excitement ratio: proportion of rewards to total.
     * High excitement indicates consistently good progress.
     */
    double excitement_ratio() const {
        int total = total_rewards + total_penalties;
        return total > 0 ? static_cast<double>(total_rewards) / total : 0.0;
    }

    /**
     * Reset emotional state to initial values.
     */
    void reset() {
        total_rewards = 0;
        total_penalties = 0;
        reward_history.clear();
        penalty_history.clear();
        depression_history.clear();
        excitement_history.clear();
        lr_reset_count = 0;
    }
};

/**
 * Configuration for reward/penalty system.
 * All values match Python defaults from network.py lines 200-212.
 */
struct RewardPenaltyConfig {
    // Improvement thresholds
    double cost_improvement_threshold = 0.005;       // 0.5% improvement
    double efficiency_improvement_threshold = 0.02;  // 2% efficiency improvement

    // Learning rate bounds
    double min_learning_rate = 1e-5;
    double max_learning_rate = 0.5;
    double baseline_learning_rate = 0.1;

    // Adjustment factors
    double max_adjustment_factor = 1.5;  // Maximum LR increase (penalty)
    double min_adjustment_factor = 0.7;  // Minimum LR decrease (reward)

    // Emotional state thresholds
    double extreme_threshold = 0.7;   // Full reset if depression/excitement > 70%
    double moderate_threshold = 0.6;  // Partial correction if > 60%

    // History window size
    size_t window_size = 15;
};

/**
 * Improvement metrics computed from training history.
 * Matches Python _compute_improvement_metrics() from network.py.
 */
struct ImprovementMetrics {
    double cost_improvement;       // (old - new) / old (positive = good)
    double efficiency_improvement; // new - old (positive = good)
    double cost_trend;            // Average rate of change (negative = good)
    double efficiency_trend;       // Average rate of change (positive = good)
};

/**
 * Compute improvement metrics from training history.
 * Matches Python implementation exactly.
 *
 * @param cost_history Vector of cost values
 * @param efficiency_history Vector of efficiency values
 * @param window_size Number of epochs to consider
 * @return ImprovementMetrics struct with computed values
 */
inline ImprovementMetrics compute_improvement_metrics(
    const std::vector<double>& cost_history,
    const std::vector<double>& efficiency_history,
    size_t window_size = 15
) {
    ImprovementMetrics metrics = {0.0, 0.0, 0.0, 0.0};

    if (cost_history.size() < 2 || efficiency_history.size() < 2) {
        return metrics;
    }

    // Cost improvement: (previous - current) / previous
    // Positive means improvement (cost decreased)
    size_t n = cost_history.size();
    double prev_cost = cost_history[n - 2];
    double curr_cost = cost_history[n - 1];
    if (prev_cost > 1e-10) {
        metrics.cost_improvement = (prev_cost - curr_cost) / prev_cost;
    }

    // Efficiency improvement: current - previous
    // Positive means improvement (efficiency increased)
    size_t m = efficiency_history.size();
    metrics.efficiency_improvement = efficiency_history[m - 1] - efficiency_history[m - 2];

    // Compute trends over window
    size_t actual_window = std::min(window_size, n);
    if (actual_window >= 2) {
        // Cost trend: (last - first) / (window * first_cost)
        size_t start_idx = n - actual_window;
        double start_cost = cost_history[start_idx];
        double end_cost = cost_history[n - 1];
        if (start_cost > 1e-10) {
            metrics.cost_trend = (end_cost - start_cost) / (actual_window * start_cost);
        }

        // Efficiency trend
        actual_window = std::min(window_size, m);
        if (actual_window >= 2) {
            start_idx = m - actual_window;
            double start_eff = efficiency_history[start_idx];
            double end_eff = efficiency_history[m - 1];
            metrics.efficiency_trend = (end_eff - start_eff) / actual_window;
        }
    }

    return metrics;
}

/**
 * Check if training should receive a reward.
 * Reward is given for good progress: cost improving AND efficiency good AND trend positive.
 * Matches Python _should_reward() from network.py lines 1852-1880.
 *
 * @param metrics Computed improvement metrics
 * @param config Reward/penalty configuration
 * @return Pair of (should_reward, magnitude)
 */
inline std::pair<bool, double> should_reward(
    const ImprovementMetrics& metrics,
    const RewardPenaltyConfig& config
) {
    // Cost improving
    bool cost_improving = metrics.cost_improvement > config.cost_improvement_threshold;

    // Efficiency is good (not degrading significantly)
    bool efficiency_good = metrics.efficiency_improvement > -config.efficiency_improvement_threshold;

    // Trend is positive (cost decreasing over window)
    bool trend_positive = metrics.cost_trend < 0;

    bool should = cost_improving && efficiency_good && trend_positive;

    double magnitude = 0.0;
    if (should) {
        // Magnitude proportional to improvement
        magnitude = std::abs(metrics.cost_improvement) +
                   std::abs(metrics.efficiency_improvement) * 0.5;
        magnitude = std::min(magnitude, 1.0);  // Cap at 1.0
    }

    return {should, magnitude};
}

/**
 * Check if training should receive a penalty.
 * Penalty is given for poor progress: cost degrading OR (efficiency bad AND trend negative).
 * Matches Python _should_penalize() from network.py lines 1882-1910.
 *
 * @param metrics Computed improvement metrics
 * @param config Reward/penalty configuration
 * @return Pair of (should_penalize, magnitude)
 */
inline std::pair<bool, double> should_penalize(
    const ImprovementMetrics& metrics,
    const RewardPenaltyConfig& config
) {
    // Cost degrading
    bool cost_degrading = metrics.cost_improvement < -config.cost_improvement_threshold;

    // Efficiency is bad (degrading)
    bool efficiency_bad = metrics.efficiency_improvement < -config.efficiency_improvement_threshold;

    // Trend is negative (cost increasing over window)
    bool trend_negative = metrics.cost_trend > 0;

    bool should = cost_degrading || (efficiency_bad && trend_negative);

    double magnitude = 0.0;
    if (should) {
        // Magnitude proportional to degradation
        magnitude = std::abs(metrics.cost_improvement) +
                   std::abs(metrics.efficiency_improvement) * 0.5;
        magnitude = std::min(magnitude, 1.0);  // Cap at 1.0
    }

    return {should, magnitude};
}

/**
 * Apply reward by decreasing learning rate.
 * The intuition: good progress means we're on the right track,
 * so we can take smaller steps to fine-tune.
 * Matches Python _apply_reward() from network.py lines 1912-1948.
 *
 * @param lr Current learning rate
 * @param magnitude Reward magnitude (0 to 1)
 * @param config Configuration
 * @param state Emotional state (modified in-place)
 * @return New learning rate
 */
inline double apply_reward(
    double lr,
    double magnitude,
    const RewardPenaltyConfig& config,
    EmotionalState& state
) {
    // Decrease factor proportional to magnitude
    // magnitude of 1.0 -> multiply by min_adjustment_factor (0.7)
    // magnitude of 0.0 -> no change
    double decrease_factor = 1.0 - magnitude * (1.0 - config.min_adjustment_factor);

    double new_lr = lr * decrease_factor;

    // Enforce bounds
    new_lr = std::max(config.min_learning_rate, std::min(config.max_learning_rate, new_lr));

    // Validate result
    if (!std::isfinite(new_lr) || new_lr <= 0) {
        new_lr = config.baseline_learning_rate;
    }

    // Update emotional state
    state.total_rewards++;
    state.reward_history.push_back(magnitude);

    return new_lr;
}

/**
 * Apply penalty by increasing learning rate.
 * The intuition: poor progress means we might be stuck,
 * so we need larger steps to escape.
 * Matches Python _apply_penalty() from network.py lines 1950-1986.
 *
 * @param lr Current learning rate
 * @param magnitude Penalty magnitude (0 to 1)
 * @param config Configuration
 * @param state Emotional state (modified in-place)
 * @return New learning rate
 */
inline double apply_penalty(
    double lr,
    double magnitude,
    const RewardPenaltyConfig& config,
    EmotionalState& state
) {
    // Increase factor proportional to magnitude
    // magnitude of 1.0 -> multiply by max_adjustment_factor (1.5)
    // magnitude of 0.0 -> no change
    double increase_factor = 1.0 + magnitude * (config.max_adjustment_factor - 1.0);

    double new_lr = lr * increase_factor;

    // Enforce bounds
    new_lr = std::max(config.min_learning_rate, std::min(config.max_learning_rate, new_lr));

    // Validate result
    if (!std::isfinite(new_lr) || new_lr <= 0) {
        new_lr = config.baseline_learning_rate;
    }

    // Update emotional state
    state.total_penalties++;
    state.penalty_history.push_back(magnitude);

    return new_lr;
}

/**
 * Check for extreme emotional states and adjust LR if needed.
 * Matches Python _check_extreme_states() from network.py lines 1988-2029.
 *
 * @param lr Current learning rate
 * @param config Configuration
 * @param state Emotional state (modified in-place)
 * @return Pair of (adjusted_lr, state_description)
 */
inline std::pair<double, std::string> check_extreme_states(
    double lr,
    const RewardPenaltyConfig& config,
    EmotionalState& state
) {
    double depression = state.depression_ratio();
    double excitement = state.excitement_ratio();

    std::string state_str = "neutral";

    // Check for extreme states - full reset to baseline
    if (depression > config.extreme_threshold) {
        lr = config.baseline_learning_rate;
        state.lr_reset_count++;
        state_str = "extreme_depression";
    } else if (excitement > config.extreme_threshold) {
        lr = config.baseline_learning_rate;
        state.lr_reset_count++;
        state_str = "extreme_excitement";
    }
    // Check for moderate states - partial correction toward baseline
    else if (depression > config.moderate_threshold) {
        // Move 50% toward baseline
        lr = lr + 0.5 * (config.baseline_learning_rate - lr);
        state_str = "depressed";
    } else if (excitement > config.moderate_threshold) {
        // Move 50% toward baseline
        lr = lr + 0.5 * (config.baseline_learning_rate - lr);
        state_str = "excited";
    }

    // Record history
    state.depression_history.push_back(depression);
    state.excitement_history.push_back(excitement);

    return {lr, state_str};
}

/**
 * Apply the complete reward/penalty system for one epoch.
 * Matches Python _apply_reward_penalty_system() from network.py lines 2031-2068.
 *
 * @param lr Current learning rate
 * @param cost_history Training cost history
 * @param efficiency_history Training efficiency history
 * @param config Configuration
 * @param state Emotional state (modified in-place)
 * @return Pair of (new_learning_rate, action_taken)
 *         action_taken: "reward", "penalty", "neutral", "extreme_depression_reset", "extreme_excitement_reset"
 */
inline std::pair<double, std::string> apply_reward_penalty_system(
    double lr,
    const std::vector<double>& cost_history,
    const std::vector<double>& efficiency_history,
    const RewardPenaltyConfig& config,
    EmotionalState& state
) {
    // Compute metrics
    ImprovementMetrics metrics = compute_improvement_metrics(
        cost_history, efficiency_history, config.window_size);

    std::string action = "neutral";

    // Check for reward
    auto [do_reward, reward_mag] = should_reward(metrics, config);
    if (do_reward) {
        lr = apply_reward(lr, reward_mag, config, state);
        action = "reward";
    } else {
        // Check for penalty
        auto [do_penalty, penalty_mag] = should_penalize(metrics, config);
        if (do_penalty) {
            lr = apply_penalty(lr, penalty_mag, config, state);
            action = "penalty";
        }
    }

    // Check for extreme states (may override previous adjustment)
    auto [new_lr, extreme_state] = check_extreme_states(lr, config, state);
    if (extreme_state.find("extreme") != std::string::npos) {
        action = extreme_state + "_reset";
    }

    return {new_lr, action};
}

/**
 * Compute efficiency metric based on cost reduction using windowed averaging.
 * Matches Python _compute_efficiency() from network.py lines 1579-1601.
 *
 * @param cost_history Vector of cost values
 * @param initial_efficiency Default efficiency if history too short
 * @param k Sigmoid steepness parameter
 * @return Efficiency value in [0, 1]
 */
inline double compute_efficiency(
    const std::vector<double>& cost_history,
    double initial_efficiency = 0.5,
    double k = 5.0
) {
    if (cost_history.size() < 2) {
        return initial_efficiency;
    }

    // Use a window for more stable efficiency computation
    size_t window_size = std::min(size_t(5), cost_history.size());
    size_t start_idx = cost_history.size() - window_size;

    // Calculate average improvement rate over window
    double first_cost = cost_history[start_idx];
    double last_cost = cost_history[cost_history.size() - 1];
    double total_improvement = 0.0;

    if (first_cost > 1e-8) {
        total_improvement = (first_cost - last_cost) / first_cost;
    }
    double avg_improvement = total_improvement / (window_size - 1);

    // Use sigmoid transformation for bounded, smooth efficiency
    // This maps any real number to (0, 1) range naturally
    double scaled_improvement = avg_improvement * 100.0;
    double efficiency = 1.0 / (1.0 + std::exp(-k * scaled_improvement / 5.0));

    return efficiency;
}

/**
 * Compute adaptive sigmoid threshold for saturation/efficiency decisions.
 * Matches Python _sigmoid_threshold() from network.py.
 *
 * @param efficiency Current efficiency
 * @param base_threshold Base threshold value
 * @param threshold_range Range of threshold adjustment
 * @param k Sigmoid steepness parameter
 * @return Adjusted threshold
 */
inline double sigmoid_threshold(
    double efficiency,
    double base_threshold = 0.3,
    double threshold_range = 0.5,
    double k = 5.0
) {
    // Sigmoid transformation centered at 0.5
    double sigmoid = 1.0 / (1.0 + std::exp(-k * (efficiency - 0.5)));

    // Map to threshold range
    // Low efficiency -> lower threshold -> more aggressive changes
    // High efficiency -> higher threshold -> conservative changes
    return base_threshold + threshold_range * sigmoid;
}

} // namespace training
} // namespace dnn
