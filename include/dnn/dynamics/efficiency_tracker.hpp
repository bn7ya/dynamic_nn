#pragma once

#include "../exceptions/dnn_exception.hpp"
#include <cstddef>
#include <deque>

namespace dnn {
namespace dynamics {

/**
 * Configuration for the local-maximum efficiency gate.
 *
 * Gradient descent finds a local minimum of cost (dC/dt -> 0,
 * d2C/dt2 > 0). The analog for efficiency E(t) is a local maximum:
 * dE/dt -> 0 with d2E/dt2 <= 0. EfficiencyTracker implements the
 * discrete-time test of that condition on a sliding window of recent
 * per-epoch efficiency values.
 *
 * Defaults match LayerManagerConfig defaults and are validated there;
 * EfficiencyTracker itself only checks the minimum viable values it
 * needs for its arithmetic to make sense.
 */
struct EfficiencyGateConfig {
    std::size_t window = 10;            // W: number of epochs in the slope/curvature window
    double slope_epsilon = 1e-4;        // |dE/dt| threshold for "flat enough"
    std::size_t min_epochs_before_shrink = 8;  // warmup before any shrink may fire
    std::size_t cooldown_epochs = 5;    // epochs to wait after a shrink fires
};

/**
 * Sliding-window local-maximum detector on a scalar efficiency time series.
 *
 * Discrete derivatives:
 *   slope = least-squares regression slope over the window
 *           ( = Sum_t (t - tbar)(E_t - Ebar) / Sum_t (t - tbar)^2 ).
 *   curvature = mean of finite second differences
 *               ( = mean over interior points of E_{t+1} - 2 E_t + E_{t-1} ).
 *
 * is_at_local_max() is the conjunction of:
 *   1. window full,
 *   2. enough total ticks (warmup),
 *   3. enough ticks since the last shrink (cooldown),
 *   4. |slope| <= slope_epsilon (E is "flat"),
 *   5. curvature <= 0 (peak or descent, not a trough on the way up).
 *
 * Pure host-side scalar math; no device interaction.
 */
class EfficiencyTracker {
public:
    EfficiencyTracker() : EfficiencyTracker(EfficiencyGateConfig{}) {}

    explicit EfficiencyTracker(const EfficiencyGateConfig& config)
        : config_(config)
        , epochs_seen_(0)
        , epochs_since_last_shrink_(config.cooldown_epochs) {
        if (config.window < 3) {
            throw exceptions::InvalidArgumentException(
                "EfficiencyGateConfig.window",
                "window must be >= 3 to estimate a finite second difference");
        }
        if (!(config.slope_epsilon >= 0.0)) {
            throw exceptions::InvalidArgumentException(
                "EfficiencyGateConfig.slope_epsilon",
                "slope_epsilon must be non-negative");
        }
    }

    /** Append an efficiency sample for the current epoch. */
    void tick(double efficiency) {
        window_.push_back(efficiency);
        if (window_.size() > config_.window) {
            window_.pop_front();
        }
        ++epochs_seen_;
        if (epochs_since_last_shrink_ < config_.cooldown_epochs) {
            ++epochs_since_last_shrink_;
        }
    }

    /** Reset cooldown counter after a shrink action has fired. */
    void notify_shrink_fired() {
        epochs_since_last_shrink_ = 0;
    }

    /** Least-squares slope of efficiency over the window. */
    double slope() const {
        const std::size_t n = window_.size();
        if (n < 2) return 0.0;
        double mean_t = 0.0;
        double mean_e = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            mean_t += static_cast<double>(i);
            mean_e += window_[i];
        }
        mean_t /= static_cast<double>(n);
        mean_e /= static_cast<double>(n);
        double num = 0.0;
        double den = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double dt = static_cast<double>(i) - mean_t;
            num += dt * (window_[i] - mean_e);
            den += dt * dt;
        }
        if (den <= 0.0) return 0.0;
        return num / den;
    }

    /** Mean second-difference estimate of curvature over the window. */
    double curvature() const {
        const std::size_t n = window_.size();
        if (n < 3) return 0.0;
        double sum = 0.0;
        std::size_t count = 0;
        for (std::size_t i = 1; i + 1 < n; ++i) {
            sum += window_[i + 1] - 2.0 * window_[i] + window_[i - 1];
            ++count;
        }
        if (count == 0) return 0.0;
        return sum / static_cast<double>(count);
    }

    /**
     * Local-maximum test: window full, warmup satisfied, cooldown
     * satisfied, slope flat, curvature non-positive.
     */
    bool is_at_local_max() const {
        if (window_.size() < config_.window) return false;
        if (epochs_seen_ < config_.min_epochs_before_shrink) return false;
        if (epochs_since_last_shrink_ < config_.cooldown_epochs) return false;
        double s = slope();
        if (s > config_.slope_epsilon || s < -config_.slope_epsilon) return false;
        return curvature() <= 0.0;
    }

    std::size_t epochs_seen() const { return epochs_seen_; }
    std::size_t window_size() const { return window_.size(); }
    const EfficiencyGateConfig& config() const { return config_; }

private:
    EfficiencyGateConfig config_;
    std::deque<double> window_;
    std::size_t epochs_seen_;
    std::size_t epochs_since_last_shrink_;
};

}  // namespace dynamics
}  // namespace dnn
