#include "enn/training/early_stopping.hpp"

#include <algorithm>
#include <cmath>


namespace enn::training {


void EarlyStopping::reset() {
    best_utilization_ = 0.0;
    epochs_since_improvement_ = 0;
    epochs_ = 0;
}


StoppingDecision EarlyStopping::evaluate(
    const std::vector<double>& utilization_history,
    const std::vector<double>& cost_history) {
    (void)cost_history;
    epochs_ = utilization_history.size();
    if (epochs_ < config_.min_epochs) {
        return {false, "warmup", epochs_};
    }
    double current = utilization_history.back();
    if (current > best_utilization_ + config_.min_improvement) {
        best_utilization_ = current;
        epochs_since_improvement_ = 0;
    } else {
        ++epochs_since_improvement_;
    }
    if (epochs_since_improvement_ >= config_.patience) {
        return {true, "patience_exhausted", epochs_};
    }
    std::size_t n = std::min(config_.recent_window,
                              utilization_history.size());
    double recent = 0.0;
    for (std::size_t i = utilization_history.size() - n;
         i < utilization_history.size(); ++i) {
        recent += utilization_history[i];
    }
    recent /= static_cast<double>(n);
    if (recent < best_utilization_ * config_.regression_threshold &&
        epochs_ >= 2 * config_.min_epochs) {
        return {true, "utilization_regression", epochs_};
    }
    if (current < config_.critical_utilization &&
        epochs_ >= 2 * config_.min_epochs) {
        return {true, "critical_utilization", epochs_};
    }
    return {false, "none", epochs_};
}


}  // namespace enn::training
