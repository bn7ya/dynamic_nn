#pragma once

#include <cstddef>
#include <string>
#include <vector>


namespace enn::training {


struct EarlyStoppingConfig {
    std::size_t patience = 10;
    double min_improvement = 0.001;
    std::size_t min_epochs = 10;
    double regression_threshold = 0.8;
    double critical_utilization = 0.1;
    std::size_t recent_window = 10;
};


struct StoppingDecision {
    bool should_stop;
    std::string reason;
    std::size_t epochs_seen;
};


class EarlyStopping {
public:
    explicit EarlyStopping(EarlyStoppingConfig cfg = {}) : config_(cfg) {}

    StoppingDecision evaluate(const std::vector<double>& utilization_history,
                               const std::vector<double>& cost_history);
    void reset();
    const EarlyStoppingConfig& config() const noexcept { return config_; }

private:
    EarlyStoppingConfig config_;
    double best_utilization_ = 0.0;
    std::size_t epochs_since_improvement_ = 0;
    std::size_t epochs_ = 0;
};


}  // namespace enn::training
