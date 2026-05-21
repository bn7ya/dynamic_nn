#pragma once

#include <cstddef>
#include <deque>
#include <string>
#include <utility>
#include <vector>


namespace enn::controllers {


struct AdaptiveLRConfig {
    double cost_improvement_threshold = 0.005;
    double utilization_improvement_threshold = 0.02;
    double critical_imbalance_threshold = 0.7;
    double moderate_imbalance_threshold = 0.6;
    double max_increase_factor = 1.5;
    double min_decrease_factor = 0.7;
    double min_learning_rate = 1e-5;
    double max_learning_rate = 0.5;
    double baseline_learning_rate = 0.1;
    std::size_t trend_window = 15;
    std::size_t history_window = 500;
    double sigmoid_steepness = 5.0;
    double sigmoid_base = 0.3;
    double sigmoid_range = 0.5;
};


class AdaptiveLRState {
public:
    std::size_t improvement_signals = 0;
    std::size_t regression_signals = 0;
    std::size_t lr_reset_count = 0;
    std::deque<double> improvement_history;
    std::deque<double> regression_history;
    std::deque<double> learning_rate_history;
    std::deque<std::string> action_history;

    double regression_rate() const;
    double improvement_rate() const;
    void reset(std::size_t cap);
    void push_capped(std::deque<double>& q, double v, std::size_t cap);
    void push_capped(std::deque<std::string>& q, std::string v,
                     std::size_t cap);
};


struct ImprovementMetrics {
    double cost_improvement;
    double utilization_improvement;
    double cost_trend;
    double utilization_trend;
};


ImprovementMetrics compute_improvement_metrics(
    const std::vector<double>& cost_history,
    const std::vector<double>& utilization_history,
    std::size_t window_size);


std::pair<bool, double> should_reward(const ImprovementMetrics& m,
                                       const AdaptiveLRConfig& cfg);

std::pair<bool, double> should_penalize(const ImprovementMetrics& m,
                                         const AdaptiveLRConfig& cfg);

double apply_reward(double lr, double magnitude,
                    const AdaptiveLRConfig& cfg, AdaptiveLRState& s);

double apply_penalty(double lr, double magnitude,
                     const AdaptiveLRConfig& cfg, AdaptiveLRState& s);

std::pair<double, std::string> check_extreme_states(
    double lr, const AdaptiveLRConfig& cfg, AdaptiveLRState& s);

std::pair<double, std::string> step_adaptive_lr_controller(
    double lr,
    const std::vector<double>& cost_history,
    const std::vector<double>& utilization_history,
    const AdaptiveLRConfig& cfg, AdaptiveLRState& s);

double sigmoid_threshold(double utilization, double base,
                          double range, double k);


}  // namespace enn::controllers
