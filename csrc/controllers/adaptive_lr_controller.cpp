#include "enn/controllers/adaptive_lr_controller.hpp"

#include <algorithm>
#include <cmath>


namespace enn::controllers {


double AdaptiveLRState::regression_rate() const {
    auto total = improvement_signals + regression_signals;
    return total > 0
        ? static_cast<double>(regression_signals) /
              static_cast<double>(total)
        : 0.0;
}


double AdaptiveLRState::improvement_rate() const {
    auto total = improvement_signals + regression_signals;
    return total > 0
        ? static_cast<double>(improvement_signals) /
              static_cast<double>(total)
        : 0.0;
}


void AdaptiveLRState::reset(std::size_t cap) {
    improvement_signals = 0;
    regression_signals = 0;
    lr_reset_count += 1;
    while (improvement_history.size() > cap) improvement_history.pop_front();
    while (regression_history.size() > cap) regression_history.pop_front();
    while (learning_rate_history.size() > cap)
        learning_rate_history.pop_front();
    while (action_history.size() > cap) action_history.pop_front();
}


void AdaptiveLRState::push_capped(std::deque<double>& q, double v,
                                   std::size_t cap) {
    q.push_back(v);
    while (q.size() > cap) q.pop_front();
}


void AdaptiveLRState::push_capped(std::deque<std::string>& q,
                                   std::string v, std::size_t cap) {
    q.push_back(std::move(v));
    while (q.size() > cap) q.pop_front();
}


double sigmoid_threshold(double utilization, double base, double range,
                          double k) {
    double centered = utilization - 0.5;
    double sig = 1.0 / (1.0 + std::exp(-k * centered));
    return base + range * sig;
}


ImprovementMetrics compute_improvement_metrics(
    const std::vector<double>& cost_history,
    const std::vector<double>& utilization_history,
    std::size_t window_size) {
    ImprovementMetrics m{0.0, 0.0, 0.0, 0.0};
    if (cost_history.size() < 2) return m;

    auto compute_trend = [&](const std::vector<double>& h) {
        std::size_t n = std::min(window_size, h.size());
        if (n < 2) return 0.0;
        double first = h[h.size() - n];
        double last = h.back();
        return (first - last) / std::max(std::abs(first), 1e-9);
    };

    m.cost_improvement =
        (cost_history.size() >= 2)
            ? (cost_history[cost_history.size() - 2] - cost_history.back()) /
                  std::max(cost_history[cost_history.size() - 2], 1e-9)
            : 0.0;
    if (!utilization_history.empty()) {
        m.utilization_improvement =
            (utilization_history.size() >= 2)
                ? (utilization_history.back() -
                   utilization_history[utilization_history.size() - 2])
                : 0.0;
    }
    m.cost_trend = compute_trend(cost_history);
    m.utilization_trend = compute_trend(utilization_history);
    return m;
}


std::pair<bool, double> should_reward(const ImprovementMetrics& m,
                                       const AdaptiveLRConfig& cfg) {
    bool cost_ok = m.cost_improvement > cfg.cost_improvement_threshold;
    bool util_ok =
        m.utilization_improvement > cfg.utilization_improvement_threshold;
    if (cost_ok && util_ok) {
        double mag = std::min(
            1.0, (m.cost_improvement + m.utilization_improvement) /
                     (cfg.cost_improvement_threshold +
                      cfg.utilization_improvement_threshold));
        return {true, mag};
    }
    return {false, 0.0};
}


std::pair<bool, double> should_penalize(const ImprovementMetrics& m,
                                         const AdaptiveLRConfig& cfg) {
    bool cost_bad = m.cost_improvement < -cfg.cost_improvement_threshold;
    bool util_bad =
        m.utilization_improvement < -cfg.utilization_improvement_threshold;
    if (cost_bad || util_bad) {
        double cost_mag = std::max(0.0, -m.cost_improvement) /
                          std::max(cfg.cost_improvement_threshold, 1e-9);
        double util_mag = std::max(0.0, -m.utilization_improvement) /
                          std::max(cfg.utilization_improvement_threshold, 1e-9);
        return {true, std::min(1.0, std::max(cost_mag, util_mag))};
    }
    return {false, 0.0};
}


double apply_reward(double lr, double magnitude,
                    const AdaptiveLRConfig& cfg, AdaptiveLRState& s) {
    double factor = cfg.min_decrease_factor +
                    (1.0 - cfg.min_decrease_factor) * (1.0 - magnitude);
    double next = std::max(cfg.min_learning_rate,
                            std::min(cfg.max_learning_rate, lr * factor));
    s.improvement_signals += 1;
    return next;
}


double apply_penalty(double lr, double magnitude,
                     const AdaptiveLRConfig& cfg, AdaptiveLRState& s) {
    double factor = 1.0 + (cfg.max_increase_factor - 1.0) * magnitude;
    double next = std::max(cfg.min_learning_rate,
                            std::min(cfg.max_learning_rate, lr * factor));
    s.regression_signals += 1;
    return next;
}


std::pair<double, std::string> check_extreme_states(
    double lr, const AdaptiveLRConfig& cfg, AdaptiveLRState& s) {
    double dep = s.regression_rate();
    double exc = s.improvement_rate();
    if (dep >= cfg.critical_imbalance_threshold) {
        s.reset(cfg.history_window);
        return {cfg.baseline_learning_rate, "critical_regression_reset"};
    }
    if (exc >= cfg.critical_imbalance_threshold) {
        s.reset(cfg.history_window);
        return {cfg.baseline_learning_rate, "critical_improvement_reset"};
    }
    if (dep >= cfg.moderate_imbalance_threshold) {
        return {std::min(cfg.max_learning_rate,
                          lr * cfg.max_increase_factor),
                "moderate_regression_correction"};
    }
    if (exc >= cfg.moderate_imbalance_threshold) {
        return {std::max(cfg.min_learning_rate,
                          lr * cfg.min_decrease_factor),
                "moderate_improvement_correction"};
    }
    return {lr, "none"};
}


std::pair<double, std::string> step_adaptive_lr_controller(
    double lr,
    const std::vector<double>& cost_history,
    const std::vector<double>& utilization_history,
    const AdaptiveLRConfig& cfg, AdaptiveLRState& s) {
    auto m = compute_improvement_metrics(cost_history, utilization_history,
                                          cfg.trend_window);
    auto [reward, r_mag] = should_reward(m, cfg);
    if (reward) {
        double next = apply_reward(lr, r_mag, cfg, s);
        s.push_capped(s.learning_rate_history, next, cfg.history_window);
        s.push_capped(s.action_history, "reward", cfg.history_window);
        return {next, "reward"};
    }
    auto [penalize, p_mag] = should_penalize(m, cfg);
    if (penalize) {
        double next = apply_penalty(lr, p_mag, cfg, s);
        s.push_capped(s.learning_rate_history, next, cfg.history_window);
        s.push_capped(s.action_history, "penalty", cfg.history_window);
        return {next, "penalty"};
    }
    auto [next_lr, action] = check_extreme_states(lr, cfg, s);
    s.push_capped(s.learning_rate_history, next_lr, cfg.history_window);
    s.push_capped(s.action_history, action, cfg.history_window);
    return {next_lr, action};
}


}  // namespace enn::controllers
