#include <gtest/gtest.h>

#include "enn/controllers/adaptive_lr_controller.hpp"


using enn::controllers::AdaptiveLRConfig;
using enn::controllers::AdaptiveLRState;
using enn::controllers::compute_improvement_metrics;
using enn::controllers::sigmoid_threshold;
using enn::controllers::step_adaptive_lr_controller;


TEST(AdaptiveLRState, RatesSumToOne) {
    AdaptiveLRState s;
    s.improvement_signals = 3;
    s.regression_signals = 7;
    EXPECT_NEAR(s.improvement_rate() + s.regression_rate(), 1.0, 1e-9);
}

TEST(AdaptiveLRState, EmptyRatesAreZero) {
    AdaptiveLRState s;
    EXPECT_EQ(s.regression_rate(), 0.0);
    EXPECT_EQ(s.improvement_rate(), 0.0);
}

TEST(SigmoidThreshold, ReturnsBaseAtZeroAndApproachesBasePlusRangeAtOne) {
    double low = sigmoid_threshold(-1.0, 0.3, 0.5, 5.0);
    double mid = sigmoid_threshold(0.5, 0.3, 0.5, 5.0);
    double high = sigmoid_threshold(2.0, 0.3, 0.5, 5.0);
    EXPECT_LT(low, mid);
    EXPECT_LT(mid, high);
    EXPECT_NEAR(mid, 0.3 + 0.5 * 0.5, 1e-3);
}

TEST(AdaptiveLRController, RewardOnSteadyImprovement) {
    AdaptiveLRConfig cfg;
    AdaptiveLRState s;
    std::vector<double> cost{1.0, 0.9, 0.8, 0.7, 0.6};
    std::vector<double> util{0.2, 0.3, 0.4, 0.5, 0.6};
    auto [next_lr, action] = step_adaptive_lr_controller(
        cfg.baseline_learning_rate, cost, util, cfg, s);
    EXPECT_GT(s.improvement_signals + s.regression_signals, 0u);
    EXPECT_NE(action, "");
}

TEST(AdaptiveLRController, PenaltyOnRegression) {
    AdaptiveLRConfig cfg;
    AdaptiveLRState s;
    std::vector<double> cost{0.5, 0.6, 0.7, 0.8, 0.9};
    std::vector<double> util{0.6, 0.5, 0.4, 0.3, 0.2};
    auto [next_lr, action] = step_adaptive_lr_controller(
        cfg.baseline_learning_rate, cost, util, cfg, s);
    EXPECT_GE(next_lr, cfg.min_learning_rate);
}
