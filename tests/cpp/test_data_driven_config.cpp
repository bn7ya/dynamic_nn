#include <gtest/gtest.h>
#include <torch/torch.h>

#include "enn/training/data_driven_config.hpp"


using enn::training::compute_dataset_statistics;
using enn::training::derive_adaptive_lr_config;
using enn::training::derive_phase_schedule;
using enn::training::derive_plateau_config;
using enn::training::derive_pruning_config;
using enn::training::derive_stability_config;


TEST(DataDrivenConfig, GaussianStatistics) {
    torch::manual_seed(0);
    auto X = torch::randn({256, 16});
    auto y = torch::randint(0, 4, {256}, torch::kInt64);
    auto s = compute_dataset_statistics(X, y);
    EXPECT_EQ(s.num_samples, 256);
    EXPECT_EQ(s.num_features, 16);
    EXPECT_EQ(s.num_classes, 4);
    EXPECT_FALSE(s.is_regression);
    EXPECT_GE(s.label_entropy, 0.0);
    EXPECT_LE(s.label_entropy, 1.0);
}

TEST(DataDrivenConfig, DerivedConfigsFiniteAndPositive) {
    torch::manual_seed(0);
    auto X = torch::randn({200, 12});
    auto y = torch::randint(0, 3, {200}, torch::kInt64);
    auto s = compute_dataset_statistics(X, y);
    auto lr = derive_adaptive_lr_config(s);
    EXPECT_GT(lr.baseline_learning_rate, 0.0);
    EXPECT_LT(lr.min_learning_rate, lr.max_learning_rate);
    auto st = derive_stability_config(s);
    EXPECT_GT(st.growth_anomaly_threshold, 0.0);
    EXPECT_LT(st.growth_anomaly_threshold, 1.0);
    auto pr = derive_pruning_config(s);
    EXPECT_GT(pr.prune_utilization_threshold, 0.0);
    auto ps = derive_phase_schedule(s);
    EXPECT_GT(ps.topology_discovery_epochs, 0);
    auto pl = derive_plateau_config(s);
    EXPECT_GE(pl.window, std::size_t{3});
}
