#include <gtest/gtest.h>

#include "enn/controllers/stability_monitor.hpp"


using enn::controllers::StabilityConfig;
using enn::controllers::StabilityMonitor;
using enn::controllers::StabilityState;


TEST(StabilityMonitor, InitialIsStable) {
    StabilityConfig cfg;
    StabilityMonitor m(3, 30, cfg);
    EXPECT_EQ(m.current_state(), StabilityState::Stable);
}

TEST(StabilityMonitor, BelowMinLayersIsCritical) {
    StabilityConfig cfg;
    cfg.min_layers = 2;
    StabilityMonitor m(1, 4, cfg);
    EXPECT_EQ(m.current_state(), StabilityState::Critical);
}

TEST(StabilityMonitor, ManyGrowthEventsRaiseAnomalyScore) {
    StabilityConfig cfg;
    cfg.history_window = 20;
    StabilityMonitor m(2, 10, cfg);
    for (int i = 0; i < 20; ++i) {
        m.update_epoch(i);
        m.record_change(1, 5);
        m.update_topology(2 + i, 10 + 5 * i);
    }
    auto s = m.current_state();
    EXPECT_TRUE(s == StabilityState::PathologicalGrowth ||
                s == StabilityState::ExcessiveGrowthRisk);
}

TEST(StabilityMonitor, AllowLayerAdditionWhenStable) {
    StabilityConfig cfg;
    StabilityMonitor m(3, 30, cfg);
    EXPECT_TRUE(m.allow_layer_addition());
}
