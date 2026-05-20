#include <gtest/gtest.h>

#include "enn/controllers/plateau_detector.hpp"


using enn::controllers::PlateauConfig;
using enn::controllers::PlateauDetector;


TEST(PlateauDetector, FlatSignalIsLocalMax) {
    PlateauConfig cfg;
    cfg.window = 10;
    cfg.warmup_epochs = 5;
    cfg.cooldown_epochs = 0;
    cfg.slope_epsilon = 1e-3;
    PlateauDetector d(cfg);
    for (int i = 0; i < 20; ++i) d.tick(0.5);
    EXPECT_TRUE(d.is_at_local_max());
    EXPECT_NEAR(d.slope(), 0.0, 1e-9);
}

TEST(PlateauDetector, RisingSignalNotLocalMax) {
    PlateauConfig cfg;
    cfg.window = 10;
    cfg.warmup_epochs = 5;
    cfg.cooldown_epochs = 0;
    cfg.slope_epsilon = 1e-3;
    PlateauDetector d(cfg);
    for (int i = 0; i < 20; ++i) d.tick(0.05 * i);
    EXPECT_FALSE(d.is_at_local_max());
    EXPECT_GT(d.slope(), 0.0);
}

TEST(PlateauDetector, CooldownSuppressesAfterEvent) {
    PlateauConfig cfg;
    cfg.window = 5;
    cfg.warmup_epochs = 0;
    cfg.cooldown_epochs = 3;
    cfg.slope_epsilon = 1e-3;
    PlateauDetector d(cfg);
    for (int i = 0; i < 10; ++i) d.tick(0.5);
    EXPECT_TRUE(d.is_at_local_max());
    d.notify_event_fired();
    d.tick(0.5);
    EXPECT_FALSE(d.is_at_local_max());
}

TEST(PlateauDetector, RejectsTooSmallWindow) {
    PlateauConfig cfg;
    cfg.window = 2;
    EXPECT_THROW(PlateauDetector d(cfg), std::invalid_argument);
}
