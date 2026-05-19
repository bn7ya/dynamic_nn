#include <gtest/gtest.h>

#include <thread>
#include <vector>

#include "dnn/training/runtime/metrics_bus.hpp"

using dnn::training::runtime::MetricsBus;
using dnn::training::runtime::MetricSample;
using dnn::training::runtime::StageId;

// 2.9: lock-free ring preserves ordering and the most-recent window.
TEST(MetricsBus, SnapshotAndLatestOrdering) {
    MetricsBus bus(4);
    for (uint64_t e = 1; e <= 6; ++e) {
        MetricSample s;
        s.stage = (e % 2 == 0) ? StageId::Main : StageId::Exploration;
        s.epoch = e;
        s.cost = static_cast<double>(e);
        bus.publish(s);
    }
    EXPECT_EQ(bus.total_published(), 6u);

    auto snap = bus.snapshot();
    ASSERT_EQ(snap.size(), 4u);  // capacity-bounded
    EXPECT_EQ(snap.front().epoch, 3u);
    EXPECT_EQ(snap.back().epoch, 6u);

    EXPECT_EQ(bus.latest(StageId::Main).epoch, 6u);
    EXPECT_EQ(bus.latest(StageId::Exploration).epoch, 5u);
}

// 2.9: concurrent multi-producer publish stays consistent (count exact,
// every observed sample well-formed).
TEST(MetricsBus, ConcurrentProducers) {
    MetricsBus bus(1024);
    auto producer = [&](StageId id) {
        for (int i = 0; i < 5000; ++i) {
            MetricSample s;
            s.stage = id;
            s.epoch = static_cast<uint64_t>(i);
            s.cost = 1.0;
            bus.publish(s);
        }
    };
    std::thread t1(producer, StageId::Exploration);
    std::thread t2(producer, StageId::Main);
    std::thread t3(producer, StageId::Standard);
    t1.join();
    t2.join();
    t3.join();

    EXPECT_EQ(bus.total_published(), 15000u);
    for (const auto& s : bus.snapshot()) EXPECT_DOUBLE_EQ(s.cost, 1.0);
}
