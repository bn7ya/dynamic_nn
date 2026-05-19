#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dnn {
namespace training {
namespace runtime {

/**
 * Identifier for which stage worker emitted a metric sample.
 */
enum class StageId : uint8_t {
    Exploration = 0,
    Estimation  = 1,
    Main        = 2,
    Standard    = 3,
    Controller  = 4
};

/**
 * One metric sample published by a stage worker. Lightweight and copyable
 * so the bus can hand it back by value.
 */
struct MetricSample {
    StageId stage = StageId::Controller;
    uint64_t epoch = 0;
    double cost = 0.0;
    double efficiency = 0.0;
    double learning_rate = 0.0;
    uint64_t topology_version = 0;
};

/**
 * Bounded ring buffer carrying the most-recent N MetricSamples emitted by
 * stage workers. Workers call publish() from their training thread; the
 * controller drains via snapshot() to make decisions about which stages
 * to wake or suspend.
 *
 * Lock-free: each publish() reserves a unique slot via an atomic
 * fetch-add and writes it, then publishes visibility with a release
 * store on next_. Readers acquire-load next_ and copy the live window.
 * The observer that drains this re-polls every few ms and only needs
 * the most-recent trend, so a rare slightly-stale read is acceptable;
 * the previous design held a mutex purely to copy an 80-byte struct.
 */
class MetricsBus {
public:
    explicit MetricsBus(size_t capacity = 4096)
        : capacity_(capacity), buffer_(capacity) {}

    void publish(const MetricSample& sample) {
        // Reserve a unique slot (multi-producer safe), write it, then
        // release-publish the advanced count so a reader that acquires
        // the new count also sees the slot contents.
        uint64_t idx = reserved_.fetch_add(1, std::memory_order_relaxed);
        buffer_[idx % capacity_] = sample;
        next_.store(idx + 1, std::memory_order_release);
    }

    /**
     * Snapshot of the most recent samples (ordered oldest -> newest).
     * Returns at most `capacity` items.
     */
    std::vector<MetricSample> snapshot() const {
        uint64_t n = next_.load(std::memory_order_acquire);
        std::vector<MetricSample> out;
        size_t count = n < capacity_ ? n : capacity_;
        out.reserve(count);
        size_t start = n < capacity_ ? 0 : n - capacity_;
        for (size_t i = 0; i < count; ++i) {
            out.push_back(buffer_[(start + i) % capacity_]);
        }
        return out;
    }

    /**
     * Most recent sample emitted by `stage`, or std::nullopt-like default
     * if none yet.
     */
    MetricSample latest(StageId stage) const {
        uint64_t n = next_.load(std::memory_order_acquire);
        size_t count = n < capacity_ ? n : capacity_;
        size_t start = n < capacity_ ? 0 : n - capacity_;
        MetricSample out;  // default-initialised cost=0
        for (size_t i = 0; i < count; ++i) {
            const auto& s = buffer_[(start + i) % capacity_];
            if (s.stage == stage) out = s;  // keep last
        }
        return out;
    }

    uint64_t total_published() const {
        return next_.load(std::memory_order_acquire);
    }

private:
    size_t capacity_;
    std::vector<MetricSample> buffer_;
    std::atomic<uint64_t> reserved_{0};
    std::atomic<uint64_t> next_{0};
};

}  // namespace runtime
}  // namespace training
}  // namespace dnn
