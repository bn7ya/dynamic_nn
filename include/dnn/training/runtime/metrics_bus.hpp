#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
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
 * A single mutex is enough: this is firmly off the hot path (one push per
 * epoch per stage), and a real lock-free implementation would just hide
 * the cost of copying an 80-byte struct.
 */
class MetricsBus {
public:
    explicit MetricsBus(size_t capacity = 4096)
        : capacity_(capacity), buffer_(capacity) {}

    void publish(const MetricSample& sample) {
        std::lock_guard<std::mutex> lock(mu_);
        buffer_[next_ % capacity_] = sample;
        ++next_;
    }

    /**
     * Snapshot of the most recent samples (ordered oldest -> newest).
     * Returns at most `capacity` items.
     */
    std::vector<MetricSample> snapshot() const {
        std::lock_guard<std::mutex> lock(mu_);
        std::vector<MetricSample> out;
        size_t count = next_ < capacity_ ? next_ : capacity_;
        out.reserve(count);
        size_t start = next_ < capacity_ ? 0 : next_ - capacity_;
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
        std::lock_guard<std::mutex> lock(mu_);
        size_t count = next_ < capacity_ ? next_ : capacity_;
        size_t start = next_ < capacity_ ? 0 : next_ - capacity_;
        MetricSample out;  // default-initialised cost=0
        for (size_t i = 0; i < count; ++i) {
            const auto& s = buffer_[(start + i) % capacity_];
            if (s.stage == stage) out = s;  // keep last
        }
        return out;
    }

    uint64_t total_published() const {
        std::lock_guard<std::mutex> lock(mu_);
        return next_;
    }

private:
    mutable std::mutex mu_;
    size_t capacity_;
    std::vector<MetricSample> buffer_;
    uint64_t next_ = 0;
};

}  // namespace runtime
}  // namespace training
}  // namespace dnn
