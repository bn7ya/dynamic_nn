#pragma once

#include "adaptive_config.hpp"
#include "metrics_bus.hpp"
#include "stage_worker.hpp"
#include "topology_lock.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace dnn {
namespace training {
namespace runtime {

/**
 * Coordinator for the four-stage training pipeline.
 *
 * The trainer drives the controller by handing each stage in as a pair of
 * callables — `step(epoch)` runs one epoch's body, `should_continue()`
 * decides whether to keep stepping. The controller runs them in
 * feedback-loop fashion:
 *
 *   1. The Estimation **observer** thread runs in parallel with whichever
 *      training stage is currently active. It drains the MetricsBus,
 *      computes a rolling cost trend, and (a) nudges the
 *      RuntimeAdaptiveConfig scalars and (b) raises rewind_requested()
 *      when convergence stalls, asking the controller to rewind to an
 *      earlier stage.
 *   2. Training stages check rewind_requested() between steps and bail
 *      out of their inner loop when it fires. The controller then
 *      restarts the requested earlier stage from its preserved state.
 *   3. Topology mutations (add_nodes / remove_nodes / mark_layer_inactive)
 *      acquire the TopologyLock in write mode; concurrent observer reads
 *      take it in shared mode. Combined with the soft-delete masks on
 *      Layer<T>, no in-flight worker ever sees a half-applied change.
 *
 * The controller is intentionally type-agnostic: it doesn't know about
 * Tensor<T>, the network, or the trainer. Callers wrap their per-epoch
 * body in a lambda that captures `this` from the trainer, and the
 * controller orchestrates the schedule and the bus.
 */
class StageController {
public:
    using StageStep = std::function<void(uint64_t /*epoch_in_stage*/)>;
    using StageContinue = std::function<bool()>;

    StageController(MetricsBus& bus,
                    RuntimeAdaptiveConfig& cfg,
                    TopologyLock& lock)
        : bus_(bus), cfg_(cfg), lock_(lock) {}

    ~StageController() {
        stop_observer();
    }

    StageController(const StageController&) = delete;
    StageController& operator=(const StageController&) = delete;

    /**
     * Run a single stage as a sequence of step() invocations.
     *
     * @param id              Which stage is running (for metrics tagging).
     * @param step            The per-epoch body. Receives the in-stage
     *                        epoch index so the body can use it for
     *                        per-stage scheduling decisions (e.g. "first
     *                        20% of epochs apply perturbation").
     * @param should_continue Predicate consulted between steps. Stages
     *                        should also check rewind_requested() inside
     *                        their step() body for fast-exit.
     * @param max_epochs      Hard cap on iterations regardless of
     *                        should_continue (0 = no cap).
     *
     * Returns the number of steps actually executed.
     */
    uint64_t run_stage(StageId id,
                       const StageStep& step,
                       const StageContinue& should_continue,
                       uint64_t max_epochs = 0) {
        active_stage_.store(id);
        uint64_t executed = 0;
        while (true) {
            if (max_epochs != 0 && executed >= max_epochs) break;
            if (!should_continue()) break;
            if (rewind_requested() && id != StageId::Estimation) break;

            // Topology shared-lock around the step body. The body may take
            // the write lock internally for soft mutations; that upgrade
            // is the caller's responsibility.
            {
                auto guard = lock_.read_lock();
                step(executed);
            }
            ++executed;
        }
        return executed;
    }

    /**
     * Spawn the parallel Estimation observer.
     *
     * The observer reads the MetricsBus, computes recent cost-trend, and
     * mutates the RuntimeAdaptiveConfig (e.g. nudges patience up when
     * progress is slow, dials LR floor when it's volatile). It also
     * raises rewind_requested(target) when cost stops improving so the
     * active stage exits and the controller can wake an earlier stage.
     *
     * Idempotent: calling start while already running is a no-op.
     */
    void start_observer() {
        bool expected = false;
        if (!observer_running_.compare_exchange_strong(expected, true)) return;
        observer_thread_ = std::thread([this] { observer_loop(); });
    }

    void stop_observer() {
        if (!observer_running_.load()) return;
        observer_running_.store(false);
        if (observer_thread_.joinable()) {
            observer_thread_.join();
        }
    }

    /**
     * Tell the observer to skip its next sleep and re-check immediately.
     * Useful right after a stage publishes a fresh metric.
     */
    void notify_observer() {
        std::lock_guard<std::mutex> lock(observer_mu_);
        observer_cv_.notify_all();
    }

    /**
     * Has the observer asked the controller to rewind to an earlier stage?
     *
     * `rewind_requested()` uses acquire ordering paired with the
     * release-store inside `request_rewind()`, so any reader that sees
     * the flag set is guaranteed to also see the matching `rewind_target_`
     * value written before it.
     */
    bool rewind_requested() const { return rewind_.load(std::memory_order_acquire); }
    StageId rewind_target() const { return rewind_target_.load(std::memory_order_acquire); }
    void clear_rewind() {
        rewind_.store(false, std::memory_order_release);
    }
    void request_rewind(StageId target) {
        // Publish the target first with release ordering, then flip the
        // flag with another release. A reader that sees the flag (via
        // acquire) is guaranteed to also see this target store.
        rewind_target_.store(target, std::memory_order_release);
        rewind_.store(true, std::memory_order_release);
    }

    /**
     * Convenience for stages that want to publish a metric sample.
     */
    void publish_metric(StageId id,
                        uint64_t epoch,
                        double cost,
                        double efficiency,
                        double learning_rate,
                        uint64_t topology_version) {
        MetricSample s;
        s.stage = id;
        s.epoch = epoch;
        s.cost = cost;
        s.efficiency = efficiency;
        s.learning_rate = learning_rate;
        s.topology_version = topology_version;
        bus_.publish(s);
        notify_observer();
    }

    StageId active_stage() const { return active_stage_.load(); }

private:
    /**
     * Observer loop: every ~50ms, peek at the latest metric from the
     * currently-active training stage and update the adaptive scalars.
     *
     * The current heuristic is intentionally simple — patience nudges,
     * LR-floor nudges, and a stall-detector that raises rewind. The
     * point of this skeleton is to lock in the contract; smarter
     * estimators (Welford, Page–Hinkley change-point detection) plug
     * in here without touching the rest of the runtime.
     */
    void observer_loop() {
        constexpr uint64_t kStallEpochs = 12;
        constexpr double  kStallDelta  = 1e-5;

        std::deque<double> cost_window;

        while (observer_running_.load()) {
            {
                std::unique_lock<std::mutex> lock(observer_mu_);
                // 5 ms fallback only: notify_observer() wakes this
                // immediately on each publish_metric. The short timeout
                // bounds stall/efficiency-drop detection latency on fast
                // models (was 50 ms ~= one epoch of lag at 20 epochs/s).
                observer_cv_.wait_for(lock, std::chrono::milliseconds(5));
            }
            if (!observer_running_.load()) break;

            StageId active = active_stage_.load();
            if (active == StageId::Estimation || active == StageId::Controller) {
                continue;
            }

            auto sample = bus_.latest(active);
            if (sample.epoch == 0 && sample.cost == 0.0) continue;

            cost_window.push_back(sample.cost);
            if (cost_window.size() > kStallEpochs) cost_window.pop_front();

            // Stall detection: improvement < kStallDelta over the window.
            if (cost_window.size() == kStallEpochs) {
                double improvement = cost_window.front() - cost_window.back();
                if (improvement < kStallDelta) {
                    // Nudge patience up (give the active stage more rope)
                    // and raise rewind so the controller can wake an
                    // earlier stage.
                    cfg_.patience.nudge(+1.0);
                    StageId target = (active == StageId::Main)
                        ? StageId::Estimation
                        : StageId::Exploration;
                    request_rewind(target);
                    cost_window.clear();  // reset so we don't re-fire immediately
                }
            }

            // Efficiency feedback: if efficiency is below target,
            // gently lower the saturation threshold so more growth
            // gets approved.
            if (sample.efficiency > 0.0 &&
                sample.efficiency < cfg_.target_efficiency.current() * 0.5) {
                cfg_.exploration_saturation_threshold.nudge(-0.005);
            }

            // ---- Tier-3 dynamic-thresholds refinement ----------------
            // These nudges refine the dataset-derived seeds set by
            // RuntimeAdaptiveConfig::apply_static_config when training
            // signals deviate from what the seeded thresholds predicted.
            // Bounded per-iteration magnitude (<= 1% of each scalar's
            // range) and AdaptiveScalar::nudge clamps to [min, max].
            constexpr size_t kTrendWindow = 8;
            if (cost_window.size() >= kTrendWindow) {
                double trend_start = *(cost_window.end() - kTrendWindow);
                double trend_end   = cost_window.back();
                double trend_slope = (trend_end - trend_start) /
                                     (kTrendWindow * std::max(trend_start, 1e-12));

                if (trend_slope > 0.0) {
                    // Cost drifting up over a recent window: relax the
                    // reward/penalty improvement bar so we don't overreact
                    // to shallow regressions, and tighten min_improvement
                    // so early stop notices real plateaus sooner.
                    cfg_.cost_improvement_threshold.nudge(+1e-4);
                    cfg_.min_improvement.scale(0.95);
                } else if (trend_slope < -0.001) {
                    // Strong improvement: we can afford a stricter reward
                    // gate (less LR thrash) and a slightly looser
                    // min_improvement (don't early-stop on small dips).
                    cfg_.cost_improvement_threshold.nudge(-5e-5);
                    cfg_.min_improvement.scale(1.02);
                }
            }
        }
    }

    MetricsBus& bus_;
    RuntimeAdaptiveConfig& cfg_;
    TopologyLock& lock_;

    std::atomic<StageId> active_stage_{StageId::Controller};
    std::atomic<bool> rewind_{false};
    std::atomic<StageId> rewind_target_{StageId::Estimation};

    std::atomic<bool> observer_running_{false};
    std::thread observer_thread_;
    std::mutex observer_mu_;
    std::condition_variable observer_cv_;
};

}  // namespace runtime
}  // namespace training
}  // namespace dnn
