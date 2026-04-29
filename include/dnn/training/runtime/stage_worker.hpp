#pragma once

#include "metrics_bus.hpp"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace dnn {
namespace training {
namespace runtime {

/**
 * Lifecycle state of a stage worker.
 *
 * - Standby:  thread is alive, waiting on cv_ for activate(). Cheap to
 *             resume because all per-stage state (LR, emotional state,
 *             counters) is held on the worker object itself.
 * - Active:   thread is running its training loop. step() is being called
 *             repeatedly until the worker decides it's done or the
 *             controller tells it to pause.
 * - Finished: stage has reached its terminal state (e.g. target reduction
 *             achieved, max epochs hit). Won't re-enter Active without
 *             an explicit reset() from the controller.
 *
 * The fourth lifecycle event is shutdown(), which exits the loop and
 * joins the thread; only the destructor does that.
 */
enum class StageState : uint8_t {
    Standby  = 0,
    Active   = 1,
    Finished = 2
};

/**
 * Long-lived worker that owns one of the four training stages.
 *
 * Each subclass overrides step() with that stage's per-epoch body and
 * should_continue() with its termination predicate. The controller drives
 * the worker via activate() / suspend() / reset(); workers report progress
 * by pushing MetricSamples to the shared MetricsBus.
 *
 * The worker never destroys per-stage state on suspend, so the controller
 * can move freely between Active and Standby without losing learning rate,
 * emotional counters, or partial cost histories.
 */
class StageWorker {
public:
    StageWorker(StageId id, MetricsBus& bus)
        : id_(id), bus_(bus) {}

    virtual ~StageWorker() {
        shutdown();
    }

    StageWorker(const StageWorker&) = delete;
    StageWorker& operator=(const StageWorker&) = delete;

    /**
     * Spawn the worker thread and park it in Standby.
     */
    void start() {
        if (thread_.joinable()) return;
        shutdown_.store(false);
        thread_ = std::thread([this] { run_loop(); });
    }

    /**
     * Wake the worker into Active. Returns immediately; the worker thread
     * picks up the change at the next condition-variable wakeup.
     */
    void activate() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (state_ == StageState::Finished) return;
            state_ = StageState::Active;
        }
        cv_.notify_all();
    }

    /**
     * Park the worker back in Standby. Does not interrupt an in-flight
     * step() call; the worker will check the state at the next loop
     * iteration.
     */
    void suspend() {
        std::lock_guard<std::mutex> lock(mu_);
        if (state_ == StageState::Active) {
            state_ = StageState::Standby;
        }
    }

    /**
     * Reset a Finished worker back to Standby so the controller can
     * restart it. Subclasses can override on_reset() to also reset
     * stage-local counters (epoch index, patience, etc.).
     */
    void reset() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            state_ = StageState::Standby;
        }
        on_reset();
        cv_.notify_all();
    }

    /**
     * Tell the worker it's done for good. Subsequent activate() calls are
     * no-ops; the controller uses this when target metrics are reached.
     */
    void finish() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            state_ = StageState::Finished;
        }
        cv_.notify_all();
    }

    /**
     * Stop the worker thread permanently. Called from the destructor;
     * controllers normally don't invoke this directly.
     */
    void shutdown() {
        if (!thread_.joinable()) return;
        {
            std::lock_guard<std::mutex> lock(mu_);
            shutdown_.store(true);
            state_ = StageState::Finished;
        }
        cv_.notify_all();
        thread_.join();
    }

    StageState state() const {
        std::lock_guard<std::mutex> lock(mu_);
        return state_;
    }

    StageId id() const { return id_; }

    /**
     * Block until the worker leaves Active state. Used by tests and by
     * controllers that want to drive workers in lock-step.
     */
    void wait_until_idle() const {
        std::unique_lock<std::mutex> lock(mu_);
        cv_idle_.wait(lock, [this] {
            return state_ != StageState::Active || shutdown_.load();
        });
    }

protected:
    /**
     * One unit of stage work. Typically a single training epoch.
     * Implementations must be re-entrant across suspend/activate cycles
     * — i.e. they keep their own per-stage epoch counter.
     */
    virtual void step() = 0;

    /**
     * Predicate evaluated after every step(); when it returns false the
     * worker auto-suspends back to Standby. Returning true indicates that
     * the stage has more work to do if reactivated.
     */
    virtual bool should_continue() = 0;

    /**
     * Hook called from reset() so subclasses can clear stage-local
     * counters. Default does nothing.
     */
    virtual void on_reset() {}

    /**
     * Convenience: emit a metric sample tagged with this stage's id.
     */
    void publish(uint64_t epoch, double cost, double efficiency,
                 double lr, uint64_t topology_version) {
        MetricSample sample;
        sample.stage = id_;
        sample.epoch = epoch;
        sample.cost = cost;
        sample.efficiency = efficiency;
        sample.learning_rate = lr;
        sample.topology_version = topology_version;
        bus_.publish(sample);
    }

    MetricsBus& bus() { return bus_; }

private:
    void run_loop() {
        while (true) {
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [this] {
                    return state_ == StageState::Active || shutdown_.load();
                });
                if (shutdown_.load()) return;
            }

            // Run while Active and the subclass has more work to do.
            while (true) {
                step();

                bool exit_loop = false;
                {
                    std::lock_guard<std::mutex> lock(mu_);
                    if (shutdown_.load() || state_ != StageState::Active) {
                        exit_loop = true;
                    }
                }
                if (exit_loop) break;

                if (!should_continue()) {
                    std::lock_guard<std::mutex> lock(mu_);
                    state_ = StageState::Standby;
                    break;
                }
            }
            cv_idle_.notify_all();
        }
    }

    StageId id_;
    MetricsBus& bus_;

    mutable std::mutex mu_;
    std::condition_variable cv_;
    mutable std::condition_variable cv_idle_;
    std::thread thread_;

    StageState state_ = StageState::Standby;
    std::atomic<bool> shutdown_{false};
};

}  // namespace runtime
}  // namespace training
}  // namespace dnn
