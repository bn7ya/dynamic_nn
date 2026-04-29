#pragma once

#include <shared_mutex>

namespace dnn {
namespace training {
namespace runtime {

/**
 * Read/write coordinator for the network's topology under concurrent
 * stage workers.
 *
 * Workers acquire a shared (read) lock around forward and backward — they
 * may run in parallel as long as nobody is mutating the graph. The
 * controller / layer manager takes the exclusive (write) lock to flip
 * active masks or rewire layers; combined with the soft-delete semantics
 * in Layer<T>, this means no in-flight worker ever observes a half-applied
 * topology change.
 *
 * Header-only and tiny on purpose: any stage that wants to participate
 * just composes a TopologyLock by reference rather than instantiating its
 * own synchronisation.
 */
class TopologyLock {
public:
    using ReadGuard = std::shared_lock<std::shared_mutex>;
    using WriteGuard = std::unique_lock<std::shared_mutex>;

    ReadGuard read_lock() { return ReadGuard(mu_); }
    WriteGuard write_lock() { return WriteGuard(mu_); }

private:
    std::shared_mutex mu_;
};

}  // namespace runtime
}  // namespace training
}  // namespace dnn
