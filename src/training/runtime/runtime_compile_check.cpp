// Translation unit that exists only to force compilation of the runtime
// header building blocks. Replace with real worker/controller bodies once
// they're wired up.
#include "dnn/training/runtime/adaptive_config.hpp"
#include "dnn/training/runtime/metrics_bus.hpp"
#include "dnn/training/runtime/stage_controller.hpp"
#include "dnn/training/runtime/stage_worker.hpp"
#include "dnn/training/runtime/topology_lock.hpp"

namespace dnn {
namespace training {
namespace runtime {

// Force template/struct instantiation so any latent type errors surface
// at link time rather than at first use.
namespace {
[[maybe_unused]] void instantiate_runtime_types() {
    AdaptiveScalar lr{0.01, 1e-6, 1.0};
    (void)lr.current();
    RuntimeAdaptiveConfig cfg;
    cfg.reset_to_defaults();
    MetricsBus bus(64);
    MetricSample s;
    bus.publish(s);
    (void)bus.snapshot();
    (void)bus.latest(StageId::Main);
    TopologyLock tl;
    auto rg = tl.read_lock();
    (void)rg;

    StageController ctrl(bus, cfg, tl);
    (void)ctrl.active_stage();
    ctrl.publish_metric(StageId::Exploration, 0, 0.0, 0.0, 0.0, 0);
    ctrl.request_rewind(StageId::Estimation);
    ctrl.clear_rewind();
}
}  // namespace

}  // namespace runtime
}  // namespace training
}  // namespace dnn
