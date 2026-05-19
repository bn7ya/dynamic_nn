#pragma once

#include <cstdint>

namespace dnn {
namespace training {

/**
 * TrainStrategy — extraction interface for 3.12.
 *
 * `Trainer::train()` and `Trainer::train_phased()` currently duplicate
 * batching / normalization / callback / history logic with small
 * variations. The intended cleanup is to express each as a thin wrapper
 * around a strategy implementing the hooks below, so the shared loop
 * lives in one place.
 *
 * STATUS: interface only — deliberately NOT wired yet.
 *
 * Rationale (see include/dnn/training/CLAUDE.md): the legacy
 * `train_phased` body is the bit-for-bit reproducibility baseline for
 * every saved model and notebook, and it threads 30+ formerly-static
 * constants plus the one-rewind policy and the RAII
 * TrainingInProgressGuard. Proving the extraction is byte-for-byte
 * identical requires the end-to-end Python/`_dnn_core` cost-trajectory
 * smoke, which cannot be built in the current environment (no
 * pybind11) — and a CUDA path that cannot be exercised here. Landing
 * the rewrite unverified would risk a silent reproducibility
 * regression, which the maintenance contract explicitly forbids.
 *
 * When a pybind/CUDA-capable environment is available: implement a
 * DefaultPhasedStrategy whose hooks reproduce the current
 * `train_phased` body verbatim, gate the swap behind a seeded
 * cost-trajectory parity test (legacy vs strategy) on a small dataset,
 * then make `train_phased` delegate to it.
 */
template <typename T>
class TrainStrategy {
public:
    virtual ~TrainStrategy() = default;

    virtual void on_epoch_start(uint64_t epoch) = 0;
    virtual void on_batch(uint64_t epoch, uint64_t batch) = 0;
    virtual void on_epoch_end(uint64_t epoch, double cost,
                              double efficiency) = 0;
    virtual bool should_continue(uint64_t epoch) = 0;
};

}  // namespace training
}  // namespace dnn
