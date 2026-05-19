# CLAUDE.md — `include/dnn/core/`

## Purpose

The numerical and structural core of the network. Defines `Tensor<T>`,
`Node`, `Layer<T>`, and `Network<T>` plus their helpers (activations,
initializers, RNG, device enum). All other subsystems sit on top of
these types. **The soft-topology contract** — mid-training mutations
flip an active mask instead of erasing weights — lives here.

## Files

| File | Role |
|---|---|
| `tensor.hpp` | `Tensor<T>` storage, shape, basic ops. `[invariant]` shape semantics; `[hot]` data layout. |
| `node.hpp` | `Node` (one neuron's metrics + adaptive `EfficiencyWeights` + EMA-smoothed `efficiency_score()`). `[invariant]` `w_alive=0.25` baseline. |
| `layer.hpp` | `Layer<T>` (dense forward/backward, soft `add_nodes` / `remove_nodes`, `compact()`, OpenMP batch loops). `[hot]` `[invariant]` |
| `network.hpp` | `Network<T>` (chain of layers, `mark_layer_inactive`, `compact()`, `topology_version_`). `[invariant]` |
| `activations.hpp` | ReLU / Sigmoid / Tanh / Softmax forward + backward. |
| `initializers.hpp` | He / Xavier / Glorot weight initialisers. |
| `random.hpp` | Seeded RNG used for weight init and perturbations. |
| `device.hpp` | `Device::CPU` / `Device::CUDA` enum. |

The `.cpp` files in `src/core/` provide the explicit template
instantiations for `float` and `double`; behaviour lives in the headers.

## Invariants

- **Soft-delete is the only mid-training removal.**
  `Layer<T>::remove_nodes(indices)` flips `active_mask_[i] = 0`; it does
  **not** erase weights. `Network<T>::mark_layer_inactive(idx)` flips
  `layer_active_ = false`. Hard erasure happens *only* in
  `Layer<T>::compact()` and `Network<T>::compact()`, which are intended
  to run once at end-of-training. Code paths that bypass this are bugs.
  Reference: `layer.hpp` `add_nodes` / `remove_nodes` / `compact`,
  `network.hpp` `compact`.
- **CUDA mirror invalidation runs on every host shape change.**
  `Layer<T>::add_nodes` (in its allocate-new-rows branch) and
  `Layer<T>::compact` reset `gpu_weights_` / `gpu_biases_` so
  `forward_cuda` rebuilds the GPU buffers at the new size on the next
  call. Without this, the rank-1 CUDA path's
  `cuda_copy(*gpu_biases_, gpu_output)` mismatched sizes mid-training
  (the original "dynamic_thresholds=True crashes on GPU" bug). Any
  future code that mutates `weights_` / `biases_` shape on an
  already-uploaded layer must follow the same pattern.
- **CUDA forward honours the active mask, on-device.**
  `Layer<T>::forward_cuda_dev` calls `cuda_apply_mask_broadcast(pre_act,
  gpu_active_mask_)` to zero columns for soft-removed neurons. The mask
  is a 1-D GPU tensor cached in `gpu_active_mask_`, invalidated by
  `add_nodes` / `remove_nodes` / `set_node_active` / `compact` via
  `invalidate_gpu_active_mask_()`. This replaced an earlier D2H → host
  zero → H2D triplet on every layer forward — see `cuda_ops.hpp`
  `cuda_apply_mask_broadcast`. The CPU forward post-zeros inactive
  output columns after the SIMD matmul; both paths produce the same
  values.
- **Device-resident forward/backward chain.** When
  `Network::device_ == Device::CUDA`, `Network::forward` and
  `Network::backward` do one H2D at start and one D2H at end (or none,
  for backward). Each layer's `forward_cuda_dev` / `backward_cuda_dev`
  accepts and returns `cuda::CudaTensor<T>` so intermediates stay on
  device. The Tensor-in/Tensor-out wrappers `Layer::forward_cuda` /
  `backward_cuda` only fire when something calls them in isolation
  (tests, ad-hoc inference); they delegate to `*_cuda_dev`.
- **GPU mirror invalidation is centralised.**
  `Layer<T>::invalidate_gpu_mirrors_()` resets every GPU-resident cache
  (weights, biases, active-mask, forward-pass caches). Called from every
  site that mutates `weights_`/`biases_` shape (add_nodes,
  compact, to_device). For weight-value-only changes (e.g.
  `apply_gradients`) the GPU weight/bias mirrors are reset directly so
  they reupload on the next forward — a future optimisation is to keep
  the optimizer step on device. `invalidate_gpu_active_mask_()` is the
  cheaper variant when only mask values changed.
- **`Node::efficiency_score()` is EMA-smoothed with a warmup.**
  `Node::compute_metrics()` returns the "unknown" default (0.5) until
  `sample_count_ >= kMinSamplesForEfficiency` (16). After warmup, the
  weighted variance/gradient/alive/contribution score is run through
  an EMA (`kEmaAlpha = 0.2`) so a single noisy epoch can't trip the
  layer-manager's shrink path. `smoothed_efficiency_` and
  `smoothed_initialised_` are `mutable` because `compute_metrics()`
  is `const`; same caching idiom as `Layer`'s forward tensors. The
  EMA state is reset by `Node::reset_metrics()` alongside the
  per-node counters — preserve that pairing or a new phase will
  start training with stale smoothing.
- **Per-node metrics are recorded from one row, not the full output.**
  `Layer::forward_cuda_dev` D2Hs only the first batch row of the
  pre-activation (output_size_ floats) for `record_activation`; the
  rest of the pre-activation / output stays GPU-resident. Same for
  `record_first_row_gradient_gpu_` in backward. The host
  `cached_output_` / `cached_pre_activation_` are no longer populated
  on the CUDA path — `compute_metrics().output_variance` will be zero
  unless the network was run on CPU.
- **`Network::insert_layer` / `remove_layer` propagate `device_`.**
  Newly created and rebuilt layers inherit the network's device.
  Earlier versions defaulted to `Device::CPU` here even when the
  network was on CUDA, silently splitting compute across the boundary.
- **Forward pass zeroes inactive outputs, backward zeroes their
  gradients.** `Layer<T>::forward_cpu`, `forward_cuda`, `backward_cpu`,
  `backward_cuda`, and `apply_gradients` all gate on
  `is_node_active(i)`. Don't add code paths that touch
  `weight_gradients_` for inactive rows.
- **`add_nodes` reactivates dormant slots first.** It only allocates
  new rows when no dormant capacity is left. This is what makes
  remove→add round-trip param-preserving; preserve it.
- **Topology version monotonicity.** `topology_version_` only ever
  increments. Every soft mutation bumps it. Workers cache it to detect
  shifts.
- **Dense layouts.** `weights_` is row-major `(output_size,
  input_size)`. `biases_` is `(output_size,)`. Don't change without
  updating every kernel and SIMD op.

## Maintenance notes

- The CUDA backward path in `layer.hpp:386-468` does part of the work
  on GPU and part on CPU (weight-grad accumulation loops). It correctly
  gates on `is_node_active(i)`, but the GPU/CPU split is a known perf
  cost. Worth migrating the accumulation onto GPU eventually; preserve
  the active-mask gating when you do.
- `Network<T>::compact()` includes a fan-in repair pass that **resets
  the rebuilt layer's weights**. This is acceptable because compaction
  is end-of-training; calling it mid-training would silently regress
  weights. Guarded by `Network<T>::training_in_progress_`: the
  trainer's `TrainingInProgressGuard` (RAII in
  `include/dnn/training/trainer.hpp`) flips the flag for the duration
  of `train_phased{,_runtime}`, and `compact()` throws
  `InvalidArgumentException` if called while the flag is set. Tests
  that need to exercise compaction in isolation can pass
  `compact(/*allow_unsafe=*/true)`.
- `Layer<T>::clone()` copies `active_mask_`, `layer_active_`,
  `topology_version_`. New per-layer state must be added here too or
  inherit-based features (`Network::inherit_from`) lose state.
- `Network<T>::clone()` copies `state_`, `topology_version_`, and
  `training_in_progress_`. The last is deliberate: a clone of an
  in-training network must also refuse `compact()`. The clone is for
  inheritance, not for divergent concurrent training.

## Memory & reliability notes

- `Layer<T>` keeps three `Tensor<T>` caches (`cached_input_`,
  `cached_pre_activation_`, `cached_output_`). They're sized to the
  most recent forward pass. Don't replace with per-call allocations —
  that would add allocator pressure to every batch.
- `Network::clone()` deep-copies all layers' tensors. It's not in the
  hot path but is used by `Network::inherit_from`; expect it to be
  expensive on large nets.
- `Layer::forward_cpu` and `Layer::backward_cpu` (rank-2 path) now
  route through `simd::SIMDOps<T>::matmul` instead of the old manual
  nested loops. The SIMD matmul provides AVX/AVX2/SSE vectorisation
  *and* OpenMP parallelism on the outer i-block (see
  `include/dnn/simd/CLAUDE.md`). The previous per-batch
  `#pragma omp parallel for` at the layer level was removed because
  nesting it inside the matmul's OpenMP team causes oversubscription.
  The `kOpenMpBatchThreshold = 4` constant is retained for potential
  future use but is no longer referenced by layer code. Forward uses
  a lazily-rebuilt transposed-weights cache `weights_t_cached_` of
  shape `(input_size, output_size)`; backward materialises a small
  `grad_activation^T` scratch (output_size × batch_size) so the
  weight-gradient matmul can run as `dW = G^T @ X`.
- `Layer::reset_node_metrics()` zeroes per-node counters but **not** the
  `active_mask_`. Don't conflate the two.

## Cross-refs

- Soft topology is consumed by [`include/dnn/dynamics/`](../dynamics/CLAUDE.md)
  (`LayerManager` decides when to mutate) and
  [`include/dnn/training/`](../training/CLAUDE.md) (the trainer holds
  the network).
- The CUDA path under `forward_cuda` / `backward_cuda` calls into
  [`include/dnn/cuda/`](../cuda/CLAUDE.md).
- OpenMP rules are described in
  [`include/dnn/simd/CLAUDE.md`](../simd/CLAUDE.md).

## Updating this file

When you add a file to `include/dnn/core/` or `src/core/`, append a row
to the file table. When you change any of the soft-topology invariants
or the dense-layout contract, edit the **Invariants** section in the
same commit. Keep this file ≤200 lines.
