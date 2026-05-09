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
| `node.hpp` | `Node` (one neuron's metrics + adaptive `EfficiencyWeights`). `[invariant]` `w_alive=0.25` baseline. |
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
- **CUDA forward honours the active mask.** `Layer<T>::forward_cuda`
  zeros inactive-node columns of the pre-activation tensor (via
  `zero_inactive_components`) and re-uploads the masked tensor to the
  GPU before activation. This makes `remove_nodes` actually shrink the
  network on CUDA — without the masking step the dense `gemv` / `gemm`
  produces non-zero outputs for soft-removed neurons and the next
  layer happily consumes them. CPU's per-node forward already gates on
  `is_node_active(i)`; both paths now agree.
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

## Memory & reliability notes

- `Layer<T>` keeps three `Tensor<T>` caches (`cached_input_`,
  `cached_pre_activation_`, `cached_output_`). They're sized to the
  most recent forward pass. Don't replace with per-call allocations —
  that would add allocator pressure to every batch.
- `Network::clone()` deep-copies all layers' tensors. It's not in the
  hot path but is used by `Network::inherit_from`; expect it to be
  expensive on large nets.
- The OpenMP `#pragma omp parallel for` directives in `forward_cpu`
  (batch loop) and `backward_cpu` (grad-input loop) are gated by
  `if (batch_size > 4)` so small batches don't pay the team-spawn
  cost. Preserve that guard.
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
