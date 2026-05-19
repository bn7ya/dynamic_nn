# CLAUDE.md — `include/dnn/cuda/`

## Purpose

GPU acceleration path. Provides `CudaTensor<T>`, the `CudaMemoryPool`
singleton, kernel launch wrappers, and stubs that throw cleanly when
CUDA isn't compiled in. Selected at runtime via `Device::CUDA` and at
compile time via `DNN_ENABLE_CUDA`.

## Files

| File | Role |
|---|---|
| `cuda_tensor.hpp` | `CudaTensor<T>` (GPU storage, host↔device transfer, optional `stream_`). `[invariant]` lifetime/move semantics. |
| `cuda_memory_pool.hpp` | `CudaMemoryPool` singleton. `[invariant]` 20 size classes 256B–256MB; thread-safe via `pool_mutex_`. |
| `cuda_ops.hpp` | Public kernel-launch wrappers (matmul, gemv, elementwise, activations, reductions, optimizers). |
| `cuda_common.hpp` | `KernelConfig` (carries `cudaStream_t`), `ScopedStream` RAII. |
| `cuda_stubs.hpp` | Forward declarations + no-CUDA fallback. `[invariant]` throws clean errors when `!DNN_ENABLE_CUDA`. |

The `.cu` files in `src/cuda/` implement the kernels:

| File | Kernels |
|---|---|
| `cuda_runtime.cu` | Device init, sync, stream / event helpers. |
| `cuda_blas.cu` | gemm, gemv (cuBLAS-backed). `[hot]` |
| `cuda_elementwise.cu` | add, sub, mul, div, scalar variants. `[hot]` |
| `cuda_activations.cu` | ReLU, Sigmoid, Tanh, Softmax fwd+bwd. `[hot]` |
| `cuda_reductions.cu` | sum, max, min, dot. |
| `cuda_optimizers.cu` | SGD / Momentum / Adam / RMSprop weight updates. |
| `cuda_memory_pool.cu` | Pool implementation. `[invariant]` |

## Invariants

- **All GPU allocations go through the singleton pool.**
  `CudaTensor::allocate()` calls `CudaMemoryPool::instance().allocate()`.
  Don't `cudaMalloc` directly; the pool tracks size classes and
  prevents fragmentation. (`cuda_tensor.hpp:188-194`,
  `cuda_memory_pool.hpp`.)
- **The pool has two allocation modes.**
  `CudaMemoryPool::AllocMode::Device` (default) calls `cudaMalloc`
  and pins memory to VRAM. `AllocMode::Managed` calls
  `cudaMallocManaged` so the CUDA driver pages between VRAM and host
  RAM transparently; if host RAM is also exhausted the OS pages out
  to swap (disk). Switching is via `set_alloc_mode()` on the
  singleton, or from Python via `pydnn.set_cuda_memory_mode("managed"
  | "device")`. The mode applies to *future* allocations only —
  existing pooled blocks keep their original allocation type. Set
  the mode before constructing CUDA networks for it to apply
  uniformly. (`cuda_memory_pool.hpp` `AllocMode`,
  `cuda_memory_pool.cu` `device_alloc`.)
- **`CudaMemoryPool` is a thread-safe singleton.** `pool_mutex_` and
  `large_mutex_` cover the free-lists and the large-allocation map.
  Concurrent stage workers are expected to share the pool. Don't
  introduce a per-thread or per-stream pool without revising the
  singleton contract.
- **`CudaTensor::stream_` defaults to `nullptr`.** That is the default
  CUDA stream and matches legacy behaviour. Per-worker streams must
  be opted into explicitly via `set_stream()`. Don't change the
  default.
- **`cuda_stubs.hpp` must keep compiling without `DNN_ENABLE_CUDA`.**
  This is what keeps the CPU-only build green. If you add a function
  here, provide both branches.
- **`CudaTensor` is move-only.** Never make it copyable; copying GPU
  buffers silently is a footgun.

## Maintenance notes

- `CudaMemoryPool::deallocate_async(ptr, stream)` (line 78) currently
  notes "not using async allocation" — it falls through to the sync
  path. Migrating to `cudaMallocAsync` / `cudaFreeAsync` would help
  multi-stream workloads but requires per-stream free-lists or event
  records. See the runtime plan for the design.
- Per-node metric recording now uses
  `CudaTensor::copy_to_host_strided(dst, offset, n)` to D2H exactly the
  first row (`output_size_` floats) instead of the whole (B, O) tensor
  (2.1). If you move metrics to GPU, drop this copy entirely.
- 2.4 (cudaMemcpy2D input slicing → cuBLAS `lda`) and 2.5 (compute/
  copy CUDA streams in the trainer) and 2.2 (`CudaTensor` shared
  device buffer) are deferred: they need a CUDA host to verify and
  2.5 risks silent correctness on an untestable path. Tracked for a
  CUDA-host session.
- The current kernel launches use `KernelConfig` which carries a
  stream, but the trainer mostly passes `nullptr`. Wiring per-worker
  streams from `train_phased_runtime()` is a known follow-up
  (referenced in [`include/dnn/training/runtime/CLAUDE.md`](../training/runtime/CLAUDE.md)).

## Memory & reliability notes

- `CudaMemoryPool::clear()` releases everything. Only call it at
  shutdown or test reset; calling it mid-training will free buffers
  that `CudaTensor` instances still hold pointers into.
- `peak_usage()` is monotonic until `reset_peak_usage()`. Use it to
  flag VRAM ceiling regressions in benchmarks.
- Allocations larger than 256MB bypass the pool and go to
  `large_allocations_`. They're still tracked but free-list reuse
  doesn't apply. Avoid those sizes when you can.
- The `PooledMemory<T>` RAII wrapper at `cuda_memory_pool.hpp:177-230`
  is the recommended way to hand out scratch buffers to kernels;
  prefer it over manual `allocate` / `deallocate`.
- **CUDA error handling:** kernel launches set the device error flag;
  the public wrappers in `cuda_ops.hpp` are responsible for checking
  and throwing. Don't add silent kernel launches.

## Cross-refs

- The Layer's CUDA forward/backward lives in
  [`include/dnn/core/CLAUDE.md`](../core/CLAUDE.md) — that's where
  per-tensor stream binding gets used (when the trainer wires it).
- Per-worker stream policy is described in
  [`include/dnn/training/runtime/CLAUDE.md`](../training/runtime/CLAUDE.md).
- The CPU-only path is in [`include/dnn/simd/CLAUDE.md`](../simd/CLAUDE.md).

## Updating this file

When you add a `.cu` kernel, add a row to the kernels table. When you
change the pool's size classes, allocation strategy, or thread-safety
contract, revise the **Invariants** section. When the
`deallocate_async` TODO is implemented, drop the corresponding
maintenance note. Keep this file ≤200 lines.
