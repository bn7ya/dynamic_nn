# CLAUDE.md — `include/dnn/memory/`

## Purpose

CPU-side allocator infrastructure: a singleton size-class pool, an
aligned allocator suitable for SIMD, and a resource monitor that
tracks peak usage. The GPU equivalent lives in
[`include/dnn/cuda/`](../cuda/CLAUDE.md).

## Files

| File | Role |
|---|---|
| `pool_allocator.hpp` | `PoolAllocator` singleton: 16 size classes 64B–2MB, thread-safe via `pool_mutex_`. `[invariant]` |
| `aligned_allocator.hpp` | `aligned_alloc(size, alignment)` / `aligned_free()` for cache-line + SIMD-boundary alignment. `[invariant]` |
| `resource_monitor.hpp` | Peak memory + per-component byte counters. |

`src/memory/` provides:
- `pool_allocator.cpp` — pool implementation.
- `resource_monitor.cpp` — counter implementation.

## Invariants

- **`PoolAllocator` is a singleton with a single mutex.** Concurrent
  CPU stage workers share it. Don't shard without revising the
  thread-safety contract. (`pool_allocator.hpp:220` mutex.)
- **Aligned allocation is a hard requirement for SIMD ops.** AVX/SSE
  paths assume 32-byte alignment. Don't replace
  `aligned_allocator.hpp` with `std::malloc` — it will silently
  segfault on aligned loads.
- **The pool's size classes are 64B → 2MB.** Allocations larger than
  2MB go to the system allocator. If you change these bounds, audit
  every `Tensor` / `Layer` allocation to see whether it lands above
  the new ceiling.

## Maintenance notes

- `PoolAllocator` doesn't release back to the OS until destruction.
  That's intentional — it keeps the pool warm. If you ever care about
  rss returning to baseline mid-process (e.g. a long-running server),
  add an explicit `PoolAllocator::shrink_to_fit()` rather than
  `free()`-ing inside the destructor.
- `ResourceMonitor` is not threaded into hot paths; it's mostly useful
  for benchmarks. Don't pay extra cost to make it precise unless a
  benchmark asks for it.
- The aligned allocator wraps `std::aligned_alloc` (POSIX) or
  `_aligned_malloc` (Windows). If you add a new platform, mirror the
  existing pattern; don't introduce a third allocator path.

## Memory & reliability notes

- Tensors do not currently route through `PoolAllocator` for their
  backing storage — `Tensor<T>` uses `std::vector<T>` internally. The
  pool is used by callers that need scratch buffers or arena-style
  allocation. Don't break this layering by trying to make `Tensor`
  use the pool unless you're prepared to handle alignment + size
  guarantees throughout.
- `peak_usage()` is monotonic until reset. Treat it like a high-water
  mark for regression tests.
- The pool's per-size-class free-list is a `std::vector<void*>`.
  Don't drain it from one thread while another is allocating — the
  mutex covers it but starvation is still possible if you call
  `shrink_to_fit` (when added) under contention.

## Cross-refs

- GPU allocator: [`include/dnn/cuda/CLAUDE.md`](../cuda/CLAUDE.md)
  (`CudaMemoryPool`).
- SIMD ops that consume the alignment guarantee:
  [`include/dnn/simd/CLAUDE.md`](../simd/CLAUDE.md).
- `Tensor<T>` storage decisions are described in
  [`include/dnn/core/CLAUDE.md`](../core/CLAUDE.md).

## Updating this file

When you change the size-class bounds or the alignment requirement,
revise the **Invariants** section in the same commit. When a new
allocator path is added (e.g. pinned-host memory for CUDA staging),
add it as a new file row. Keep this file ≤140 lines.
