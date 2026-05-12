# CLAUDE.md — `include/dnn/simd/`

## Purpose

CPU vectorized arithmetic. A runtime-detected dispatch picks the best
implementation for the host CPU (AVX-512 → AVX2 → AVX → SSE → scalar).
OpenMP `#pragma omp parallel for` directives also live here, gated by
`DNN_HAS_OPENMP`. This is the CPU fallback path used whenever the
network is on `Device::CPU` and the layer's helpers reach for a SIMD
op via `Tensor`.

## Files

| File | Role |
|---|---|
| `simd_detect.hpp` | Runtime CPU feature detection (`SIMDLevel` enum). |
| `simd_ops.hpp` | `SIMDOps<T>` abstract interface + `ScalarOps<T>` portable fallback. `[hot]` `[fallback]` |
| `avx_ops.hpp` | AVX/AVX2/AVX512 implementation. `[hot]` |
| `sse_ops.hpp` | SSE implementation. `[hot]` `[fallback]` |

`src/simd/`:
- `simd_detect.cpp` — implements the detection.
- `simd_ops.cpp` — runtime dispatch + factory selecting the best `SIMDOps<T>`.

## Invariants

- **`SIMDOps::create()` must return a non-null pointer.**
  `ScalarOps<T>` is the always-available fallback. Don't introduce
  a code path that returns nullptr from the factory.
- **Element-wise ops process exactly `n` elements in `c[0..n-1]`.**
  No off-by-one writes outside the destination range. SIMD
  implementations handle the tail (last `n % VECTOR_WIDTH` elements)
  with scalar fallbacks; preserve that pattern when adding new ops.
- **Pointers passed to SIMD ops are aligned for AVX/SSE.** Tensors
  back this via [`include/dnn/memory/CLAUDE.md`](../memory/CLAUDE.md).
  Misaligned pointers will fault; never dispatch to AVX paths from
  unaligned scratch buffers.
- **`#pragma omp` directives are guarded by `#ifdef DNN_HAS_OPENMP`.**
  The CMake config defines this when `find_package(OpenMP)` succeeds.
  Don't drop the guard; some platforms ship without OpenMP and the
  pragmas are then no-ops.

## Maintenance notes

- **OpenMP is on every matmul.** `ScalarOps<T>::matmul`
  (`simd_ops.hpp:133-180`), `AVXOps<float>::matmul`
  (`avx_ops.hpp:206-275`) and `SSEOps<float>::matmul`
  (`sse_ops.hpp:204-262`) all carry `#pragma omp parallel for
  schedule(static) if (m > 32)` on both the zero-output loop and the
  outer i0-block. Distinct i0 blocks write disjoint output rows, so
  there's no race. The `if (m > 32)` guard skips the team-spawn cost
  on small matrices.
- **`Layer::forward_cpu` / `Layer::backward_cpu` (rank-2) now call
  `SIMDOps::matmul` directly.** Layer-level batch-loop pragmas were
  removed to avoid nesting OpenMP teams inside the matmul. The layer
  is now the primary caller of `SIMDOps::matmul` from the trainer hot
  path. Pre-existing layer-level OpenMP guards (`kOpenMpBatchThreshold
  = 4`) are no longer wired in; see `include/dnn/core/CLAUDE.md`
  maintenance notes.
- **Thread count is controllable from Python.**
  `python/pydnn/_bindings.cpp` exposes
  `pydnn.set_num_threads(n)` / `pydnn.get_num_threads()` /
  `pydnn.openmp_available()`, which route to `omp_set_num_threads` /
  `omp_get_max_threads`. `OMP_NUM_THREADS` env var continues to work
  via libgomp default.
- `simd_detect.cpp` runs once at load time. Don't make it stateful
  per-thread.

## Memory & reliability notes

- These ops never allocate. They write into pre-allocated buffers
  passed in by the caller. Don't introduce internal allocations on
  the hot path.
- `dot()` and `sum()` reductions accumulate into a single scalar.
  For very long vectors with `float`, watch for precision loss; the
  AVX path uses pairwise summation in places. Keep that property
  if you rewrite.
- `ScalarOps::matmul` zeroes `c` before accumulating — preserve that;
  callers expect `C := A @ B`, not `C += A @ B`.

## Cross-refs

- Layer forward/backward calls into these via `Tensor` ops:
  [`include/dnn/core/CLAUDE.md`](../core/CLAUDE.md).
- Alignment requirement is owned by
  [`include/dnn/memory/CLAUDE.md`](../memory/CLAUDE.md).
- The GPU equivalent kernels live in
  [`include/dnn/cuda/CLAUDE.md`](../cuda/CLAUDE.md).

## Updating this file

When you add a new SIMD implementation (e.g. ARM NEON), add a row to
the file table and document its detection condition. When you add an
OpenMP `#pragma`, document it in **Maintenance notes** so future
work can reason about nesting. Keep this file ≤160 lines.
