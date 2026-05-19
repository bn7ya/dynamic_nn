# CLAUDE.md — tests/

> C++ unit tests. Read the root `CLAUDE.md` first.

## Layout

| File | Role |
|---|---|
| `cpp/test_tensor.cpp` | `Tensor<T>` construction, indexing, transpose, `view`/`const_view` |
| `cpp/test_network.cpp` | `Network<T>` forward, `clone`, weight-preserving rebuild |
| `cpp/test_simd.cpp` | `SIMDOps<T>` add/fill parity vs scalar reference |
| `cpp/test_pool_allocator.cpp` | `PoolAllocator` size-class correctness (added in Tier 1) |
| `cpp/test_trainer.cpp` | gradient clip, perturbation, optimizer wiring (added in Tier 1) |
| `cpp/test_metrics.cpp` | early-stopping deque, `MetricsBus` ring (added in Tier 2) |

## How to run

```bash
cmake -S . -B build_test -DDNN_ENABLE_CUDA=OFF -DDNN_BUILD_PYTHON=OFF -DDNN_BUILD_EXAMPLES=OFF
cmake --build build_test -j
ctest --test-dir build_test --output-on-failure
```

## Invariants

- GoogleTest is acquired via `FetchContent` (pinned `release-1.12.1`) when no
  system GTest is found — see the `DNN_BUILD_TESTS` block in the root
  `CMakeLists.txt`. Network egress to github is required for the first
  configure.
- Every new test file must be added to the `dnn_tests` `add_executable` list
  in the root `CMakeLists.txt` in the same commit.
- Tests link `dnn_core` + `GTest::gtest_main`; no `main()` is written by hand.
- CUDA-only behavior is not exercised here (CPU-only build). Tests assert
  correctness/equivalence, never wall-clock timing.
