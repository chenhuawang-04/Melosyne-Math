# DynamicBitSet / BitOps Optimization Report (2026-04-23)

## Scope

This round focused on **performance-oriented refactor** for:

- `include/fast_math/bitset_dynamic.h`
- `include/fast_math/bit_ops_core_simd.h`
- `include/fast_math/bit_ops_count_simd.h`
- Added benchmark coverage: `bench/suites/bench_bitset_datapath.cpp`

Baseline source: `docs/benchmarks/BITSET_BASELINE_2026-04-23.md`  
Post-optimization raw logs:
- `docs/benchmarks/postopt_datastructure_2026-04-23_v3.txt`
- `docs/benchmarks/postopt_bitops_2026-04-23_v3.txt`

## Main Refactor Decisions

1. **DynamicBitSet allocator redesign**
   - Added capacity-aware storage reuse (`capacity_words_`), avoiding repeated free/alloc in assignment paths.
   - Added `reserveBits`, `resize`, `shrinkToFit`.
   - Added adaptive alignment policy:
     - large remote buffers (`>=256 words`) use 32-byte alignment (SIMD-friendly),
     - smaller remote buffers use natural 8-byte alignment (lower allocation overhead).
   - Added accurate alignment-aware deallocation bookkeeping.

2. **Core BitOps SIMD path hardening**
   - Rebuilt AVX2 core ops with safe aligned/unaligned dual paths:
     - `bitwiseAndSimd`, `bitwiseOrSimd`, `bitwiseXorSimd`, `copySimd`, `equalSimd`.
   - Added public optimized wrappers:
     - `bitwiseOrOptimized`, `bitwiseXorOptimized`, `copyOptimized`, `equalOptimized`.
   - Tuned dispatch thresholds to avoid over-eager SIMD on medium kernels.

3. **Count SIMD tuning**
   - Fixed `allSimd` alignment handling (avoid aligned-load-only assumption).
   - Tuned `popcountOptimized` threshold to prevent mid-size regressions.

## Before vs After (Key Metrics)

### DataStructure suite

| Case | Baseline ns/op | Post ns/op | Change |
|---|---:|---:|---:|
| random set/reset/test (16K) | 2.968 | 2.946 | **+0.74%** |
| setAll/resetAll/flipAll/popcount (64K) | 326.562 | 325.168 | **+0.43%** |
| copy/move (64K) | 360.872 | 277.993 | **+22.97%** |
| lifecycle (4K) | 212.741 | 211.676 | **+0.50%** |

### BitOps suite (optimized vs scalar)

| Case | Baseline | Post | Delta |
|---|---:|---:|---:|
| dynamic and+popcount (64K) | 0.843x | 0.990x | **+17.4 pp** |
| dynamic and+count+rank (4K) | 1.063x | 0.973x | -9.0 pp |
| static and+count+rank (4K) | 0.944x | 1.010x | **+6.6 pp** |

## Validation

- `fast_math_tests`: **34/34 passed**
- Benchmark command set:
  - `.\build_perf_baseline\fast_math_bench.exe --filter=DataStructure --repeats=11 --warmup=4 --measure=10 --min-ms=20`
  - `.\build_perf_baseline\fast_math_bench.exe --filter=BitOps --repeats=11 --warmup=4 --measure=10 --min-ms=20`

## Next Performance Step

The remaining weak spot is the **dynamic 4K mixed kernel** (`0.973x`).  
Recommended next step: add per-kernel auto-tuning thresholds for `rankOptimized` + `flipRangeOptimized` interaction under clang/AVX2.

