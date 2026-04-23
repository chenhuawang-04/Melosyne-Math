# Bitset Baseline Metrics (Pre-Optimization)

Date: **2026-04-23 16:37:41 +08:00**  
Repo: `fast_math` (pre-optimization snapshot for DynamicBitSet/BitOps refactor)

## Environment

- CPU: `12th Gen Intel(R) Core(TM) i7-12700H` (14C/20T, Max 2300 MHz)
- Compiler: `clang++ 22.1.0` (`x86_64-pc-windows-msvc`)
- Build system: `CMake 4.0.4` + `Ninja`
- Build dir: `build_perf_baseline`
- Configure flags:
  - `-DCMAKE_C_COMPILER=clang`
  - `-DCMAKE_CXX_COMPILER=clang++`
  - `-DFAST_MATH_BUILD_MODULES=OFF`

## Commands

```powershell
cmake -S . -B build_perf_baseline -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DFAST_MATH_BUILD_MODULES=OFF
cmake --build build_perf_baseline -j 8 --target fast_math_bench
.\build_perf_baseline\fast_math_bench.exe --filter=DataStructure --repeats=11 --warmup=4 --measure=10 --min-ms=20
.\build_perf_baseline\fast_math_bench.exe --filter=BitOps --repeats=11 --warmup=4 --measure=10 --min-ms=20
```

Raw logs:
- `docs/benchmarks/baseline_datastructure_2026-04-23.txt`
- `docs/benchmarks/baseline_bitops_2026-04-23.txt`

## Baseline Results

### DataStructure suite

| Case | Median ns/op | Mops/s |
|---|---:|---:|
| dynamic bitset random set/reset/test (16K bits) | 2.968 | 336.880 |
| dynamic bitset setAll/resetAll/flipAll/popcount (64K bits) | 326.562 | 3.062 |
| dynamic bitset copy/move (64K bits) | 360.872 | 2.771 |
| dynamic bitset lifecycle (4K bits) | 212.741 | 4.701 |

### BitOps suite

| Case | Backend | Median ns/op | Speedup vs scalar |
|---|---|---:|---:|
| dynamic bitset and+popcount (64K bits) | scalar | 241.054 | 1.000x |
| dynamic bitset and+popcount (64K bits) | optimized | 285.805 | 0.843x |
| dynamic bitset and+count+rank kernel (4K bits) | scalar | 14.608 | 1.000x |
| dynamic bitset and+count+rank kernel (4K bits) | optimized | 13.737 | 1.063x |
| static bitset and+count+rank kernel (4K bits) | scalar | 14.782 | 1.000x |
| static bitset and+count+rank kernel (4K bits) | optimized | 15.658 | 0.944x |

## Current Bottleneck Signals (Baseline Interpretation)

1. DynamicBitSet large-kernel optimized path regresses (`0.843x`) vs scalar.
2. DynamicBitSet copy/move and lifecycle are relatively expensive.
3. Static and dynamic optimized throughput both show unstable/noisy gains near 4K-bit kernel.

