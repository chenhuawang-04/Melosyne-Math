# Repository Guidelines

## Project Structure & Module Organization
- `include/fast_math/`: header-only core library (scalar + SIMD paths). Keep feature headers grouped by domain (`vec*`, `mat*`, `aabb*`, `bit_ops*`, `power*`, `trig*`).
- `test/`: unified correctness framework (`framework/` + `suites/`). Each suite focuses on one subsystem.
- `bench/`: unified performance framework (`framework/`, `adapters/`, `suites/`) with optional GLM/Eigen/DirectXMath comparisons.
- `docs/`: design and implementation notes.
- `build/`: generated artifacts only; do not commit.

## Build, Test, and Development Commands
```powershell
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build -j 8
.\build\fast_math_tests.exe
.\build\fast_math_bench.exe --quick
ctest --test-dir build --output-on-failure
```
Use `-DFAST_MATH_BUILD_TESTS=OFF` or `-DFAST_MATH_BUILD_BENCHMARKS=OFF` to trim local builds.

## Coding Style & Naming Conventions
- C++20, 4-space indentation, minimal includes, no exceptions in hot paths.
- Public API stays in `MMath`; benchmark/test helper code stays in local namespaces (`fmbench`, `fmtest`).
- Types: PascalCase (`Vec4`, `Mat4`); functions: lowerCamelCase (`vec3Normalize`, `mat4MulVec4`); macros/constants: UPPER_SNAKE_CASE.
- Preserve `MMATH_FORCE_INLINE`, `noexcept`, and existing SIMD dispatch conventions.

## Testing Guidelines
- Add tests under `test/suites/test_<feature>.cpp`; keep one clear purpose per file.
- Prefer deterministic randomized checks (fixed seeds) + tolerance-based assertions for floating point.
- For cross-library parity, gate external checks with `FM_HAVE_GLM/FM_HAVE_EIGEN/FM_HAVE_DIRECTXMATH`.
- Before PR: run `fast_math_tests` and at least `fast_math_bench --quick`.

## Commit & Pull Request Guidelines
- Follow Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `bench:`).
- Keep commits scoped (one subsystem per commit).
- PRs should include: intent, key files changed, exact build/test/bench commands, and concise result summary.
