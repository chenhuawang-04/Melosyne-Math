# Melosyne Math

Melosyne Math is a high-performance C++20 math library built around:

- strict POD data types
- free-function APIs
- SIMD-first implementations
- deterministic tests and benchmarks
- optional C++20 module delivery

It is designed for real-time workloads that need fast and predictable vector,
matrix, AABB, bitset, and scalar math primitives.

---

## Key Features

- **Vectors**: `Vec2`, `Vec3`, `Vec4`, plus normalize/dot/cross helpers
- **Matrices**: `Mat3`, `Mat4`, canonical multiply/transform/projection APIs
- **Quaternions**: `Quat` with Hamilton multiply, rotation, SLERP/NLERP,
  axis-angle and Euler conversion, and a world-frame integrator
- **AABBs**: 2D and 3D axis-aligned bounding box helpers
- **BitOps**: static and dynamic bitsets, popcount/count/rank/range/scans
- **Scalar math**: `sqrt`, `inverseSqrt`, `reciprocal`, `pow`, `trig`
- **SIMD paths**: scalar, SSE, AVX, and FMA specializations where available
- **Module support**: real `.cppm` interfaces under `modules/fast_math/`
- **Validation**: unified correctness tests and performance benchmarks

Optional cross-library checks can target:

- DirectXMath
- Eigen
- GLM

---

## Repository Layout

```text
include/fast_math/      Public header API
include/fast_math/detail/  Shared implementation details for header/module parity
modules/fast_math/      C++20 module interface units (.cppm)
test/                   Unified correctness framework
bench/                  Unified benchmark framework
docs/                   Design and implementation notes
build/                  Generated artifacts only; do not commit
```

---

## Build Requirements

- CMake 3.20 or newer
- A C++20 compiler
- Ninja
- Clang recommended for this repository

Module builds require **CMake 3.28+**. When CMake is older, module targets are
automatically disabled during configuration.

---

## Quick Start

### Configure and build with Clang + Ninja

```powershell
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build -j 8
```

### Run tests

```powershell
ctest --test-dir build --output-on-failure
```

### Run benchmarks

```powershell
.\build\fast_math_bench.exe --quick
```

---

## Module Build

If your environment supports CMake 3.28 or newer, you can also build the module
interfaces:

```powershell
cmake -S . -B build_modules -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build_modules -j 8 --target fast_math_modules_smoke
```

When module support is unavailable, set:

```powershell
-DFAST_MATH_BUILD_MODULES=OFF
```

---

## Usage

### Header-based usage

```cpp
#include <fast_math/vec3.h>
#include <fast_math/mat4.h>
#include <fast_math/quat.h>

MMath::Vec3 a{1.0f, 2.0f, 3.0f};
MMath::Vec3 b{4.0f, 5.0f, 6.0f};
auto sum = MMath::vec3Add(a, b);

MMath::Mat4 m = MMath::mat4Identity();
MMath::Vec4 v{1.0f, 2.0f, 3.0f, 1.0f};
auto transformed = MMath::mat4MultiplyVec4(m, v);

// Quaternion: build a 90-degree yaw, apply it to +X, get +Y.
const MMath::Quat yaw = MMath::quatFromAxisAngle(
    MMath::Vec3{0.0f, 0.0f, 1.0f}, 1.5707963f);
MMath::Vec3 rotated = MMath::quatRotateVec3(yaw, MMath::Vec3{1.0f, 0.0f, 0.0f});

// Physics tick: world-frame angular velocity integration with renormalize.
MMath::Quat orientation = MMath::quatIdentity();
MMath::Vec3 omega{0.0f, 0.0f, 1.0f};   // 1 rad/s about Z
orientation = MMath::quatIntegrate(orientation, omega, 1.0f / 240.0f);
```

### C++20 module usage

```cpp
import fast_math.all;

MMath::Vec3 a{1.0f, 2.0f, 3.0f};
MMath::Vec3 b{4.0f, 5.0f, 6.0f};
auto sum = MMath::vec3Add(a, b);
```

You can also import smaller units when you want tighter dependencies:

```cpp
import fast_math.types;
import fast_math.vec3;
import fast_math.mat4;
import fast_math.quat;
import fast_math.bit_ops;
```

---

## API Conventions

- The primary public namespace is **`MMath`**
- Bitset utilities are bridged through **`Melosyne::BitOps`**
- Canonical names prefer the `Multiply` spelling:
  - `mat3Multiply`
  - `mat3MultiplyAffine`
  - `mat4Multiply`
  - `mat4MultiplyVec4`
- Shorter legacy aliases are still present for compatibility, but are deprecated
- For matrix inversion, prefer explicit-failure APIs:
  - `mat3TryInverse`
  - `mat3InverseChecked`
  - `mat4TryInverse`
  - `mat4InverseChecked`

### Quaternion conventions

`Quat` is a 16-byte aligned POD with the fixed layout `{x, y, z, w}` where
`w` is the real (scalar) part and identity is `{0, 0, 0, 1}`. All
quaternion APIs follow these locked rules:

- **Hamilton, right-handed, radians**
- `quatMultiply(a, b)` applies `b` first and then `a`, so
  `quatRotateVec3(quatMultiply(a, b), v) == quatRotateVec3(a, quatRotateVec3(b, v))`
- `quatIntegrate(q, omega, dt)` integrates a world-frame angular velocity
  (`q_dot = 0.5 * omega_quat * q`) and renormalizes the result
- `quatNormalize` is **Deterministic-Profile compliant**: it routes through
  `MMath::inverseSqrt` (rsqrt + one Newton step) and never `std::sqrt` or
  raw `_mm_rsqrt_*`. Use `quatNormalizeFast` when you want the raw rsqrt
  throughput path
- Only `quatNlerp` and `quatSlerp` are exposed for interpolation; raw
  `quatLerp` is intentionally omitted because it yields non-unit results
- Vec4 overloads of `quatIntegrate` and `quatRotateVec4` exist for SoA /
  Vec4-column callers so they do not have to repack data on the hot path

---

## CMake Options

### Core options

- `FAST_MATH_ENABLE_AGGRESSIVE_FLAGS=ON|OFF`
- `FAST_MATH_BUILD_TESTS=ON|OFF`
- `FAST_MATH_BUILD_BENCHMARKS=ON|OFF`
- `FAST_MATH_BUILD_MODULES=ON|OFF`

### Optional backend discovery

- `FAST_MATH_DIRECTXMATH_DIR`
- `FAST_MATH_EIGEN_DIR`
- `FAST_MATH_GLM_DIR`

These paths may point either to a package root or directly to the include
directory that contains the relevant headers.

---

## Testing

The test suite is organized by subsystem and focuses on:

- numerical correctness
- boundary conditions
- parity between header and module delivery modes
- optional cross-library consistency

Run the full test suite with:

```powershell
ctest --test-dir build --output-on-failure
```

---

## Benchmarking

The benchmark suite compares Melosyne Math against scalar baselines and optional
third-party backends when present.

Run a quick benchmark pass with:

```powershell
.\build\fast_math_bench.exe --quick
```

For focused runs, use the benchmark binary filters provided by the framework.

---

## Implementation Notes

- Shared implementation details live in `include/fast_math/detail/`
- Public headers and module interfaces are kept in parity
- `mat4_d3d` has been removed; `Mat4` now uses the canonical standard path
- The legacy `mat4FromQuat(const Vec4&)` entry point is preserved for
  back-compat; new code should prefer `quatToMat4(const Quat&)`, which
  produces byte-identical output and accepts the explicit `Quat` type
- `build/` and other generated artifacts are intentionally excluded from source
  control

---

## Contributing

If you are modifying the library:

- keep public APIs in `MMath`
- keep benchmark/test helpers in local namespaces
- prefer deterministic tests with fixed seeds
- avoid pushing aggressive compiler flags to downstream consumers
- update tests and benchmarks when changing numerical behavior

For repository-specific workflow notes, see `AGENTS.md`.
