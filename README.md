# Melosyne Math

High-performance C++20 math library focused on **strict POD data types**, **free-function APIs**, and **SIMD-first implementations**.

This repository now provides **two equivalent delivery modes**:

- **Header API** under `include/fast_math/`
- **Native C++20 modules (.cppm)** under `modules/fast_math/` (real module interfaces, not header wrappers)

---

## Highlights

- Vector/Matrix/AABB/BitOps toolset for real-time workloads
- Scalar + SSE/AVX/FMA optimized paths
- Unified test and benchmark framework
- Optional cross-library validation/benchmark against:
  - DirectXMath
  - Eigen 3.4.0
  - GLM 1.0.2

---

## Project Layout

```text
include/fast_math/      # Header API (types, vec*, mat*, aabb*, bit_ops*, trig/power/sqrt/...)
modules/fast_math/      # C++20 module interface units (*.cppm)
test/                   # Unified correctness framework
bench/                  # Unified performance framework
docs/ + *_REPORT.md     # Design/perf notes
build/                  # Generated artifacts (do not commit)
```

---

## Build & Run (Clang + Ninja Recommended)

```powershell
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build -j 8
ctest --test-dir build --output-on-failure
.\build\fast_math_bench.exe --quick
```

### Module-enabled build

```powershell
cmake -S . -B build_modules -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build_modules -j 8 --target fast_math_modules_smoke
```

> `FAST_MATH_BUILD_MODULES` requires **CMake >= 3.28**.

---

## Usage

### 1) Header style

```cpp
#include <fast_math/vec3.h>

MMath::Vec3 a{1,2,3};
MMath::Vec3 b{4,5,6};
auto c = MMath::vec3Add(a, b);
```

### 2) C++20 module style

```cpp
import fast_math.all;

MMath::Vec3 a{1,2,3};
MMath::Vec3 b{4,5,6};
auto c = MMath::vec3Add(a, b);
```

You can also import finer-grained modules, e.g.:

```cpp
import fast_math.types;
import fast_math.vec3;
import fast_math.mat4;
import fast_math.bit_ops;
```

---

## Key CMake Options

- `FAST_MATH_ENABLE_AGGRESSIVE_FLAGS=ON|OFF`
- `FAST_MATH_BUILD_TESTS=ON|OFF`
- `FAST_MATH_BUILD_BENCHMARKS=ON|OFF`
- `FAST_MATH_BUILD_MODULES=ON|OFF`

Optional comparison dependency roots:

- `FAST_MATH_DIRECTXMATH_DIR`
- `FAST_MATH_EIGEN_DIR`
- `FAST_MATH_GLM_DIR`

---

## Notes

- Public API namespaces are `MMath` and `Melosyne::BitOps`.
- Macro/config definitions used by headers/modules are centralized in:
  - `include/fast_math/config_macros.h`
- For contribution details, see `AGENTS.md`.
