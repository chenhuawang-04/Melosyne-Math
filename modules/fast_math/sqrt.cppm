module;

#include "fast_math/config_macros.h"
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>
#include <bit>
#include <iterator>
#include <type_traits>
#include <memory>
#include <memory_resource>
#include <cfloat>
#include <smmintrin.h>

export module fast_math.sqrt;

export import fast_math.types;
export import fast_math.sqrt_simd;

export {
/**
 * @file sqrt.h
 * @brief Square root and reciprocal functions
 *
 * Design Philosophy:
 * - STRICT POD structures (NO member functions)
 * - FREE FUNCTIONS only
 * - Scalar versions use SSE scalar instructions
 * - Array versions dispatch to SIMD batch processing
 *
 * Performance Optimizations:
 * - sqrt: Hardware sqrtss instruction (exact, 14-15 cycle latency on modern CPUs)
 * - inverseSqrt: rsqrt + Newton-Raphson iteration (2-3x faster than 1/sqrt)
 * - reciprocal: rcp + Newton-Raphson iteration (2x faster than 1/x on older CPUs)
 *
 * Accuracy:
 * - sqrt: Exact (hardware precision)
 * - inverseSqrt: currently test-validated within 2e-3f absolute error on
 *   representative positive finite inputs; no blanket ULP guarantee
 * - reciprocal: currently test-validated within 2e-3f absolute error on
 *   representative non-zero finite inputs; no blanket ULP guarantee
 *
 * References:
 * - Fast rsqrt with SSE/AVX: https://stackoverflow.com/questions/31555260/fast-vectorized-rsqrt-and-reciprocal-with-sse-avx-depending-on-precision
 * - Newton-Raphson accuracy: https://www.mdpi.com/1099-4300/23/1/86
 * - DirectXMath optimization guide: https://learn.microsoft.com/en-us/windows/win32/dxmath/pg-xnamath-optimizing
 * - GLM fast square root: https://glm.g-truc.net/0.9.3/api/a00168.html
 */



#if defined(__SSE4_1__)
#endif

namespace MMath {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winclude-angled-in-module-purview"
#endif
#include "fast_math/detail/sqrt_core.inl"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
} // namespace MMath

// Include SIMD batch processing implementations
}
