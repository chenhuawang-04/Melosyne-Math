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

export module fast_math.common;

export import fast_math.types;

export {
/**
 * @file common.h
 * @brief Common mathematical functions (min, max, clamp, abs, sign)
 *
 * Design Philosophy:
 * - STRICT POD structures (NO member functions)
 * - FREE FUNCTIONS only
 * - Scalar versions use SIMD scalar instructions (branch-free)
 * - Array versions dispatch to SIMD batch processing
 *
 * Performance Optimizations:
 * - min/max/clamp: Branch-free SSE scalar instructions (minss/maxss)
 * - abs: Bit manipulation (AND with 0x7FFFFFFF mask to clear sign bit)
 * - sign: Simple comparison (compiler optimizes to branchless code)
 *
 * References:
 * - GLM common functions: https://github.com/g-truc/glm/blob/master/glm/detail/func_common.inl
 * - DirectXMath optimization: https://learn.microsoft.com/en-us/windows/win32/dxmath/pg-xnamath-optimizing
 * - SIMD bit tricks: https://hackaday.com/2024/12/22/faster-integer-division-with-floating-point/
 */



#if defined(__SSE4_1__)
#endif

namespace MMath {

// ============================================================================
// Comparison Functions (Branch-Free)
// ============================================================================

/**
 * @brief Minimum of two floats
 *
 * Implementation:
 * - Uses std::min which compilers optimize to minss on x86
 * - Single instruction, no branches
 * - Handles NaN consistently (returns second argument if either is NaN)
 *
 * @param a_ First value
 * @param b_ Second value
 * @return Smaller of a_ and b_
 *
 * Performance: Same as minss instruction (1 cycle latency, 0.5-1 cycle throughput)
 */
MMATH_FORCE_INLINE float min(float a_, float b_) noexcept {
    return (a_ < b_) ? a_ : b_;
}

/**
 * @brief Maximum of two floats
 *
 * Implementation:
 * - Uses std::max which compilers optimize to maxss on x86
 * - Single instruction, no branches
 *
 * @param a_ First value
 * @param b_ Second value
 * @return Larger of a_ and b_
 *
 * Performance: Same as maxss instruction (1 cycle latency, 0.5-1 cycle throughput)
 */
MMATH_FORCE_INLINE float max(float a_, float b_) noexcept {
    return (a_ > b_) ? a_ : b_;
}

/**
 * @brief Clamp value to [min_val, max_val] range
 *
 * Formula: max(min_val, min(x, max_val))
 * Implementation: Chained min/max calls that compile to minss/maxss sequence
 *
 * @param x_ Value to clamp
 * @param min_val_ Minimum bound
 * @param max_val_ Maximum bound
 * @return Clamped value
 *
 * @note If min_val_ > max_val_, behavior is undefined (but safe)
 *
 * Use cases (from audio engine analysis):
 * - Volume limiting: clamp(volume, 0.0f, 1.0f) - 27 occurrences
 * - Pan limiting: clamp(pan, -1.0f, 1.0f)
 * - Soft clipping: clamp(tanh_approx(x), -1.0f, 1.0f)
 *
 * Performance: 2 instructions (1x maxss + 1x minss), ~2-3 cycles
 */
MMATH_FORCE_INLINE float clamp(float x_, float min_val_, float max_val_) noexcept {
    // Chained approach: max(min_val, min(x, max_val))
    // Compiles to: maxss then minss (2 instructions, fully pipelined)
    float temp = (x_ < max_val_) ? x_ : max_val_;  // min(x, max_val)
    return (temp > min_val_) ? temp : min_val_;     // max(temp, min_val)
}

// ============================================================================
// Absolute Value & Sign Functions
// ============================================================================

/**
 * @brief Absolute value
 *
 * Implementation: Uses fabsf which compiles to a single instruction
 * - x86: andps xmm, [0x7FFFFFFF] (clears sign bit)
 * - No branching, no conversion overhead
 *
 * @param x_ Input value
 * @return |x_|
 *
 * Note: Removed SSE intrinsics - fabsf generates optimal code directly
 */
MMATH_FORCE_INLINE float abs(float x_) noexcept {
    return std::fabs(x_);  // C++ standard, works with float
}

/**
 * @brief Sign function: returns -1, 0, or +1
 *
 * Returns:
 *  -1.0f if x_ < 0
 *   0.0f if x_ == 0 (including -0.0f)
 *  +1.0f if x_ > 0
 *
 * Note: NaN returns 0.0f (since NaN comparisons are false)
 *
 * @param x_ Input value
 * @return Sign of x_
 */
MMATH_FORCE_INLINE float sign(float x_) noexcept {
    // Compiler optimizes this to branchless code (conditional moves)
    return (x_ > 0.0f) ? 1.0f : ((x_ < 0.0f) ? -1.0f : 0.0f);
}

/**
 * @brief Saturate value to [0, 1] range
 *
 * Equivalent to: clamp(x, 0, 1)
 * Common in graphics programming (color/alpha clamping, texture coordinates)
 *
 * @param x_ Input value
 * @return Clamped value in [0, 1]
 */
MMATH_FORCE_INLINE float saturate(float x_) noexcept {
    return clamp(x_, 0.0f, 1.0f);
}

/**
 * @brief Copy sign from y_ to x_ (like std::copysign)
 *
 * Returns: magnitude of x_ with sign of y_
 *
 * Implementation:
 * - Extract sign bit from y_ (bit 31)
 * - Clear sign bit from x_
 * - OR them together
 *
 * Use case: Ensuring consistent sign in calculations
 *
 * @param x_ Value providing magnitude
 * @param y_ Value providing sign
 * @return x_ with sign of y_
 */
MMATH_FORCE_INLINE float copySign(float x_, float y_) noexcept {
    return std::copysign(x_, y_);  // C++ standard, works with float
}

// ============================================================================
// Comparison Utilities
// ============================================================================

/**
 * @brief Check if two floats are nearly equal (epsilon comparison)
 *
 * Formula: |a - b| <= epsilon
 *
 * Use cases (from audio engine):
 * - Floating-point comparison in tests
 * - Physics tolerance checks
 *
 * @param a_ First value
 * @param b_ Second value
 * @param epsilon_ Tolerance (default: 1e-5f)
 * @return true if values are within epsilon
 */
MMATH_FORCE_INLINE bool nearEqual(float a_, float b_, float epsilon_ = 1e-5f) noexcept {
    return MMath::abs(a_ - b_) <= epsilon_;
}

/**
 * @brief Check if float is approximately zero
 *
 * Formula: |x| <= epsilon
 *
 * @param x_ Value to check
 * @param epsilon_ Tolerance (default: 1e-6f)
 * @return true if value is within epsilon of zero
 */
MMATH_FORCE_INLINE bool isZero(float x_, float epsilon_ = 1e-6f) noexcept {
    return MMath::abs(x_) <= epsilon_;
}

} // namespace MMath

// Include SIMD batch processing implementations
}
