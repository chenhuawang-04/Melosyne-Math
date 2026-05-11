#pragma once

// ============================================================================
// Core inlining / alignment / restrict macros
// ============================================================================

#if defined(_MSC_VER)
    #define MMATH_ALIGN(x) __declspec(align(x))
    #define MMATH_FORCE_INLINE __forceinline
    #define MMATH_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
    #define MMATH_ALIGN(x) __attribute__((aligned(x)))
    #define MMATH_FORCE_INLINE __attribute__((always_inline)) inline
    #define MMATH_RESTRICT __restrict__
#else
    #define MMATH_ALIGN(x)
    #define MMATH_FORCE_INLINE inline
    #define MMATH_RESTRICT
#endif

#if defined(_MSC_VER)
    #define BITOPS_ALIGN(x) __declspec(align(x))
    #define BITOPS_FORCEINLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
    #define BITOPS_ALIGN(x) __attribute__((aligned(x)))
    #define BITOPS_FORCEINLINE __attribute__((always_inline)) inline
#else
    #define BITOPS_ALIGN(x)
    #define BITOPS_FORCEINLINE inline
#endif

#if defined(__has_cpp_attribute)
    #if __has_cpp_attribute(deprecated)
        #define MMATH_DEPRECATED(message) [[deprecated(message)]]
    #else
        #define MMATH_DEPRECATED(message)
    #endif
#else
    #define MMATH_DEPRECATED(message)
#endif

// ============================================================================
// SIMD capability macros (module-internal preprocessor switches)
// ============================================================================

#if (defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)) && !defined(MMATH_SIMD_SSE2)
    #define MMATH_SIMD_SSE2 1
#endif

#if defined(__SSE3__) && !defined(MMATH_SIMD_SSE3)
    #define MMATH_SIMD_SSE3 1
#endif

#if defined(__SSSE3__) && !defined(MMATH_SIMD_SSSE3)
    #define MMATH_SIMD_SSSE3 1
#endif

#if defined(__SSE4_1__) && !defined(MMATH_SIMD_SSE4_1)
    #define MMATH_SIMD_SSE4_1 1
#endif

#if defined(__AVX__) && !defined(MMATH_SIMD_AVX)
    #define MMATH_SIMD_AVX 1
#endif

#if defined(__FMA__) && !defined(MMATH_SIMD_FMA)
    #define MMATH_SIMD_FMA 1
#endif

#if defined(__AVX2__) && defined(__FMA__) && !defined(MMATH_HAS_AVX2_FMA)
    #define MMATH_HAS_AVX2_FMA 1
#endif

#if defined(__SSE4_1__) && defined(__FMA__) && !defined(MMATH_HAS_SSE_FMA)
    #define MMATH_HAS_SSE_FMA 1
#endif
