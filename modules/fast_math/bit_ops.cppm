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

export module fast_math.bit_ops;

export import fast_math.bitset_static;
export import fast_math.bitset_dynamic;
export import fast_math.bitset_view;
export import fast_math.bit_ops_core;
export import fast_math.bit_ops_count;
export import fast_math.bit_ops_scan;
export import fast_math.bit_ops_range;
export import fast_math.bit_ops_advanced;
export import fast_math.bit_ops_core_simd;
export import fast_math.bit_ops_count_simd;
export import fast_math.bit_ops_scan_simd;
export import fast_math.bit_ops_range_simd;
export import fast_math.bit_ops_advanced_simd;

export {
/**
 * @file bit_ops.h
 * @brief Unified header for complete bit operations library
 *
 * This header provides access to the entire high-performance bit operations library:
 *
 * **Data Structures:**
 * - BitSet<N>: Compile-time fixed-size bitset (pure POD, alignas(32))
 * - DynamicBitSet: Runtime-sized bitset with PMR allocator support
 * - BitSetView: Zero-cost abstraction unifying static and dynamic bitsets
 *
 * **Core Operations:**
 * - Bitwise logical: AND, OR, XOR, NOT, ANDNOT
 * - Copy and equality testing
 *
 * **Statistical Operations:**
 * - popcount, hammingDistance, any/none/all
 * - Set operations: union, intersection, isSubsetOf, intersects
 * - Similarity metrics: jaccardSimilarity, diceCoefficient
 *
 * **Scan Operations:**
 * - Find: findFirst, findLast, findNext, findPrev
 * - Find zero: findFirstZero, findLastZero
 * - Select/Rank: k-th set bit, count up to position
 * - Iterator: Range-based for loop support
 *
 * **Range Operations:**
 * - Modify: setRange, resetRange, flipRange
 * - Query: testAll, testAny, testNone
 * - Count: popcountRange
 *
 * **Advanced Operations:**
 * - Bit reversal: SWAR algorithm
 * - Rotation: Circular shift
 * - Byte swap: Endianness conversion
 * - Parity: XOR of all bits
 * - PEXT/PDEP: BMI2 parallel bit extract/deposit
 * - Gray code: Binary ↔ Gray code
 * - Morton code: 2D spatial indexing
 * - Permutation: Gosper's hack
 * - Scatter/Gather: Indexed bit access
 *
 * **SIMD Optimizations:**
 * When compiled with AVX2 support, many operations automatically dispatch to
 * optimized SIMD implementations. Functions with "Optimized" suffix provide
 * automatic selection between scalar and SIMD based on data size.
 *
 * **Performance Characteristics:**
 * - Core ops (AND/OR/XOR): 2-4x faster with AVX2 for large bitsets
 * - popcount: 2.4x faster with Harley-Seal algorithm (medium bitsets)
 * - scan ops: Scalar optimal (1-cycle TZCNT/LZCNT, early-exit)
 * - range ops: 4x faster with AVX2 for bulk operations
 * - advanced ops: Mostly scalar optimal (SWAR, BMI2)
 *
 * **Usage:**
 * @code
 * #include <fast_math/bit_ops.h>
 *
 * using namespace Melosyne;
 *
 * // Create bitsets
 * BitSet<256> bs1 = {};
 * BitSet<256> bs2 = {};
 *
 * // Core operations (automatically uses AVX2 if beneficial)
 * BitOps::bitwiseAndOptimized(bs1, bs2);
 *
 * // Statistical operations
 * std::size_t count = BitOps::popcountOptimized(bs1);
 * std::size_t distance = BitOps::hammingDistanceOptimized(bs1, bs2);
 *
 * // Scan operations (scalar is optimal)
 * std::size_t first = BitOps::findFirst(bs1);
 * std::size_t next = BitOps::findNext(bs1, first);
 *
 * // Range operations (automatically uses AVX2 if beneficial)
 * BitOps::setRangeOptimized(bs1, 10, 100);
 * std::size_t count_in_range = BitOps::popcountRangeOptimized(bs1, 10, 100);
 *
 * // Advanced operations
 * uint64_t reversed = BitOps::reverseBits64(0xDEADBEEF);
 * uint64_t extracted = BitOps::pext64(value, mask);
 *
 * // Range-based iteration
 * for (std::size_t pos : BitOps::BitPositionIterator(bs1)) {
 *     // Process each set bit
 * }
 * @endcode
 *
 * **Design Philosophy:**
 * - Free functions (no member methods on BitSet)
 * - Zero-cost abstractions (BitSetView)
 * - Automatic SIMD dispatch (no manual selection needed)
 * - Compile-time optimization (constexpr where possible)
 * - Standards compliance (C++20, POD requirements)
 *
 * @note For maximum performance, compile with:
 *       - GCC/Clang: -O3 -march=native -mavx2
 *       - MSVC: /O2 /arch:AVX2
 *       - BMI2: Add -mbmi2 (GCC/Clang) for hardware PEXT/PDEP
 *
 * @author Melosyne High-Performance Computing Team
 * @version 2.0.0
 */


// ============================================================================
// Data Structures
// ============================================================================


// ============================================================================
// Scalar Operations (always available)
// ============================================================================


// ============================================================================
// SIMD Optimizations (optional, automatic dispatch)
// ============================================================================

#if defined(__AVX2__) || defined(__SSE4_1__)


#endif

// ============================================================================
// Convenience Namespace Alias (optional)
// ============================================================================

namespace Melosyne {
namespace BitOps {

/**
 * @brief Recommended API: Use "Optimized" suffixed functions for best performance
 *
 * These functions automatically choose between scalar and SIMD implementations
 * based on data size and availability of SIMD instructions.
 *
 * **Core Operations:**
 * - bitwiseAndOptimized, bitwiseOrOptimized, bitwiseXorOptimized
 * - bitwiseNotOptimized, bitwiseAndNotOptimized
 *
 * **Count Operations:**
 * - popcountOptimized, hammingDistanceOptimized
 * - allOptimized (any/none are always scalar)
 * - unionCountOptimized, intersectionCountOptimized
 * - jaccardSimilarityOptimized, diceCoefficientOptimized
 *
 * **Scan Operations:**
 * - selectOptimized, rankOptimized
 * - findFirst/Last/Next/Prev (always scalar, 1-cycle TZCNT/LZCNT)
 *
 * **Range Operations:**
 * - setRangeOptimized, resetRangeOptimized, flipRangeOptimized
 * - popcountRangeOptimized
 * - testAll/Any/None (always scalar, early-exit advantage)
 *
 * **Advanced Operations:**
 * - reverseOptimized, byteSwapOptimized
 * - rotateLeftOptimized, rotateRightOptimized
 * - reverseBits64/32/16/8 (always scalar SWAR)
 * - pext64/32, pdep64/32 (BMI2 or fallback)
 * - binaryToGray, grayToBinary, interleaveBits32, etc. (always scalar)
 */

// No additional aliases needed - use functions directly with Optimized suffix
// Example: BitOps::popcountOptimized(bs) for best performance
//          BitOps::popcount(bs) for guaranteed scalar version

} // namespace BitOps
} // namespace Melosyne

// ============================================================================
// Implementation Notes
// ============================================================================

/*
 * **When to use Optimized versions:**
 *
 * Always prefer "Optimized" suffix for operations on:
 * - Large bitsets (> 512 bits / 8 words)
 * - Bulk operations (core ops, range ops, counting)
 * - Performance-critical code paths
 *
 * The Optimized versions have zero overhead when SIMD is not beneficial:
 * - Small bitsets: Falls back to scalar automatically
 * - No AVX2: Functions become aliases to scalar versions
 * - Early-exit ops: Always uses scalar (findFirst, any, testAll)
 *
 * **When to use scalar versions:**
 *
 * Use non-Optimized versions when:
 * - Explicitly need scalar behavior (deterministic performance)
 * - Debugging or profiling specific code paths
 * - Operations are already scalar-optimal (findFirst, TZCNT-based)
 *
 * **SIMD Availability Detection:**
 *
 * SIMD optimizations are enabled automatically when compiling with:
 * - __AVX2__ defined: Full AVX2 optimizations
 * - __SSE4_1__ defined: Limited SSE4.1 optimizations
 *
 * Runtime CPU detection is NOT performed - SIMD is a compile-time decision.
 * This ensures:
 * - Zero runtime overhead
 * - Inlining and optimization opportunities
 * - Binary size optimization (dead code elimination)
 *
 * **Performance Tuning:**
 *
 * For critical code, benchmark both scalar and SIMD versions:
 * @code
 * // Benchmark scalar
 * auto start = std::chrono::high_resolution_clock::now();
 * for (int i = 0; i < 1000000; ++i) {
 *     BitOps::popcount(bs);
 * }
 * auto end = std::chrono::high_resolution_clock::now();
 *
 * // Benchmark SIMD
 * start = std::chrono::high_resolution_clock::now();
 * for (int i = 0; i < 1000000; ++i) {
 *     BitOps::popcountOptimized(bs);
 * }
 * end = std::chrono::high_resolution_clock::now();
 * @endcode
 *
 * Threshold sizes for SIMD dispatch:
 * - Core ops (AND/OR/XOR): > 8 words (512 bits)
 * - Popcount: 8-512 words (512B-32KB, Harley-Seal optimal)
 * - Select/Rank: > 8 words
 * - Range ops: >= 4 full words in range
 * - Advanced ops: > 16 words (reverse, byteSwap)
 *
 * **Thread Safety:**
 *
 * All functions are thread-safe when operating on different bitsets.
 * Concurrent access to the same bitset requires external synchronization.
 *
 * **Memory Alignment:**
 *
 * BitSet<N> is aligned to 32 bytes (alignas(32)) for optimal AVX2 performance.
 * DynamicBitSet ensures proper alignment through allocator.
 * Unaligned access is supported but may be slower (_mm256_loadu_si256).
 *
 * **Compiler Support:**
 *
 * Tested with:
 * - GCC 11+ (best SIMD codegen)
 * - Clang 14+ (good SIMD codegen)
 * - MSVC 19.30+ (requires /arch:AVX2)
 *
 * C++20 required for:
 * - std::popcount (C++20)
 * - std::countr_zero, std::countl_zero (C++20)
 * - Concepts and constraints (future work)
 */
}
