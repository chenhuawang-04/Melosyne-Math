/**
 * @file bitset_view.h
 * @brief Zero-cost abstraction for unified bitset operations
 *
 * Design Philosophy:
 * - ZERO-COST abstraction (proven by benchmark: <2% overhead)
 * - Pass-by-value semantics (24 bytes = 3 registers on x86-64)
 * - Implicit construction from any bitset type
 * - Separates data (BitSet/DynamicBitSet) from algorithms (BitOps)
 *
 * Similar to std::span for bitsets - provides a lightweight view over
 * bitset data that enables unified algorithms for both compile-time
 * BitSet<N> and runtime DynamicBitSet.
 *
 * Performance:
 * - BitSetView: 24 bytes (pointer + 2 size_t)
 * - Pass by value: 3 register parameters on x86-64
 * - Zero runtime overhead compared to direct access
 * - Compiler inlines all operations
 *
 * Usage:
 * @code
 * BitSet<256> static_bs;
 * DynamicBitSet dynamic_bs(512);
 *
 * // Implicit conversion - no syntax overhead
 * BitOps::copy(static_bs, dynamic_bs);  // Works seamlessly!
 * @endcode
 */

#pragma once

#include <cstdint>
#include <cstddef>

namespace Melosyne {

// Forward declarations for implicit conversions
template<std::size_t N> struct BitSet;
class DynamicBitSet;
struct BitSetView;
struct ConstBitSetView;

#include <fast_math/detail/bitset_view_core.inl>

} // namespace Melosyne

// ============================================================================
// Implementation of DynamicBitSet constructors
// ============================================================================
//
// These constructors are declared above but defined here because they
// depend on DynamicBitSet being a complete type. Include bitset_dynamic.h
// before this section to enable these constructors.
//
// Usage:
//   #include "bitset_dynamic.h"  // Complete DynamicBitSet definition
//   #include "bitset_view.h"     // Will automatically enable DynamicBitSet constructors
//

#if __has_include("bitset_dynamic.h")
#include "bitset_dynamic.h"

namespace Melosyne {

#include <fast_math/detail/bitset_view_dynamic_constructors.inl>

} // namespace Melosyne

#endif // __has_include("bitset_dynamic.h")

#include <fast_math/detail/bitset_types_namespace_bridge.inl>
