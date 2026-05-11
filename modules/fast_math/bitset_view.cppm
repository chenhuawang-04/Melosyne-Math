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

export module fast_math.bitset_view;

export import fast_math.bitset_static;
export import fast_math.bitset_dynamic;

export {
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



namespace Melosyne {
#include "fast_math/detail/bitset_view_core.inl"

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

namespace Melosyne {

#include "fast_math/detail/bitset_view_dynamic_constructors.inl"

} // namespace Melosyne
}
