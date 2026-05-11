/**
 * @file bit_ops_core.h
 * @brief Scalar bitwise logical operations (no SIMD intrinsics)
 *
 * Design Philosophy:
 * - FREE FUNCTIONS only (no methods)
 * - Scalar versions for single bitset operations
 * - Simple loops (compiler auto-vectorization)
 * - For batch/array operations, use bit_ops_core_simd.h
 *
 * Performance:
 * - Suitable for small bitsets (<= 256 bits)
 * - Compiler may auto-vectorize loops
 * - Zero overhead compared to member functions
 *
 * Usage:
 * @code
 * BitSet<256> a = {}, b = {};
 * bitOpsAnd(a, b);  // a &= b (scalar version)
 * @endcode
 */

#pragma once

#include "bitset_view.h"
#include <cstring>
#include <algorithm>

namespace Melosyne {
namespace BitOps {
#include <fast_math/detail/bit_ops_core_scalar.inl>
} // namespace BitOps
} // namespace Melosyne

#include <fast_math/detail/bitops_namespace_alias.inl>
