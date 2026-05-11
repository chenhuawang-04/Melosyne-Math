/**
 * @file bitset_static.h
 * @brief Compile-time fixed-size bitset (STRICT POD, NO member functions)
 *
 * Design Philosophy:
 * - STRICT POD (Plain Old Data) - C-compatible
 * - NO constructors, NO destructors, NO member functions
 * - Manipulated ONLY by free functions in bit_ops.h and bit_ops_simd.h
 * - Cache-line aligned for SIMD performance (32-byte alignment)
 *
 * POD guarantees:
 * - Can be memcpy'd safely
 * - Can be used in unions
 * - Compatible with C interfaces  
 * - Trivially copyable and standard layout
 *
 * Usage:
 * @code
 * BitSet<256> bs = {};  // Zero-initialize (C-style)
 * bitOpsSet(bs, 42);    // Set bit 42 using free function
 * bool is_set = bitOpsTest(bs, 42);  // Test bit 42
 * @endcode
 */

#pragma once

#include "config_macros.h"

#include <cstdint>
#include <type_traits>

namespace Melosyne {

#include <fast_math/detail/bitset_static_core.inl>

} // namespace Melosyne

#include <fast_math/detail/bitset_types_namespace_bridge.inl>
