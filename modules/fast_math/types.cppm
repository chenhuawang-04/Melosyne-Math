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

export module fast_math.types;

export {
/**
 * @file types.h
 * @brief Strict POD data structures (NO member functions)
 *
 * Design Philosophy:
 * - STRICT POD (Plain Old Data) - C-compatible
 * - NO constructors, NO destructors, NO member functions
 * - Manipulated ONLY by free functions
 * - Cache-line aligned for SIMD performance
 */




namespace MMath {

#include "fast_math/detail/types_core.inl"

} // namespace MMath
}
