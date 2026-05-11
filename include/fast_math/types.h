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

#pragma once

#include "config_macros.h"

#include <cstdint>
#include <cstring>

namespace MMath {

#include <fast_math/detail/types_core.inl>

} // namespace MMath
