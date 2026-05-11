/**
 * @file trig.h
 * @brief High-level trigonometry API (operates on POD structures)
 *
 * Design:
 * - Operates on STRICT POD structures
 * - FREE FUNCTIONS only (NO member functions)
 * - Automatic SIMD dispatch (AVX2 > SSE4.1 > Scalar)
 * - Maximum performance batch processing
 */

#pragma once

#include "types.h"
#include "trig_simd.h"
#include <cstring>

namespace MMath {
#include <fast_math/detail/trig_core.inl>
} // namespace MMath
