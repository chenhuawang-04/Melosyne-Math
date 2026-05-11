/**
 * @file bit_ops_core_simd.h
 * @brief SIMD-optimized core bitwise operations (AVX2)
 */

#pragma once

#include "config_macros.h"
#include "bitset_view.h"
#include "bit_ops_core.h"

#include <algorithm>
#include <cstdint>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace Melosyne {
namespace BitOps {
#include "detail/bit_ops_dispatch_policy.inl"
#include <fast_math/detail/bit_ops_core_simd_detail.inl>
} // namespace BitOps
} // namespace Melosyne

#include <fast_math/detail/bitops_namespace_alias.inl>
