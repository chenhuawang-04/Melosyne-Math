#pragma once

// Centralized SIMD dispatch thresholds for bit operations.
// Values are conservative and benchmark-backed on the current AVX2 path:
// keep broad bitwise kernels eager, but defer Harley-Seal popcount and rank/select
// until the working set is large enough to amortize setup and reduction costs.

constexpr std::size_t kBitwiseSimdMinWords = 128;
constexpr std::size_t kBitwiseAndCountSimdMinWords = 2048;
constexpr std::size_t kPopcountSimdMinWords = 2048;
constexpr std::size_t kFusedCountSimdMinWords = 8;
constexpr std::size_t kSelectSimdMinWords = 128;
constexpr std::size_t kRankSimdMinWords = 256;
constexpr std::size_t kRankSimdMinPosWords = 64;
constexpr std::size_t kRangeSimdMinFullWords = 4;
constexpr std::size_t kReverseSimdMinWords = 64;
constexpr std::size_t kByteSwapSimdMinWords = 8;
constexpr std::size_t kRotateSimdMinWords = 64;
