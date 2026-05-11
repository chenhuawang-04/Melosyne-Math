/**
 * @file power.h
 * @brief Exponential and power functions (exp, log, pow)
 *
 * Design Philosophy:
 * - STRICT POD structures (NO member functions)
 * - FREE FUNCTIONS only
 * - Scalar versions use range reduction + polynomial approximation
 * - Array versions dispatch to SIMD batch processing
 *
 * Algorithm Overview (based on Julien Pommier's sse_mathfun and cephes library):
 *
 * exp(x) Implementation:
 * 1. Range Reduction: x = k*ln(2) + r, where |r| < ln(2)/2
 * 2. Polynomial Approx: exp(r) ~= 1 + r + r^2/2! + r^3/3! + r^4/4! + r^5/5!
 * 3. Reconstruction: exp(x) = 2^k * exp(r)
 *
 * log(x) Implementation:
 * 1. Extract Exponent: x = 2^e * m, where 1 <= m < 2
 * 2. Range Adjust: If m < sqrt(2)/2, then m *= 2 and e -= 1
 * 3. Polynomial Approx: log(m) ~= P(m-1) using minimax polynomial
 * 4. Reconstruction: log(x) = e*ln(2) + log(m)
 *
 * pow(x, y) = exp(y * log(x))
 *
 * Accuracy Contract (current test-validated budget):
 * - exp / log / pow10: ~2e-2 relative tolerance in scalar tests
 * - pow: ~5e-2 relative tolerance in scalar tests
 * - exp2 / powi: tighter because they avoid the full log+exp pipeline
 *
 * Performance Targets:
 * - exp: 3-5x faster than std::exp (via polynomial vs libm's Taylor series)
 * - log: 2-4x faster than std::log
 * - pow: 5-10x faster than std::pow (via log+exp decomposition)
 *
 * References:
 * - Julien Pommier's sse_mathfun: http://gruntthepeon.free.fr/ssemath/
 * - Cephes Math Library: http://www.netlib.org/cephes/
 * - Stephen L. Moshier's minimax polynomial approximations
 * - IEEE 754 floating-point representation
 *
 * Use Cases (from audio engine analysis):
 * - dB to linear: pow(10, gain_db/20) - 2+ occurrences
 * - Distance attenuation: pow(distance/min, -rolloff) - 2+ occurrences
 * - Exponential decay envelopes
 * - Filter coefficient calculations
 */

#pragma once

#include "types.h"
#include <cstring>  // for memcpy (type punning)
#include <cmath>    // for reference std:: functions
#include <cfloat>
#include <limits>

namespace MMath {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnan-infinity-disabled"
#endif
#include "fast_math/detail/power_scalar.inl"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
} // namespace MMath

// Include SIMD batch processing implementations
#include "power_simd.h"
