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

namespace MMath {

// ============================================================================
// Batch Processing API (POD + Free Functions)
// ============================================================================

/**
 * @brief Compute sin/cos for a batch of angles
 *
 * This function automatically dispatches to:
 * - AVX2 (8 floats at once) if available
 * - SSE4.1 (4 floats at once) if available
 * - Scalar fallback
 *
 * @param angles_ Input angle batch (POD structure)
 * @param result_ Output sin/cos batch (POD structure)
 *
 * Requirements:
 * - angles_.data must be valid pointer
 * - result_.sin_data and result_.cos_data must be preallocated
 * - count must match between angles_ and result_
 *
 * Performance: ~2-3 cycles per angle on AVX2
 */
inline void sincosBatch(const AngleBatch* MMATH_RESTRICT angles_,
                        SinCosBatch* MMATH_RESTRICT result_) noexcept {
    const float* MMATH_RESTRICT in = angles_->data;
    float* MMATH_RESTRICT out_sin = result_->sin_data;
    float* MMATH_RESTRICT out_cos = result_->cos_data;
    int32_t count = angles_->count;

    int32_t i = 0;

#if defined(__AVX2__) && defined(__FMA__)
    // Process 8 at a time with AVX2
    const int32_t avx2_count = (count / 8) * 8;
    for (; i < avx2_count; i += 8) {
        AnglesAvx2 angle_pack;
        std::memcpy(angle_pack.angles, in + i, 8 * sizeof(float));

        SinCosAvx2 result_pack;
        sincosAvx2(&angle_pack, &result_pack);

        std::memcpy(out_sin + i, result_pack.sins, 8 * sizeof(float));
        std::memcpy(out_cos + i, result_pack.coss, 8 * sizeof(float));
    }
#elif defined(__SSE4_1__)
    // Process 4 at a time with SSE4.1
    const int32_t sse_count = (count / 4) * 4;
    for (; i < sse_count; i += 4) {
        AnglesSse angle_pack;
        std::memcpy(angle_pack.angles, in + i, 4 * sizeof(float));

        SinCosSse result_pack;
        sincosSse(&angle_pack, &result_pack);

        std::memcpy(out_sin + i, result_pack.sins, 4 * sizeof(float));
        std::memcpy(out_cos + i, result_pack.coss, 4 * sizeof(float));
    }
#endif

    // Handle remaining elements with scalar
    for (; i < count; ++i) {
        Angle angle = { in[i] };
        SinCos sc;
        sincosScalar(angle, &sc);
        out_sin[i] = sc.sin;
        out_cos[i] = sc.cos;
    }
}

/**
 * @brief Compute sin for a batch of angles
 */
inline void sinBatch(const AngleBatch* MMATH_RESTRICT angles_,
                     float* MMATH_RESTRICT out_sin_) noexcept {
    // Use sincosBatch and discard cosine
    // (Modern CPUs make this nearly free due to pipelining)
    SinCosBatch batch;
    batch.sin_data = out_sin_;
    batch.cos_data = out_sin_;  // Dummy (will be overwritten)
    batch.count = angles_->count;
    sincosBatch(angles_, &batch);
}

/**
 * @brief Compute cos for a batch of angles
 */
inline void cosBatch(const AngleBatch* MMATH_RESTRICT angles_,
                     float* MMATH_RESTRICT out_cos_) noexcept {
    // Use sincosBatch and discard sine
    SinCosBatch batch;
    batch.sin_data = out_cos_;  // Dummy
    batch.cos_data = out_cos_;
    batch.count = angles_->count;
    sincosBatch(angles_, &batch);
}

// ============================================================================
// Convenience wrappers for C-style arrays
// ============================================================================

/**
 * @brief Compute sin/cos for plain arrays
 *
 * @param angles_ Input angles (C array)
 * @param out_sin_ Output sine values (C array)
 * @param out_cos_ Output cosine values (C array)
 * @param count_ Number of elements
 */
inline void sincosArray(const float* MMATH_RESTRICT angles_,
                        float* MMATH_RESTRICT out_sin_,
                        float* MMATH_RESTRICT out_cos_,
                        int32_t count_) noexcept {
    AngleBatch angle_batch = { angles_, count_ };
    SinCosBatch result_batch = { out_sin_, out_cos_, count_ };
    sincosBatch(&angle_batch, &result_batch);
}

/**
 * @brief Compute sin for plain array
 */
inline void sinArray(const float* MMATH_RESTRICT angles_,
                     float* MMATH_RESTRICT out_sin_,
                     int32_t count_) noexcept {
    AngleBatch angle_batch = { angles_, count_ };
    sinBatch(&angle_batch, out_sin_);
}

/**
 * @brief Compute cos for plain array
 */
inline void cosArray(const float* MMATH_RESTRICT angles_,
                     float* MMATH_RESTRICT out_cos_,
                     int32_t count_) noexcept {
    AngleBatch angle_batch = { angles_, count_ };
    cosBatch(&angle_batch, out_cos_);
}

// ============================================================================
// Single-value API (for completeness)
// ============================================================================

/**
 * @brief Compute sin/cos for a single angle (POD)
 */
inline void sincos(Angle angle_, SinCos* MMATH_RESTRICT result_) noexcept {
    sincosScalar(angle_, result_);
}

/**
 * @brief Compute sin for a single angle (returns float)
 */
inline float sin(float angle_) noexcept {
    SinCos result;
    sincosScalar(Angle{angle_}, &result);
    return result.sin;
}

/**
 * @brief Compute cos for a single angle (returns float)
 */
inline float cos(float angle_) noexcept {
    SinCos result;
    sincosScalar(Angle{angle_}, &result);
    return result.cos;
}

} // namespace MMath
