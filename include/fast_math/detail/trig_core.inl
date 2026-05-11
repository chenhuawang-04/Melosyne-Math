// Shared trig high-level API implementation for header/module parity.

namespace detail {

template <bool WriteSin, bool WriteCos>
MMATH_FORCE_INLINE void trigBatchDispatch(
    const float* MMATH_RESTRICT in_,
    float* MMATH_RESTRICT out_sin_,
    float* MMATH_RESTRICT out_cos_,
    int32_t count_
) noexcept {
    int32_t i = 0;

#if defined(__AVX2__) && defined(__FMA__)
    const int32_t avx2_count = (count_ / 8) * 8;
    for (; i < avx2_count; i += 8) {
        AnglesAvx2 angle_pack;
        std::memcpy(angle_pack.angles, in_ + i, 8 * sizeof(float));

        SinCosAvx2 result_pack;
        sincosAvx2(&angle_pack, &result_pack);

        if constexpr (WriteSin) {
            std::memcpy(out_sin_ + i, result_pack.sins, 8 * sizeof(float));
        }
        if constexpr (WriteCos) {
            std::memcpy(out_cos_ + i, result_pack.coss, 8 * sizeof(float));
        }
    }
#elif defined(__SSE4_1__)
    const int32_t sse_count = (count_ / 4) * 4;
    for (; i < sse_count; i += 4) {
        AnglesSse angle_pack;
        std::memcpy(angle_pack.angles, in_ + i, 4 * sizeof(float));

        SinCosSse result_pack;
        sincosSse(&angle_pack, &result_pack);

        if constexpr (WriteSin) {
            std::memcpy(out_sin_ + i, result_pack.sins, 4 * sizeof(float));
        }
        if constexpr (WriteCos) {
            std::memcpy(out_cos_ + i, result_pack.coss, 4 * sizeof(float));
        }
    }
#endif

    for (; i < count_; ++i) {
        SinCos sc;
        sincosScalar(Angle{in_[i]}, &sc);
        if constexpr (WriteSin) {
            out_sin_[i] = sc.sin;
        }
        if constexpr (WriteCos) {
            out_cos_[i] = sc.cos;
        }
    }
}

} // namespace detail

// ============================================================================
// Batch Processing API (POD + Free Functions)
// ============================================================================

inline void sincosBatch(const AngleBatch* MMATH_RESTRICT angles_,
                        SinCosBatch* MMATH_RESTRICT result_) noexcept {
    detail::trigBatchDispatch<true, true>(
        angles_->data,
        result_->sin_data,
        result_->cos_data,
        angles_->count
    );
}

inline void sinBatch(const AngleBatch* MMATH_RESTRICT angles_,
                     float* MMATH_RESTRICT out_sin_) noexcept {
    detail::trigBatchDispatch<true, false>(
        angles_->data,
        out_sin_,
        nullptr,
        angles_->count
    );
}

inline void cosBatch(const AngleBatch* MMATH_RESTRICT angles_,
                     float* MMATH_RESTRICT out_cos_) noexcept {
    detail::trigBatchDispatch<false, true>(
        angles_->data,
        nullptr,
        out_cos_,
        angles_->count
    );
}

// ============================================================================
// Convenience wrappers for C-style arrays
// ============================================================================

inline void sincosArray(const float* MMATH_RESTRICT angles_,
                        float* MMATH_RESTRICT out_sin_,
                        float* MMATH_RESTRICT out_cos_,
                        int32_t count_) noexcept {
    AngleBatch angle_batch = { angles_, count_ };
    SinCosBatch result_batch = { out_sin_, out_cos_, count_ };
    sincosBatch(&angle_batch, &result_batch);
}

inline void sinArray(const float* MMATH_RESTRICT angles_,
                     float* MMATH_RESTRICT out_sin_,
                     int32_t count_) noexcept {
    AngleBatch angle_batch = { angles_, count_ };
    sinBatch(&angle_batch, out_sin_);
}

inline void cosArray(const float* MMATH_RESTRICT angles_,
                     float* MMATH_RESTRICT out_cos_,
                     int32_t count_) noexcept {
    AngleBatch angle_batch = { angles_, count_ };
    cosBatch(&angle_batch, out_cos_);
}

// ============================================================================
// Single-value API (for completeness)
// ============================================================================

inline void sincos(Angle angle_, SinCos* MMATH_RESTRICT result_) noexcept {
    sincosScalar(angle_, result_);
}

inline float sin(float angle_) noexcept {
    SinCos result;
    sincosScalar(Angle{angle_}, &result);
    return result.sin;
}

inline float cos(float angle_) noexcept {
    SinCos result;
    sincosScalar(Angle{angle_}, &result);
    return result.cos;
}
