/**
 * @brief Fast normalize without zero-length check
 *
 * Uses rsqrt + one Newton-Raphson refinement step on SSE targets.
 * This is faster than the precise sqrt+divide path while keeping the
 * normalization error small enough for typical graphics workloads.
 * WARNING: Undefined behavior if input is zero vector
 */
MMATH_FORCE_INLINE Vec3 vec3NormalizeFast(Vec3 v_) noexcept {
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    float len_sq = v_.x * v_.x + v_.y * v_.y + v_.z * v_.z;
    __m128 len_sq_v = _mm_set_ss(len_sq);
    __m128 inv_len_v = _mm_rsqrt_ss(len_sq_v);
    const __m128 half = _mm_set_ss(0.5f);
    const __m128 three_halves = _mm_set_ss(1.5f);
    const __m128 inv_len_sq_v = _mm_mul_ss(inv_len_v, inv_len_v);
    inv_len_v = _mm_mul_ss(
        inv_len_v,
        _mm_sub_ss(three_halves, _mm_mul_ss(_mm_mul_ss(half, len_sq_v), inv_len_sq_v)));
    float inv_len = _mm_cvtss_f32(inv_len_v);
    return Vec3{ v_.x * inv_len, v_.y * inv_len, v_.z * inv_len };
#else
    float len_sq = v_.x * v_.x + v_.y * v_.y + v_.z * v_.z;
    float inv_len = 1.0f / std::sqrt(len_sq);
    return Vec3{ v_.x * inv_len, v_.y * inv_len, v_.z * inv_len };
#endif
}
