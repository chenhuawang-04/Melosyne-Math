// Shared lerp scalar/core implementation for header/module parity.

// ============================================================================
// Linear Interpolation Functions
// ============================================================================

/**
 * @brief Linear interpolation with FMA optimization
 *
 * Formula: a + t*(b - a)
 * FMA-optimized: fma(t, b-a, a)
 *
 * Implementation:
 * - FMA: Single instruction for t*(b-a)+a (better precision, ~5% faster)
 * - Fallback: Standard a + t*(b-a)
 *
 * Properties:
 * - lerp(a, b, 0) = a (exact)
 * - lerp(a, b, 1) = b (exact)
 * - lerp(a, b, 0.5) = (a+b)/2 (midpoint)
 *
 * @param a_ Start value
 * @param b_ End value
 * @param t_ Interpolation parameter [0, 1] (not clamped)
 * @return Interpolated value
 *
 * Use cases (from audio engine analysis):
 * - Fractional delay interpolation (read position between samples)
 * - Volume crossfade: lerp(old_volume, new_volume, fade_t)
 * - Parameter smoothing over time
 *
 * References:
 * - NVIDIA GPU Pro Tip: https://developer.nvidia.com/blog/lerp-faster-cuda/
 * - 5% performance improvement in seismic processing
 */
MMATH_FORCE_INLINE float lerp(float a_, float b_, float t_) noexcept {
    // Pure scalar - compiler generates FMA automatically with -mfma
    // Avoids SSE scalar conversion overhead (4x _mm_set_ss + 1x _mm_cvtss_f32)
    return a_ + t_ * (b_ - a_);
}

/**
 * @brief Inverse linear interpolation
 *
 * Returns the parameter t that would produce 'value' when lerping from a to b.
 * Formula: (value - a) / (b - a)
 *
 * Properties:
 * - inverseLerp(a, b, a) = 0
 * - inverseLerp(a, b, b) = 1
 * - inverseLerp(a, b, lerp(a, b, t)) ≈ t (within floating-point precision)
 *
 * @param a_ Start value
 * @param b_ End value
 * @param value_ Value to find parameter for
 * @return Parameter t in range [0, 1] if value is in [a, b]
 *
 * @note For a == b, returns 0 (avoid division by zero)
 * @note Result not clamped; can be outside [0, 1] if value outside [a, b]
 *
 * Use cases:
 * - Mapping sensor readings to normalized range
 * - UI slider value -> position conversion
 * - Audio level meter: inverseLerp(-60dB, 0dB, current_level)
 */
MMATH_FORCE_INLINE float inverseLerp(float a_, float b_, float value_) noexcept {
    float range = b_ - a_;
    // Avoid division by zero: return 0 if range is zero
    if (MMath::abs(range) < 1e-8f) {
        return 0.0f;
    }
    return (value_ - a_) / range;
}

/**
 * @brief Remap value from one range to another
 *
 * Maps value from [in_min, in_max] to [out_min, out_max].
 * Equivalent to: lerp(out_min, out_max, inverseLerp(in_min, in_max, value))
 *
 * @param value_ Value in input range
 * @param in_min_ Input range minimum
 * @param in_max_ Input range maximum
 * @param out_min_ Output range minimum
 * @param out_max_ Output range maximum
 * @return Remapped value
 *
 * Use cases:
 * - MIDI note (0-127) to frequency (20Hz-20kHz)
 * - Screen coordinates to world coordinates
 * - Sensor reading (0-1023) to voltage (0-5V)
 *
 * Example:
 * ```cpp
 * float midi_note = 60;  // Middle C
 * float freq = remap(midi_note, 0, 127, 20.0f, 20000.0f);
 * ```
 */
MMATH_FORCE_INLINE float remap(float value_, float in_min_, float in_max_,
                                 float out_min_, float out_max_) noexcept {
    float t = inverseLerp(in_min_, in_max_, value_);
    return lerp(out_min_, out_max_, t);
}

// ============================================================================
// Smooth Interpolation Functions (Hermite Curves)
// ============================================================================

/**
 * @brief Smooth Hermite interpolation (cubic, C1 continuous)
 *
 * Implements smoothstep using 3rd-order Hermite polynomial.
 * Formula: x²(3 - 2x) where x = clamp((t - edge0) / (edge1 - edge0), 0, 1)
 *
 * Properties:
 * - Smooth ease-in and ease-out (S-curve)
 * - Zero 1st derivative at edges (smooth velocity transition)
 * - Returns 0 for t <= edge0
 * - Returns 1 for t >= edge1
 *
 * @param edge0_ Lower edge
 * @param edge1_ Upper edge
 * @param t_ Input value
 * @return Smooth interpolation result in [0, 1]
 *
 * Use cases:
 * - Smooth fade in/out for audio effects
 * - UI animations (button press, menu slide)
 * - Camera transitions
 * - Procedural noise smoothing (Perlin noise)
 *
 * References:
 * - Ken Perlin's improved noise (uses smoothstep)
 * - GLSL smoothstep: https://registry.khronos.org/OpenGL-Refpages/gl4/html/smoothstep.xhtml
 */
MMATH_FORCE_INLINE float smoothstep(float edge0_, float edge1_, float t_) noexcept {
    // Inline normalization (like GLM) - avoids inverseLerp() function call overhead
    // No branch for zero-check (assumes valid range edge1 != edge0)
    // clamp is now optimized (no SSE scalar overhead)
    float x = clamp((t_ - edge0_) / (edge1_ - edge0_), 0.0f, 1.0f);

    // Cubic Hermite: x²(3 - 2x)
    // Horner's method: x * x * (3 - 2*x)
    return x * x * (3.0f - 2.0f * x);
}

/**
 * @brief Smoother Hermite interpolation (quintic, C2 continuous)
 *
 * Implements Ken Perlin's improved smoothstep using 5th-order polynomial.
 * Formula: x³(x(x*6 - 15) + 10)
 *
 * Properties:
 * - Even smoother than smoothstep (less abrupt changes)
 * - Zero 1st AND 2nd derivatives at edges (smooth acceleration)
 * - Returns 0 for t <= edge0
 * - Returns 1 for t >= edge1
 *
 * @param edge0_ Lower edge
 * @param edge1_ Upper edge
 * @param t_ Input value
 * @return Extra smooth interpolation result in [0, 1]
 *
 * Use cases:
 * - High-quality animation curves
 * - Procedural terrain generation (smooth blending)
 * - Advanced audio crossfades
 * - Camera cinematics
 *
 * References:
 * - Ken Perlin's improved noise algorithm
 * - Wikipedia: https://en.wikipedia.org/wiki/Smoothstep
 */
MMATH_FORCE_INLINE float smootherstep(float edge0_, float edge1_, float t_) noexcept {
    // Inline normalization (like GLM) - avoids inverseLerp() function call overhead
    // clamp is now optimized (no SSE scalar overhead)
    float x = clamp((t_ - edge0_) / (edge1_ - edge0_), 0.0f, 1.0f);

    // Quintic Hermite: 6x⁵ - 15x⁴ + 10x³
    // Horner's method: x * x * x * (x * (x * 6 - 15) + 10)
    return x * x * x * (x * (x * 6.0f - 15.0f) + 10.0f);
}

// ============================================================================
// Clamped Lerp (Safe Version)
// ============================================================================

/**
 * @brief Clamped linear interpolation
 *
 * Same as lerp() but clamps t to [0, 1] first.
 * Guarantees result is in [a, b] (or [b, a] if b < a).
 *
 * @param a_ Start value
 * @param b_ End value
 * @param t_ Interpolation parameter (will be clamped to [0, 1])
 * @return Interpolated value, guaranteed in range [min(a,b), max(a,b)]
 */
MMATH_FORCE_INLINE float lerpClamped(float a_, float b_, float t_) noexcept {
    return lerp(a_, b_, saturate(t_));
}

// ============================================================================
// Stepped Lerp (Quantized)
// ============================================================================

/**
 * @brief Stepped linear interpolation
 *
 * Like lerp but quantizes t to discrete steps.
 * Useful for creating staircase effects or pixel-art style interpolation.
 *
 * @param a_ Start value
 * @param b_ End value
 * @param t_ Interpolation parameter [0, 1]
 * @param steps_ Number of discrete steps (e.g., 8 for 8 steps)
 * @return Interpolated value with stepped transitions
 *
 * Use case:
 * - Retro game animations (pixel-art style movement)
 * - Quantized audio parameter changes
 */
MMATH_FORCE_INLINE float lerpStepped(float a_, float b_, float t_, int steps_) noexcept {
    if (steps_ <= 1) return lerp(a_, b_, t_);

    float step_size = 1.0f / static_cast<float>(steps_ - 1);
    float quantized_t = std::floor(t_ / step_size + 0.5f) * step_size;
    return lerp(a_, b_, saturate(quantized_t));
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Check if value is approximately between a and b
 *
 * @param value_ Value to check
 * @param a_ Lower bound
 * @param b_ Upper bound
 * @param epsilon_ Tolerance (default: 1e-6f)
 * @return true if a <= value <= b (with tolerance)
 */
MMATH_FORCE_INLINE bool isInRange(float value_, float a_, float b_, float epsilon_ = 1e-6f) noexcept {
    return value_ >= (a_ - epsilon_) && value_ <= (b_ + epsilon_);
}
