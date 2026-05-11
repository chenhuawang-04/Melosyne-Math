
// ============================================================================
// Mathematical Constants
// ============================================================================

namespace detail {
    // Natural logarithm of 2: ln(2)
    constexpr float kLn2 = 0.693147180559945309417232121458176568f;

    // Base-2 logarithm of e: log2(e) = 1/ln(2)
    constexpr float kLog2E = 1.44269504088896340735992468100189214f;

    // Base-10 logarithm of e: log10(e)
    constexpr float kLog10E = 0.434294481903251827651128918916605082f;

    // Natural logarithm of 10: ln(10)
    constexpr float kLn10 = 2.30258509299404568401799145468436421f;

    // exp(x) polynomial coefficients (minimax approximation for |x| < ln(2)/2)
    // Derived from cephes library: exp(r) ~= 1 + r*P(r^2)
    constexpr float kExpC1 = 1.0f;
    constexpr float kExpC2 = 0.5f;
    constexpr float kExpC3 = 0.166666666666666851703837234430154323f;  // 1/6
    constexpr float kExpC4 = 0.0416666666666665970564002484798622411f; // 1/24
    constexpr float kExpC5 = 0.00833333333333328885369231554654299327f; // 1/120

    // log(x) polynomial coefficients (minimax for sqrt(2)/2 < x < sqrt(2))
    // Approximates log(1+x) for small x
    constexpr float kLogP0 = 7.0376836292E-2f;
    constexpr float kLogP1 = -1.1514610310E-1f;
    constexpr float kLogP2 = 1.1676998740E-1f;
    constexpr float kLogP3 = -1.2420140846E-1f;
    constexpr float kLogP4 = 1.4249322787E-1f;
    constexpr float kLogP5 = -1.6668057665E-1f;
    constexpr float kLogP6 = 2.0000714765E-1f;
    constexpr float kLogP7 = -2.4999993993E-1f;
    constexpr float kLogP8 = 3.3333331174E-1f;

    // sqrt(2) and sqrt(2)/2 for log range adjustment
    constexpr float kSqrt2Over2 = 0.707106781186547524400844362104849039f;

    MMATH_FORCE_INLINE bool tryConvertIntegralExponent(float y_, int32_t* out_) noexcept {
        if (out_ == nullptr) {
            return false;
        }

        float integral = 0.0f;
        if (std::modf(y_, &integral) != 0.0f) {
            return false;
        }

        constexpr float kInt32Min = static_cast<float>(std::numeric_limits<int32_t>::min());
        constexpr float kInt32Max = static_cast<float>(std::numeric_limits<int32_t>::max());
        if (integral < kInt32Min || integral > kInt32Max) {
            return false;
        }

        *out_ = static_cast<int32_t>(integral);
        return true;
    }
}

// ============================================================================
// Exponential Functions
// ============================================================================

/**
 * @brief Natural exponential function: e^x
 *
 * Implementation:
 * - Range reduction: x = k*ln(2) + r, |r| < ln(2)/2
 * - Polynomial: exp(r) ~= 1 + r + r^2/2! + r^3/3! + r^4/4! + r^5/5!
 * - Reconstruction: exp(x) = 2^k * exp(r)
 *
 * @param x_ Exponent value
 * @return e raised to power x_
 *
 * Accuracy: tuned for throughput; current scalar tests validate ~2e-2 relative tolerance
 * Performance: typically faster than std::exp for hot scalar loops
 *
 * Use case: Exponential decay envelopes in audio processing
 */
MMATH_FORCE_INLINE float exp(float x_) noexcept {
    if (std::isnan(x_)) return x_;

    // Clamp to prevent overflow/underflow
    constexpr float kMaxInput = 88.376f;  // log(FLT_MAX)
    constexpr float kMinInput = -87.336f; // log(FLT_MIN)

    if (x_ > kMaxInput) return std::numeric_limits<float>::infinity();
    if (x_ < kMinInput) return 0.0f;

    // Range reduction: x = k*ln(2) + r
    float k = std::floor(x_ * detail::kLog2E + 0.5f);  // Round to nearest integer
    float r = x_ - k * detail::kLn2;  // Remainder: |r| < ln(2)/2

    // Polynomial approximation: exp(r) using Horner's method
    // exp(r) ~= 1 + r + r^2/2! + r^3/3! + r^4/4! + r^5/5!
    // Rewrite: exp(r) = 1 + r*(1 + r*(1/2 + r*(1/6 + r*(1/24 + r/120))))
    float poly = detail::kExpC5;  // 1/120
    poly = detail::kExpC4 + r * poly;  // 1/24 + r*poly
    poly = detail::kExpC3 + r * poly;  // 1/6 + r*poly
    poly = detail::kExpC2 + r * poly;  // 1/2 + r*poly
    poly = detail::kExpC1 + r * poly;  // 1 + r*poly
    poly = 1.0f + r * poly;  // 1 + r*(...)

    // Reconstruction: exp(x) = 2^k * exp(r)
    // Use bit manipulation to compute 2^k efficiently
    int32_t ki = static_cast<int32_t>(k);
    int32_t pow2_bits = (ki + 127) << 23;  // IEEE 754: exponent field
    float pow2;
    std::memcpy(&pow2, &pow2_bits, sizeof(float));

    return poly * pow2;
}

/**
 * @brief Base-2 exponential: 2^x
 *
 * Formula: exp2(x) = exp(x * ln(2))
 * More efficient than calling exp() with conversion
 *
 * @param x_ Exponent value
 * @return 2 raised to power x_
 */
MMATH_FORCE_INLINE float exp2(float x_) noexcept {
    return exp(x_ * detail::kLn2);
}

/**
 * @brief Base-10 exponential: 10^x
 *
 * Formula: pow10(x) = exp(x * ln(10))
 *
 * @param x_ Exponent value
 * @return 10 raised to power x_
 *
 * Use case: dB to linear conversion in audio
 * Example: linear_gain = pow10(gain_db / 20.0f)
 */
MMATH_FORCE_INLINE float pow10(float x_) noexcept {
    return exp(x_ * detail::kLn10);
}

// ============================================================================
// Logarithm Functions
// ============================================================================

/**
 * @brief Natural logarithm: ln(x)
 *
 * Implementation:
 * - Extract exponent: x = 2^e * m, where 1 <= m < 2
 * - Range adjust: If m < sqrt(2)/2, then m *= 2, e -= 1
 * - Polynomial: log(m) ~= P(m-1) for m near 1
 * - Reconstruction: log(x) = e*ln(2) + log(m)
 *
 * @param x_ Input value (must be positive)
 * @return Natural logarithm of x_
 *
 * @note Returns -inf for x_ == 0 and NaN for x_ < 0 or NaN
 *
 * Accuracy: tuned for throughput; current scalar tests validate ~2e-2 relative tolerance
 * Performance: typically faster than std::log for hot scalar loops
 *
 * Use case: Linear to dB conversion: db = 20 * log10(linear)
 */
MMATH_FORCE_INLINE float log(float x_) noexcept {
    // Handle special cases
    if (std::isnan(x_)) return x_;
    if (x_ == 0.0f) return -std::numeric_limits<float>::infinity();
    if (x_ < 0.0f) return std::numeric_limits<float>::quiet_NaN();

    // Extract exponent and mantissa using bit manipulation
    int32_t bits;
    std::memcpy(&bits, &x_, sizeof(float));

    int32_t exponent = ((bits >> 23) & 0xFF) - 127;  // Extract exponent field
    bits = (bits & 0x007FFFFF) | 0x3F800000;  // Force exponent to 0 (mantissa in [1, 2))
    float mantissa;
    std::memcpy(&mantissa, &bits, sizeof(float));

    // Range adjustment: if mantissa < sqrt(2)/2, multiply by 2 and decrement exponent
    if (mantissa < detail::kSqrt2Over2) {
        mantissa *= 2.0f;
        exponent -= 1;
    }

    // Transform to y = (m - 1) / (m + 1) for better approximation
    float y = mantissa - 1.0f;

    // Polynomial approximation for log(1+y) using Horner's method
    float z = y * y;
    float poly = detail::kLogP0;
    poly = detail::kLogP1 + y * poly;
    poly = detail::kLogP2 + y * poly;
    poly = detail::kLogP3 + y * poly;
    poly = detail::kLogP4 + y * poly;
    poly = detail::kLogP5 + y * poly;
    poly = detail::kLogP6 + y * poly;
    poly = detail::kLogP7 + y * poly;
    poly = detail::kLogP8 + y * poly;
    poly = y * poly * z;

    // Add linear term
    poly += -0.5f * z + y;

    // Reconstruction: log(x) = e*ln(2) + log(m)
    return static_cast<float>(exponent) * detail::kLn2 + poly;
}

/**
 * @brief Base-2 logarithm: log2(x)
 *
 * Formula: log2(x) = log(x) / ln(2)
 *
 * @param x_ Input value (must be positive)
 * @return Base-2 logarithm of x_
 */
MMATH_FORCE_INLINE float log2(float x_) noexcept {
    return log(x_) * detail::kLog2E;
}

/**
 * @brief Base-10 logarithm: log10(x)
 *
 * Formula: log10(x) = log(x) / ln(10)
 *
 * @param x_ Input value (must be positive)
 * @return Base-10 logarithm of x_
 *
 * Use case: Linear to dB conversion
 * Example: gain_db = 20.0f * log10(linear_gain)
 */
MMATH_FORCE_INLINE float log10(float x_) noexcept {
    return log(x_) * detail::kLog10E;
}

// ============================================================================
// Power Functions
// ============================================================================

MMATH_FORCE_INLINE float powi(float x_, int32_t n_) noexcept;

/**
 * @brief General power function: x^y
 *
 * Implementation: pow(x, y) = exp(y * log(x))
 *
 * @param x_ Base value
 * @param y_ Exponent value
 * @return x raised to power y
 *
 * @note Negative bases are supported only when y_ is an exact integral float.
 * @note Non-positive corner cases follow scalar math semantics where practical:
 *       pow(0, positive)=0, pow(0, negative)=+inf, pow(negative, non-integer)=NaN.
 *
 * Accuracy: current scalar tests validate ~5e-2 relative tolerance on the
 * positive-domain approximation path.
 * Performance: optimized for real-time positive-domain workloads.
 *
 * Use case: Distance attenuation in audio
 * Example: attenuation = pow(distance / min_distance, -rolloff)
 */
MMATH_FORCE_INLINE float pow(float x_, float y_) noexcept {
    // Handle special cases
    if (std::isnan(x_) || std::isnan(y_)) return std::numeric_limits<float>::quiet_NaN();
    if (y_ == 0.0f) return 1.0f;
    if (x_ == 1.0f) return 1.0f;
    if (y_ == 1.0f) return x_;

    if (x_ == 0.0f) {
        return y_ > 0.0f ? 0.0f : std::numeric_limits<float>::infinity();
    }

    if (x_ < 0.0f) {
        int32_t integral_exponent = 0;
        if (detail::tryConvertIntegralExponent(y_, &integral_exponent)) {
            return powi(x_, integral_exponent);
        }
        return std::numeric_limits<float>::quiet_NaN();
    }

    // General case: x^y = exp(y * log(x))
    return exp(y_ * log(x_));
}

/**
 * @brief Integer power function: x^n (optimized for integer exponents)
 *
 * Uses exponentiation by squaring for efficiency
 *
 * @param x_ Base value
 * @param n_ Integer exponent
 * @return x raised to power n
 *
 * Performance: O(log n) multiplications vs O(n) for naive approach
 *
 * Example: pow(2.0f, 10) = 1024.0f
 */
MMATH_FORCE_INLINE float powi(float x_, int32_t n_) noexcept {
    if (n_ == 0) return 1.0f;
    if (n_ == 1) return x_;
    if (n_ == -1) return 1.0f / x_;
    if (x_ == 0.0f) {
        return n_ < 0 ? std::numeric_limits<float>::infinity() : 0.0f;
    }

    // Handle negative exponents
    bool negative = n_ < 0;
    uint32_t exponent = negative
        ? static_cast<uint32_t>(-static_cast<int64_t>(n_))
        : static_cast<uint32_t>(n_);

    // Exponentiation by squaring
    float result = 1.0f;
    float base = x_;

    while (exponent != 0u) {
        if ((exponent & 1u) != 0u) {
            result *= base;
        }
        base *= base;
        exponent >>= 1u;
    }

    return negative ? (1.0f / result) : result;
}

