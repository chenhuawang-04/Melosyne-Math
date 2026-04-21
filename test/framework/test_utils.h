#pragma once

#include "test/framework/test_framework.h"

#include "fast_math/types.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace fmtest {

class Rng {
public:
    explicit Rng(uint32_t seed = 0xC0FFEEU) : eng_(seed) {}

    float uniform(float lo, float hi) {
        std::uniform_real_distribution<float> dist(lo, hi);
        return dist(eng_);
    }

    int uniform_int(int lo, int hi) {
        std::uniform_int_distribution<int> dist(lo, hi);
        return dist(eng_);
    }

    uint32_t uniform_u32(uint32_t lo, uint32_t hi) {
        std::uniform_int_distribution<uint32_t> dist(lo, hi);
        return dist(eng_);
    }

private:
    std::mt19937 eng_;
};

inline bool near_abs(float a, float b, float eps) {
    return std::fabs(a - b) <= eps;
}

inline bool near_rel(float a, float b, float rel_eps, float abs_eps = 1e-6f) {
    const float diff = std::fabs(a - b);
    if (diff <= abs_eps) {
        return true;
    }
    return diff <= rel_eps * std::max(std::fabs(a), std::fabs(b));
}

inline void require_vec2_near(
    const MMath::Vec2& actual,
    const MMath::Vec2& expected,
    float eps,
    const char* actual_expr,
    const char* expected_expr,
    const char* file,
    int line) {
    if (!near_abs(actual.x, expected.x, eps) || !near_abs(actual.y, expected.y, eps)) {
        std::ostringstream oss;
        oss << actual_expr << " = (" << actual.x << ", " << actual.y << "), "
            << expected_expr << " = (" << expected.x << ", " << expected.y << "), eps=" << eps;
        fail("Vec2 near", oss.str(), file, line);
    }
}

inline void require_vec3_near(
    const MMath::Vec3& actual,
    const MMath::Vec3& expected,
    float eps,
    const char* actual_expr,
    const char* expected_expr,
    const char* file,
    int line) {
    if (!near_abs(actual.x, expected.x, eps) || !near_abs(actual.y, expected.y, eps) ||
        !near_abs(actual.z, expected.z, eps)) {
        std::ostringstream oss;
        oss << actual_expr << " = (" << actual.x << ", " << actual.y << ", " << actual.z << "), "
            << expected_expr << " = (" << expected.x << ", " << expected.y << ", " << expected.z << "), eps="
            << eps;
        fail("Vec3 near", oss.str(), file, line);
    }
}

inline void require_vec4_near(
    const MMath::Vec4& actual,
    const MMath::Vec4& expected,
    float eps,
    const char* actual_expr,
    const char* expected_expr,
    const char* file,
    int line) {
    if (!near_abs(actual.x, expected.x, eps) || !near_abs(actual.y, expected.y, eps) ||
        !near_abs(actual.z, expected.z, eps) || !near_abs(actual.w, expected.w, eps)) {
        std::ostringstream oss;
        oss << actual_expr << " = (" << actual.x << ", " << actual.y << ", " << actual.z << ", " << actual.w
            << "), " << expected_expr << " = (" << expected.x << ", " << expected.y << ", " << expected.z
            << ", " << expected.w << "), eps=" << eps;
        fail("Vec4 near", oss.str(), file, line);
    }
}

inline void require_mat3_near(
    const MMath::Mat3& actual,
    const MMath::Mat3& expected,
    float eps,
    const char* actual_expr,
    const char* expected_expr,
    const char* file,
    int line) {
    const float a[9] = {
        actual.m00, actual.m01, actual.m02, actual.m10, actual.m11,
        actual.m12, actual.m20, actual.m21, actual.m22,
    };
    const float e[9] = {
        expected.m00, expected.m01, expected.m02, expected.m10, expected.m11,
        expected.m12, expected.m20, expected.m21, expected.m22,
    };
    for (int i = 0; i < 9; ++i) {
        if (!near_abs(a[i], e[i], eps)) {
            std::ostringstream oss;
            oss << actual_expr << " differs at index " << i << ": actual=" << a[i]
                << ", expected=" << e[i] << ", eps=" << eps;
            fail("Mat3 near", oss.str(), file, line);
        }
    }
}

inline void require_mat4_near(
    const MMath::Mat4& actual,
    const MMath::Mat4& expected,
    float eps,
    const char* actual_expr,
    const char* expected_expr,
    const char* file,
    int line) {
    for (int i = 0; i < 16; ++i) {
        if (!near_abs(actual.m[i], expected.m[i], eps)) {
            std::ostringstream oss;
            oss << actual_expr << " differs at m[" << i << "]: actual=" << actual.m[i]
                << ", expected=" << expected.m[i] << ", eps=" << eps;
            fail("Mat4 near", oss.str(), file, line);
        }
    }
}

inline MMath::Vec3 random_vec3(Rng& rng, float lo = -10.0f, float hi = 10.0f) {
    return MMath::Vec3{rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(lo, hi)};
}

inline MMath::Vec4 random_vec4(Rng& rng, float lo = -10.0f, float hi = 10.0f) {
    return MMath::Vec4{rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(lo, hi)};
}

inline MMath::Mat4 random_mat4(Rng& rng, float lo = -2.0f, float hi = 2.0f) {
    MMath::Mat4 m{};
    for (float& v : m.m) {
        v = rng.uniform(lo, hi);
    }
    return m;
}

} // namespace fmtest

#define FM_REQUIRE_VEC2_NEAR(ACTUAL, EXPECTED, EPS)                                           \
    ::fmtest::require_vec2_near((ACTUAL), (EXPECTED), (EPS), #ACTUAL, #EXPECTED, __FILE__, __LINE__)

#define FM_REQUIRE_VEC3_NEAR(ACTUAL, EXPECTED, EPS)                                           \
    ::fmtest::require_vec3_near((ACTUAL), (EXPECTED), (EPS), #ACTUAL, #EXPECTED, __FILE__, __LINE__)

#define FM_REQUIRE_VEC4_NEAR(ACTUAL, EXPECTED, EPS)                                           \
    ::fmtest::require_vec4_near((ACTUAL), (EXPECTED), (EPS), #ACTUAL, #EXPECTED, __FILE__, __LINE__)

#define FM_REQUIRE_MAT3_NEAR(ACTUAL, EXPECTED, EPS)                                           \
    ::fmtest::require_mat3_near((ACTUAL), (EXPECTED), (EPS), #ACTUAL, #EXPECTED, __FILE__, __LINE__)

#define FM_REQUIRE_MAT4_NEAR(ACTUAL, EXPECTED, EPS)                                           \
    ::fmtest::require_mat4_near((ACTUAL), (EXPECTED), (EPS), #ACTUAL, #EXPECTED, __FILE__, __LINE__)
