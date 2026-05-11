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

namespace FmTest {

class Rng {
public:
    explicit Rng(uint32_t seed_ = 0xC0FFEEU) : eng(seed_) {}

    float uniform(float lo_, float hi_) {
        std::uniform_real_distribution<float> dist(lo_, hi_);
        return dist(eng);
    }

    int uniformInt(int lo_, int hi_) {
        std::uniform_int_distribution<int> dist(lo_, hi_);
        return dist(eng);
    }

    uint32_t uniformU32(uint32_t lo_, uint32_t hi_) {
        std::uniform_int_distribution<uint32_t> dist(lo_, hi_);
        return dist(eng);
    }

private:
    std::mt19937 eng;
};

inline bool nearAbs(float a_, float b_, float eps_) {
    return std::fabs(a_ - b_) <= eps_;
}

inline bool nearRel(float a_, float b_, float rel_eps_, float abs_eps_ = 1e-6f) {
    const float diff = std::fabs(a_ - b_);
    if (diff <= abs_eps_) {
        return true;
    }
    return diff <= rel_eps_ * std::max(std::fabs(a_), std::fabs(b_));
}

inline void requireVec2Near(
    const MMath::Vec2& actual_,
    const MMath::Vec2& expected_,
    float eps_,
    const char* actual_expr_,
    const char* expected_expr_,
    const char* file_,
    int line_) {
    if (!nearAbs(actual_.x, expected_.x, eps_) || !nearAbs(actual_.y, expected_.y, eps_)) {
        std::ostringstream oss;
        oss << actual_expr_ << " = (" << actual_.x << ", " << actual_.y << "), "
            << expected_expr_ << " = (" << expected_.x << ", " << expected_.y << "), eps=" << eps_;
        fail("Vec2 near", oss.str(), file_, line_);
    }
}

inline void requireVec3Near(
    const MMath::Vec3& actual_,
    const MMath::Vec3& expected_,
    float eps_,
    const char* actual_expr_,
    const char* expected_expr_,
    const char* file_,
    int line_) {
    if (!nearAbs(actual_.x, expected_.x, eps_) || !nearAbs(actual_.y, expected_.y, eps_) ||
        !nearAbs(actual_.z, expected_.z, eps_)) {
        std::ostringstream oss;
        oss << actual_expr_ << " = (" << actual_.x << ", " << actual_.y << ", " << actual_.z << "), "
            << expected_expr_ << " = (" << expected_.x << ", " << expected_.y << ", " << expected_.z << "), eps="
            << eps_;
        fail("Vec3 near", oss.str(), file_, line_);
    }
}

inline void requireVec4Near(
    const MMath::Vec4& actual_,
    const MMath::Vec4& expected_,
    float eps_,
    const char* actual_expr_,
    const char* expected_expr_,
    const char* file_,
    int line_) {
    if (!nearAbs(actual_.x, expected_.x, eps_) || !nearAbs(actual_.y, expected_.y, eps_) ||
        !nearAbs(actual_.z, expected_.z, eps_) || !nearAbs(actual_.w, expected_.w, eps_)) {
        std::ostringstream oss;
        oss << actual_expr_ << " = (" << actual_.x << ", " << actual_.y << ", " << actual_.z << ", " << actual_.w
            << "), " << expected_expr_ << " = (" << expected_.x << ", " << expected_.y << ", " << expected_.z
            << ", " << expected_.w << "), eps=" << eps_;
        fail("Vec4 near", oss.str(), file_, line_);
    }
}

inline void requireMat3Near(
    const MMath::Mat3& actual_,
    const MMath::Mat3& expected_,
    float eps_,
    const char* actual_expr_,
    const char* expected_expr_,
    const char* file_,
    int line_) {
    const float a[9] = {
        actual_.m00, actual_.m01, actual_.m02, actual_.m10, actual_.m11,
        actual_.m12, actual_.m20, actual_.m21, actual_.m22,
    };
    const float e[9] = {
        expected_.m00, expected_.m01, expected_.m02, expected_.m10, expected_.m11,
        expected_.m12, expected_.m20, expected_.m21, expected_.m22,
    };
    for (int i = 0; i < 9; ++i) {
        if (!nearAbs(a[i], e[i], eps_)) {
            std::ostringstream oss;
            oss << actual_expr_ << " differs from " << expected_expr_ << " at index " << i
                << ": actual=" << a[i] << ", expected=" << e[i] << ", eps=" << eps_;
            fail("Mat3 near", oss.str(), file_, line_);
        }
    }
}

inline void requireMat4Near(
    const MMath::Mat4& actual_,
    const MMath::Mat4& expected_,
    float eps_,
    const char* actual_expr_,
    const char* expected_expr_,
    const char* file_,
    int line_) {
    for (int i = 0; i < 16; ++i) {
        if (!nearAbs(actual_.m[i], expected_.m[i], eps_)) {
            std::ostringstream oss;
            oss << actual_expr_ << " differs from " << expected_expr_ << " at m[" << i
                << "]: actual=" << actual_.m[i] << ", expected=" << expected_.m[i]
                << ", eps=" << eps_;
            fail("Mat4 near", oss.str(), file_, line_);
        }
    }
}

inline MMath::Vec3 randomVec3(Rng& rng_, float lo_ = -10.0f, float hi_ = 10.0f) {
    return MMath::Vec3{rng_.uniform(lo_, hi_), rng_.uniform(lo_, hi_), rng_.uniform(lo_, hi_)};
}

inline MMath::Vec4 randomVec4(Rng& rng_, float lo_ = -10.0f, float hi_ = 10.0f) {
    return MMath::Vec4{rng_.uniform(lo_, hi_), rng_.uniform(lo_, hi_), rng_.uniform(lo_, hi_), rng_.uniform(lo_, hi_)};
}

inline MMath::Mat4 randomMat4(Rng& rng_, float lo_ = -2.0f, float hi_ = 2.0f) {
    MMath::Mat4 m{};
    for (float& v : m.m) {
        v = rng_.uniform(lo_, hi_);
    }
    return m;
}

} // namespace FmTest

#define FM_REQUIRE_VEC2_NEAR(ACTUAL, EXPECTED, EPS)                                           \
    ::FmTest::requireVec2Near((ACTUAL), (EXPECTED), (EPS), #ACTUAL, #EXPECTED, __FILE__, __LINE__)

#define FM_REQUIRE_VEC3_NEAR(ACTUAL, EXPECTED, EPS)                                           \
    ::FmTest::requireVec3Near((ACTUAL), (EXPECTED), (EPS), #ACTUAL, #EXPECTED, __FILE__, __LINE__)

#define FM_REQUIRE_VEC4_NEAR(ACTUAL, EXPECTED, EPS)                                           \
    ::FmTest::requireVec4Near((ACTUAL), (EXPECTED), (EPS), #ACTUAL, #EXPECTED, __FILE__, __LINE__)

#define FM_REQUIRE_MAT3_NEAR(ACTUAL, EXPECTED, EPS)                                           \
    ::FmTest::requireMat3Near((ACTUAL), (EXPECTED), (EPS), #ACTUAL, #EXPECTED, __FILE__, __LINE__)

#define FM_REQUIRE_MAT4_NEAR(ACTUAL, EXPECTED, EPS)                                           \
    ::FmTest::requireMat4Near((ACTUAL), (EXPECTED), (EPS), #ACTUAL, #EXPECTED, __FILE__, __LINE__)
