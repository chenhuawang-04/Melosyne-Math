#pragma once

#include "fast_math/types.h"

#if FM_HAVE_GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#endif

#if FM_HAVE_EIGEN
#include <Eigen/Dense>
#endif

#if FM_HAVE_DIRECTXMATH
#include <DirectXMath.h>
#endif

namespace FmBench::Adapters {

constexpr bool has_glm = FM_HAVE_GLM;
constexpr bool has_eigen = FM_HAVE_EIGEN;
constexpr bool has_directxmath = FM_HAVE_DIRECTXMATH;

#if FM_HAVE_GLM
inline glm::vec3 toGlm(const MMath::Vec3& v) { return glm::vec3(v.x, v.y, v.z); }
inline glm::vec4 toGlm(const MMath::Vec4& v) { return glm::vec4(v.x, v.y, v.z, v.w); }
inline glm::mat4 toGlm(const MMath::Mat4& m) {
    glm::mat4 out(1.0f);
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            out[c][r] = m.m[c * 4 + r];
        }
    }
    return out;
}
#endif

#if FM_HAVE_EIGEN
inline Eigen::Vector3f toEigen(const MMath::Vec3& v) { return Eigen::Vector3f(v.x, v.y, v.z); }
inline Eigen::Vector4f toEigen(const MMath::Vec4& v) { return Eigen::Vector4f(v.x, v.y, v.z, v.w); }
inline Eigen::Matrix4f toEigen(const MMath::Mat4& m) {
    Eigen::Matrix4f out;
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            out(r, c) = m.m[c * 4 + r];
        }
    }
    return out;
}
#endif

#if FM_HAVE_DIRECTXMATH
inline DirectX::XMVECTOR toDxm(const MMath::Vec3& v) {
    return DirectX::XMVectorSet(v.x, v.y, v.z, 0.0f);
}
inline DirectX::XMVECTOR toDxm(const MMath::Vec4& v) {
    return DirectX::XMVectorSet(v.x, v.y, v.z, v.w);
}
#endif

} // namespace FmBench::Adapters
