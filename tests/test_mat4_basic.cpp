/**
 * @file test_mat4_basic.cpp
 * @brief Basic correctness tests for Mat4 operations
 *
 * Tests all 14 Mat4 functions for correctness
 */

#include "fast_math/mat4.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace MMath;

constexpr float TEST_EPSILON = 1e-5f;

bool test_passed = true;

void assert_float_equal(const char* test_name, float result, float expected) {
    if (std::fabs(result - expected) > TEST_EPSILON) {
        std::cout << "[FAIL] " << test_name << "\n";
        std::cout << "  Expected: " << expected << "\n";
        std::cout << "  Got:      " << result << "\n";
        test_passed = false;
    } else {
        std::cout << "[PASS] " << test_name << "\n";
    }
}

void assert_vec4_equal(const char* test_name, const Vec4& result, const Vec4& expected) {
    if (!vec4NearEqual(result, expected, TEST_EPSILON)) {
        std::cout << "[FAIL] " << test_name << "\n";
        std::cout << "  Expected: (" << expected.x << ", " << expected.y << ", " << expected.z << ", " << expected.w << ")\n";
        std::cout << "  Got:      (" << result.x << ", " << result.y << ", " << result.z << ", " << result.w << ")\n";
        test_passed = false;
    } else {
        std::cout << "[PASS] " << test_name << "\n";
    }
}

void assert_mat4_equal(const char* test_name, const Mat4& result, const Mat4& expected) {
    bool equal = true;
    for (int i = 0; i < 16; ++i) {
        if (std::fabs(result.m[i] - expected.m[i]) > TEST_EPSILON) {
            equal = false;
            break;
        }
    }

    if (!equal) {
        std::cout << "[FAIL] " << test_name << "\n";
        std::cout << "  Expected:\n";
        for (int row = 0; row < 4; ++row) {
            std::cout << "  [";
            for (int col = 0; col < 4; ++col) {
                std::cout << std::setw(10) << std::fixed << std::setprecision(4) << expected.m[col * 4 + row] << " ";
            }
            std::cout << "]\n";
        }
        std::cout << "  Got:\n";
        for (int row = 0; row < 4; ++row) {
            std::cout << "  [";
            for (int col = 0; col < 4; ++col) {
                std::cout << std::setw(10) << std::fixed << std::setprecision(4) << result.m[col * 4 + row] << " ";
            }
            std::cout << "]\n";
        }
        test_passed = false;
    } else {
        std::cout << "[PASS] " << test_name << "\n";
    }
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  Mat4 Basic Correctness Tests\n";
    std::cout << "========================================\n\n";

    // Test 1: Identity matrix
    std::cout << "Core Matrix Operations:\n";
    std::cout << "------------------------------\n";

    Mat4 identity = mat4Identity();
    Mat4 expected_identity = {{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    }};
    assert_mat4_equal("mat4Identity", identity, expected_identity);

    // Test 2: Matrix multiplication (identity * identity = identity)
    Mat4 mul_result = mat4Mul(identity, identity);
    assert_mat4_equal("mat4Mul (I * I = I)", mul_result, identity);

    // Test 3: Matrix-vector multiplication (I * v = v)
    Vec4 v = {1.0f, 2.0f, 3.0f, 1.0f};
    Vec4 mv_result = mat4MulVec4(identity, v);
    assert_vec4_equal("mat4MulVec4 (I * v = v)", mv_result, v);

    // Test 4: Transpose (I^T = I)
    Mat4 transpose_result = mat4Transpose(identity);
    assert_mat4_equal("mat4Transpose (I^T = I)", transpose_result, identity);

    // Test 5: Inverse (I^-1 = I)
    Mat4 inverse_result = mat4Inverse(identity);
    assert_mat4_equal("mat4Inverse (I^-1 = I)", inverse_result, identity);

    // Test 6: Translation matrix
    std::cout << "\nTransform Matrices:\n";
    std::cout << "------------------------------\n";

    Mat4 translation = mat4Translation(1.0f, 2.0f, 3.0f);
    Vec4 point = {0.0f, 0.0f, 0.0f, 1.0f};  // w=1 for point
    Vec4 translated = mat4MulVec4(translation, point);
    assert_vec4_equal("mat4Translation", translated, Vec4{1.0f, 2.0f, 3.0f, 1.0f});

    // Test 7: Scale matrix
    Mat4 scale = mat4Scale(2.0f, 3.0f, 4.0f);
    Vec4 vec_to_scale = {1.0f, 1.0f, 1.0f, 1.0f};
    Vec4 scaled = mat4MulVec4(scale, vec_to_scale);
    assert_vec4_equal("mat4Scale", scaled, Vec4{2.0f, 3.0f, 4.0f, 1.0f});

    // Test 8-10: Rotation matrices (90 degrees around each axis)
    std::cout << "\nRotation Matrices:\n";
    std::cout << "------------------------------\n";

    const float PI = 3.14159265358979323846f;
    const float PI_2 = PI / 2.0f;

    // Rotation X: (0, 1, 0) -> (0, 0, 1)
    Mat4 rotX = mat4RotationX(PI_2);
    Vec4 y_axis = {0.0f, 1.0f, 0.0f, 0.0f};
    Vec4 rotX_result = mat4MulVec4(rotX, y_axis);
    assert_vec4_equal("mat4RotationX (90°)", rotX_result, Vec4{0.0f, 0.0f, 1.0f, 0.0f});

    // Rotation Y: (0, 0, 1) -> (1, 0, 0)
    Mat4 rotY = mat4RotationY(PI_2);
    Vec4 z_axis = {0.0f, 0.0f, 1.0f, 0.0f};
    Vec4 rotY_result = mat4MulVec4(rotY, z_axis);
    assert_vec4_equal("mat4RotationY (90°)", rotY_result, Vec4{1.0f, 0.0f, 0.0f, 0.0f});

    // Rotation Z: (1, 0, 0) -> (0, 1, 0)
    Mat4 rotZ = mat4RotationZ(PI_2);
    Vec4 x_axis = {1.0f, 0.0f, 0.0f, 0.0f};
    Vec4 rotZ_result = mat4MulVec4(rotZ, x_axis);
    assert_vec4_equal("mat4RotationZ (90°)", rotZ_result, Vec4{0.0f, 1.0f, 0.0f, 0.0f});

    // Test 11: Quaternion to matrix (identity quaternion)
    std::cout << "\nQuaternion Conversion:\n";
    std::cout << "------------------------------\n";

    Vec4 identity_quat = {0.0f, 0.0f, 0.0f, 1.0f};  // Identity quaternion
    Mat4 quat_mat = mat4FromQuat(identity_quat);
    assert_mat4_equal("mat4FromQuat (identity)", quat_mat, identity);

    // Test 12: Perspective projection (basic construction)
    std::cout << "\nProjection Matrices:\n";
    std::cout << "------------------------------\n";

    Mat4 perspective = mat4Perspective(PI / 4.0f, 16.0f / 9.0f, 0.1f, 100.0f);
    // Just verify it doesn't crash and produces reasonable values
    bool perspective_valid =
        std::isfinite(perspective.m[0]) &&
        std::isfinite(perspective.m[5]) &&
        std::isfinite(perspective.m[10]) &&
        perspective.m[14] < 0.0f;  // Should be negative
    if (perspective_valid) {
        std::cout << "[PASS] mat4Perspective (construction)\n";
    } else {
        std::cout << "[FAIL] mat4Perspective (construction)\n";
        test_passed = false;
    }

    // Test 13: Orthographic projection
    Mat4 ortho = mat4Ortho(-1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 100.0f);
    bool ortho_valid =
        std::isfinite(ortho.m[0]) &&
        std::isfinite(ortho.m[5]) &&
        std::isfinite(ortho.m[10]);
    if (ortho_valid) {
        std::cout << "[PASS] mat4Ortho (construction)\n";
    } else {
        std::cout << "[FAIL] mat4Ortho (construction)\n";
        test_passed = false;
    }

    // Test 14: LookAt matrix
    Vec4 eye = {0.0f, 0.0f, 5.0f, 1.0f};
    Vec4 target = {0.0f, 0.0f, 0.0f, 1.0f};
    Vec4 up = {0.0f, 1.0f, 0.0f, 0.0f};
    Mat4 view = mat4LookAt(eye, target, up);

    // Verify it transforms eye to origin (approximately)
    Vec4 transformed_eye = mat4MulVec4(view, eye);
    bool lookAt_valid =
        std::fabs(transformed_eye.x) < 0.1f &&
        std::fabs(transformed_eye.y) < 0.1f &&
        std::fabs(transformed_eye.z) < 0.1f;
    if (lookAt_valid) {
        std::cout << "[PASS] mat4LookAt (eye transforms near origin)\n";
    } else {
        std::cout << "[FAIL] mat4LookAt\n";
        std::cout << "  Transformed eye: (" << transformed_eye.x << ", " << transformed_eye.y << ", " << transformed_eye.z << ", " << transformed_eye.w << ")\n";
        test_passed = false;
    }

    // Test 15: Inverse correctness (M * M^-1 = I)
    std::cout << "\nInverse Verification:\n";
    std::cout << "------------------------------\n";

    Mat4 test_matrix = mat4Mul(mat4Translation(1.0f, 2.0f, 3.0f), mat4Scale(2.0f, 2.0f, 2.0f));
    Mat4 test_inverse = mat4Inverse(test_matrix);
    Mat4 should_be_identity = mat4Mul(test_matrix, test_inverse);
    assert_mat4_equal("M * M^-1 = I", should_be_identity, identity);

    // Summary
    std::cout << "\n========================================\n";
    if (test_passed) {
        std::cout << "  ✓ ALL TESTS PASSED\n";
    } else {
        std::cout << "  ✗ SOME TESTS FAILED\n";
    }
    std::cout << "========================================\n\n";

    return test_passed ? 0 : 1;
}
