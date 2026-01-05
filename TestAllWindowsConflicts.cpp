/**
 * @file TestAllWindowsConflicts.cpp
 * @brief Comprehensive test for all potential Windows macro conflicts
 *
 * This test simulates the scenario when graphics libraries (GLFW, DirectX, etc.)
 * include <windows.h> before MMath headers.
 */

#if defined(_WIN32)
    // Define WIN32_LEAN_AND_MEAN to reduce conflicts, but still includes near/far
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>

    // Also test with common graphics library patterns
    #ifdef near
        #pragma message("WARNING: 'near' macro is defined")
    #endif
    #ifdef far
        #pragma message("WARNING: 'far' macro is defined")
    #endif
#endif

// Include all MMath headers
#include "include/fast_math/types.h"
#include "include/fast_math/vec2.h"
#include "include/fast_math/mat3.h"
#include "include/fast_math/aabb2.h"
#include "include/fast_math/trig.h"

#include <iostream>
#include <cmath>

int main() {
    using namespace MMath;

    std::cout << "=== Testing MMath with Windows.h included ===" << std::endl;

    // Test 1: Vec2 operations
    {
        Vec2 v1{1.0f, 2.0f};
        Vec2 v2{3.0f, 4.0f};
        Vec2 result = vec2Add(v1, v2);
        std::cout << "1. vec2Add: (" << result.x << ", " << result.y << ") - ";
        std::cout << (result.x == 4.0f && result.y == 6.0f ? "PASS" : "FAIL") << std::endl;
    }

    // Test 2: Mat3 identity and multiplication
    {
        Mat3 m = mat3Identity();
        Mat3 result = mat3Multiply(m, m);
        std::cout << "2. mat3Multiply: m00=" << result.m00 << " - ";
        std::cout << (std::abs(result.m00 - 1.0f) < 0.0001f ? "PASS" : "FAIL") << std::endl;
    }

    // Test 3: Mat3 NearEqual (the function with the conflict)
    {
        Mat3 m1 = mat3Identity();
        Mat3 m2 = mat3Identity();
        m2.m00 = 1.00001f;  // Slightly different

        bool near_result = mat3NearEqual(m1, m2, 0.001f);
        std::cout << "3. mat3NearEqual: " << (near_result ? "PASS" : "FAIL") << std::endl;

        bool far_result = !mat3NearEqual(m1, m2, 0.000001f);
        std::cout << "4. mat3NearEqual (should fail): " << (far_result ? "PASS" : "FAIL") << std::endl;
    }

    // Test 4: AABB2 operations
    {
        Aabb2 a1 = aabb2FromMinMax({0, 0}, {10, 10});
        Aabb2 a2 = aabb2FromMinMax({5, 5}, {15, 15});

        bool intersects = aabb2Intersects(a1, a2);
        std::cout << "5. aabb2Intersects: " << (intersects ? "PASS" : "FAIL") << std::endl;

        Aabb2 merged = aabb2Merge(a1, a2);
        Vec2 min = aabb2Min(merged);
        Vec2 max = aabb2Max(merged);
        std::cout << "6. aabb2Merge: min=(" << min.x << "," << min.y << ") max=("
                  << max.x << "," << max.y << ") - ";
        std::cout << (min.x == 0.0f && max.x == 15.0f ? "PASS" : "FAIL") << std::endl;
    }

    // Test 5: AABB2 NearEqual (the function with the conflict)
    {
        Aabb2 a1 = aabb2FromMinMax({0, 0}, {1, 1});
        Aabb2 a2 = aabb2FromMinMax({0.00001f, 0.00001f}, {1.00001f, 1.00001f});

        bool near_result = aabb2NearEqual(a1, a2, 0.001f);
        std::cout << "7. aabb2NearEqual: " << (near_result ? "PASS" : "FAIL") << std::endl;
    }

    // Test 6: Trigonometry
    {
        float angle = 3.14159265f / 4.0f;  // 45 degrees
        float sin_val = MMath::sin(angle);
        float cos_val = MMath::cos(angle);

        float expected = std::sqrt(2.0f) / 2.0f;
        bool sin_ok = std::abs(sin_val - expected) < 0.001f;
        bool cos_ok = std::abs(cos_val - expected) < 0.001f;

        std::cout << "8. Trigonometry (sin/cos): " << ((sin_ok && cos_ok) ? "PASS" : "FAIL") << std::endl;
    }

    // Test 7: Transform composition with NearEqual
    {
        Mat3 t1 = mat3FromTranslation({10.0f, 20.0f});
        Mat3 t2 = mat3FromTranslation({-10.0f, -20.0f});
        Mat3 result = mat3Multiply(t1, t2);
        Mat3 identity = mat3Identity();

        bool is_identity = mat3NearEqual(result, identity, 0.001f);
        std::cout << "9. Transform round-trip with NearEqual: " << (is_identity ? "PASS" : "FAIL") << std::endl;
    }

    std::cout << "\n=== All tests completed ===" << std::endl;

#if defined(_WIN32)
    std::cout << "\nNote: Compiled with <windows.h> included (near/far macros present)" << std::endl;
#endif

    return 0;
}
