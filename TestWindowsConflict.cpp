/**
 * @file TestWindowsConflict.cpp
 * @brief Test to reproduce Windows macro conflict with 'near' and 'far'
 */

// Simulate what happens when a graphics library includes windows.h first
#if defined(_WIN32)
    #include <windows.h>  // This defines 'near' and 'far' macros
#endif

// Now include MMath headers
#include "include/fast_math/types.h"
#include "include/fast_math/vec2.h"
#include "include/fast_math/mat3.h"
#include "include/fast_math/aabb2.h"

#include <iostream>

int main() {
    using namespace MMath;

    // Test mat3NearEqual
    Mat3 m1 = mat3Identity();
    Mat3 m2 = mat3Identity();

    bool result = mat3NearEqual(m1, m2, 0.0001f);
    std::cout << "mat3NearEqual test: " << (result ? "PASS" : "FAIL") << std::endl;

    // Test aabb2NearEqual
    Aabb2 a1 = aabb2FromMinMax({0, 0}, {1, 1});
    Aabb2 a2 = aabb2FromMinMax({0, 0}, {1, 1});

    bool result2 = aabb2NearEqual(a1, a2, 0.0001f);
    std::cout << "aabb2NearEqual test: " << (result2 ? "PASS" : "FAIL") << std::endl;

    return 0;
}
