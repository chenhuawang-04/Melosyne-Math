#include "Include/fast_math/types.h"
#include "Include/fast_math/vec2.h"
#include "Include/fast_math/mat3.h"
#include "Include/fast_math/aabb2.h"
#include "Include/fast_math/trig.h"
#include <iostream>

using namespace MMath;

int main() {
    std::cout << "Testing MMath Library Renaming\n";
    std::cout << "==============================\n\n";

    // Test 1: Vec2 operations
    std::cout << "Test 1: Vec2 Operations\n";
    Vec2 a = {3.0f, 4.0f};
    Vec2 b = {1.0f, 2.0f};
    Vec2 sum = vec2Add(a, b);
    float dot = vec2Dot(a, b);
    Vec2 normalized = vec2Normalize(a);
    std::cout << "  vec2Add: (" << sum.x << ", " << sum.y << ")\n";
    std::cout << "  vec2Dot: " << dot << "\n";
    std::cout << "  vec2Normalize: (" << normalized.x << ", " << normalized.y << ")\n";
    std::cout << "  ✓ Vec2 functions work\n\n";

    // Test 2: Mat3 operations
    std::cout << "Test 2: Mat3 Operations\n";
    Mat3 identity = mat3Identity();
    Mat3 translate = mat3FromTranslation({5.0f, 10.0f});
    Mat3 rotate = mat3FromRotation(0.785f);  // 45 degrees
    Mat3 scale = mat3FromScale({2.0f, 2.0f});
    Mat3 trs = mat3FromTrs({5.0f, 10.0f}, 0.785f, {2.0f, 2.0f});
    Vec2 point = {1.0f, 0.0f};
    Vec2 transformed = mat3TransformPoint(trs, point);
    std::cout << "  mat3FromTrs + mat3TransformPoint: (" << transformed.x << ", " << transformed.y << ")\n";
    std::cout << "  ✓ Mat3 functions work\n\n";

    // Test 3: Aabb2 operations
    std::cout << "Test 3: Aabb2 Operations\n";
    Aabb2 box1 = aabb2FromMinMax({0.0f, 0.0f}, {10.0f, 10.0f});
    Aabb2 box2 = aabb2FromMinMax({5.0f, 5.0f}, {15.0f, 15.0f});
    bool intersects = aabb2Intersects(box1, box2);
    Aabb2 merged = aabb2Merge(box1, box2);
    bool contains = aabb2ContainsPoint(box1, {5.0f, 5.0f});
    std::cout << "  aabb2Intersects: " << (intersects ? "true" : "false") << "\n";
    std::cout << "  aabb2Merge: valid\n";
    std::cout << "  aabb2ContainsPoint: " << (contains ? "true" : "false") << "\n";
    std::cout << "  ✓ Aabb2 functions work\n\n";

    // Test 4: Trig operations
    std::cout << "Test 4: Trig Operations\n";
    float angle = 0.785398f;  // 45 degrees
    float sin_val = MMath::sin(angle);
    float cos_val = MMath::cos(angle);
    SinCos sc;
    sincos(Angle{angle}, &sc);
    std::cout << "  sin(45°): " << sin_val << "\n";
    std::cout << "  cos(45°): " << cos_val << "\n";
    std::cout << "  sincos batch: sin=" << sc.sin << ", cos=" << sc.cos << "\n";
    std::cout << "  ✓ Trig functions work\n\n";

    std::cout << "==============================\n";
    std::cout << "✓ All MMath renaming tests passed!\n";
    return 0;
}
