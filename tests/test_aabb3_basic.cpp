/**
 * @file test_aabb3_basic.cpp
 * @brief Basic correctness tests for Aabb3 operations
 */

#include "fast_math/aabb3.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace MMath;
using namespace std;

// Helper function for approximate equality
bool nearEqual(float a, float b, float epsilon = 0.0001f) {
    float diff = a - b;
    return (diff < 0 ? -diff : diff) <= epsilon;
}

int main() {
    int passed = 0;
    int total = 0;

    cout << "\n================================================================\n";
    cout << "  Aabb3 Basic Correctness Tests\n";
    cout << "================================================================\n\n";

    // Test 1: aabb3FromMinMax
    {
        total++;
        Vec3 min = {-1.0f, -2.0f, -3.0f};
        Vec3 max = {4.0f, 5.0f, 6.0f};
        Aabb3 aabb = aabb3FromMinMax(min, max);

        Vec3 result_min = aabb3Min(aabb);
        Vec3 result_max = aabb3Max(aabb);

        if (vec3NearEqual(result_min, min, 0.0001f) && vec3NearEqual(result_max, max, 0.0001f)) {
            cout << "[PASS] Test 1: aabb3FromMinMax + aabb3Min/Max\n";
            passed++;
        } else {
            cout << "[FAIL] Test 1: aabb3FromMinMax + aabb3Min/Max\n";
            cout << "  Expected min: (" << min.x << ", " << min.y << ", " << min.z << ")\n";
            cout << "  Got min: (" << result_min.x << ", " << result_min.y << ", " << result_min.z << ")\n";
            cout << "  Expected max: (" << max.x << ", " << max.y << ", " << max.z << ")\n";
            cout << "  Got max: (" << result_max.x << ", " << result_max.y << ", " << result_max.z << ")\n";
        }
    }

    // Test 2: aabb3Union
    {
        total++;
        Aabb3 a = aabb3FromMinMax({0.0f, 0.0f, 0.0f}, {2.0f, 2.0f, 2.0f});
        Aabb3 b = aabb3FromMinMax({1.0f, 1.0f, 1.0f}, {3.0f, 3.0f, 3.0f});
        Aabb3 result = aabb3Union(a, b);

        Vec3 expected_min = {0.0f, 0.0f, 0.0f};
        Vec3 expected_max = {3.0f, 3.0f, 3.0f};
        Vec3 result_min = aabb3Min(result);
        Vec3 result_max = aabb3Max(result);

        if (vec3NearEqual(result_min, expected_min, 0.0001f) && vec3NearEqual(result_max, expected_max, 0.0001f)) {
            cout << "[PASS] Test 2: aabb3Union\n";
            passed++;
        } else {
            cout << "[FAIL] Test 2: aabb3Union\n";
            cout << "  Expected: [0,0,0] to [3,3,3]\n";
            cout << "  Got: [" << result_min.x << "," << result_min.y << "," << result_min.z << "] to ["
                 << result_max.x << "," << result_max.y << "," << result_max.z << "]\n";
        }
    }

    // Test 3: aabb3Overlap - overlapping boxes
    {
        total++;
        Aabb3 a = aabb3FromMinMax({0.0f, 0.0f, 0.0f}, {2.0f, 2.0f, 2.0f});
        Aabb3 b = aabb3FromMinMax({1.0f, 1.0f, 1.0f}, {3.0f, 3.0f, 3.0f});
        bool overlaps = aabb3Overlap(a, b);

        if (overlaps) {
            cout << "[PASS] Test 3: aabb3Overlap (overlapping)\n";
            passed++;
        } else {
            cout << "[FAIL] Test 3: aabb3Overlap (overlapping) - returned false\n";
        }
    }

    // Test 4: aabb3Overlap - non-overlapping boxes
    {
        total++;
        Aabb3 a = aabb3FromMinMax({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});
        Aabb3 b = aabb3FromMinMax({2.0f, 2.0f, 2.0f}, {3.0f, 3.0f, 3.0f});
        bool overlaps = aabb3Overlap(a, b);

        if (!overlaps) {
            cout << "[PASS] Test 4: aabb3Overlap (non-overlapping)\n";
            passed++;
        } else {
            cout << "[FAIL] Test 4: aabb3Overlap (non-overlapping) - returned true\n";
        }
    }

    // Test 5: aabb3Contains - point inside
    {
        total++;
        Aabb3 aabb = aabb3FromMinMax({0.0f, 0.0f, 0.0f}, {2.0f, 2.0f, 2.0f});
        Vec3 point = {1.0f, 1.0f, 1.0f};
        bool contains = aabb3Contains(aabb, point);

        if (contains) {
            cout << "[PASS] Test 5: aabb3Contains (point inside)\n";
            passed++;
        } else {
            cout << "[FAIL] Test 5: aabb3Contains (point inside) - returned false\n";
        }
    }

    // Test 6: aabb3Contains - point outside
    {
        total++;
        Aabb3 aabb = aabb3FromMinMax({0.0f, 0.0f, 0.0f}, {2.0f, 2.0f, 2.0f});
        Vec3 point = {3.0f, 3.0f, 3.0f};
        bool contains = aabb3Contains(aabb, point);

        if (!contains) {
            cout << "[PASS] Test 6: aabb3Contains (point outside)\n";
            passed++;
        } else {
            cout << "[FAIL] Test 6: aabb3Contains (point outside) - returned true\n";
        }
    }

    // Test 7: aabb3Contains - point on boundary
    {
        total++;
        Aabb3 aabb = aabb3FromMinMax({0.0f, 0.0f, 0.0f}, {2.0f, 2.0f, 2.0f});
        Vec3 point = {2.0f, 2.0f, 2.0f};
        bool contains = aabb3Contains(aabb, point);

        if (contains) {
            cout << "[PASS] Test 7: aabb3Contains (point on boundary)\n";
            passed++;
        } else {
            cout << "[FAIL] Test 7: aabb3Contains (point on boundary) - returned false\n";
        }
    }

    // Test 8: aabb3Transform - identity matrix
    {
        total++;
        Aabb3 aabb = aabb3FromMinMax({-1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, 1.0f});
        Mat4 identity = mat4Identity();
        Aabb3 result = aabb3Transform(aabb, identity);

        Vec3 result_min = aabb3Min(result);
        Vec3 result_max = aabb3Max(result);
        Vec3 expected_min = {-1.0f, -1.0f, -1.0f};
        Vec3 expected_max = {1.0f, 1.0f, 1.0f};

        if (vec3NearEqual(result_min, expected_min, 0.0001f) && vec3NearEqual(result_max, expected_max, 0.0001f)) {
            cout << "[PASS] Test 8: aabb3Transform (identity matrix)\n";
            passed++;
        } else {
            cout << "[FAIL] Test 8: aabb3Transform (identity matrix)\n";
            cout << "  Expected: [-1,-1,-1] to [1,1,1]\n";
            cout << "  Got: [" << result_min.x << "," << result_min.y << "," << result_min.z << "] to ["
                 << result_max.x << "," << result_max.y << "," << result_max.z << "]\n";
        }
    }

    // Test 9: aabb3Transform - translation
    {
        total++;
        Aabb3 aabb = aabb3FromMinMax({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});
        Mat4 translation = mat4Translation(5.0f, 10.0f, 15.0f);
        Aabb3 result = aabb3Transform(aabb, translation);

        Vec3 result_min = aabb3Min(result);
        Vec3 result_max = aabb3Max(result);
        Vec3 expected_min = {5.0f, 10.0f, 15.0f};
        Vec3 expected_max = {6.0f, 11.0f, 16.0f};

        if (vec3NearEqual(result_min, expected_min, 0.01f) && vec3NearEqual(result_max, expected_max, 0.01f)) {
            cout << "[PASS] Test 9: aabb3Transform (translation)\n";
            passed++;
        } else {
            cout << "[FAIL] Test 9: aabb3Transform (translation)\n";
            cout << "  Expected: [5,10,15] to [6,11,16]\n";
            cout << "  Got: [" << result_min.x << "," << result_min.y << "," << result_min.z << "] to ["
                 << result_max.x << "," << result_max.y << "," << result_max.z << "]\n";
        }
    }

    // Test 10: aabb3Transform - scale
    {
        total++;
        Aabb3 aabb = aabb3FromMinMax({-1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, 1.0f});
        Mat4 scale = mat4Scale(2.0f, 3.0f, 4.0f);
        Aabb3 result = aabb3Transform(aabb, scale);

        Vec3 result_min = aabb3Min(result);
        Vec3 result_max = aabb3Max(result);
        Vec3 expected_min = {-2.0f, -3.0f, -4.0f};
        Vec3 expected_max = {2.0f, 3.0f, 4.0f};

        if (vec3NearEqual(result_min, expected_min, 0.01f) && vec3NearEqual(result_max, expected_max, 0.01f)) {
            cout << "[PASS] Test 10: aabb3Transform (scale)\n";
            passed++;
        } else {
            cout << "[FAIL] Test 10: aabb3Transform (scale)\n";
            cout << "  Expected: [-2,-3,-4] to [2,3,4]\n";
            cout << "  Got: [" << result_min.x << "," << result_min.y << "," << result_min.z << "] to ["
                 << result_max.x << "," << result_max.y << "," << result_max.z << "]\n";
        }
    }

    // Test 11: aabb3Center and aabb3Extents
    {
        total++;
        Aabb3 aabb = aabb3FromMinMax({-2.0f, -4.0f, -6.0f}, {2.0f, 4.0f, 6.0f});
        Vec3 center = aabb3Center(aabb);
        Vec3 extents = aabb3Extents(aabb);

        Vec3 expected_center = {0.0f, 0.0f, 0.0f};
        Vec3 expected_extents = {2.0f, 4.0f, 6.0f};

        if (vec3NearEqual(center, expected_center, 0.0001f) && vec3NearEqual(extents, expected_extents, 0.0001f)) {
            cout << "[PASS] Test 11: aabb3Center and aabb3Extents\n";
            passed++;
        } else {
            cout << "[FAIL] Test 11: aabb3Center and aabb3Extents\n";
            cout << "  Expected center: (0, 0, 0), extents: (2, 4, 6)\n";
            cout << "  Got center: (" << center.x << ", " << center.y << ", " << center.z << ")\n";
            cout << "  Got extents: (" << extents.x << ", " << extents.y << ", " << extents.z << ")\n";
        }
    }

    // Test 12: aabb3Volume and aabb3SurfaceArea
    {
        total++;
        Aabb3 aabb = aabb3FromMinMax({0.0f, 0.0f, 0.0f}, {2.0f, 3.0f, 4.0f});
        float volume = aabb3Volume(aabb);
        float surfaceArea = aabb3SurfaceArea(aabb);

        float expected_volume = 2.0f * 3.0f * 4.0f;  // 24
        float expected_surface = 2.0f * (2.0f*3.0f + 3.0f*4.0f + 4.0f*2.0f);  // 2*(6+12+8) = 52

        if (nearEqual(volume, expected_volume) && nearEqual(surfaceArea, expected_surface)) {
            cout << "[PASS] Test 12: aabb3Volume and aabb3SurfaceArea\n";
            passed++;
        } else {
            cout << "[FAIL] Test 12: aabb3Volume and aabb3SurfaceArea\n";
            cout << "  Expected volume: " << expected_volume << ", surface: " << expected_surface << "\n";
            cout << "  Got volume: " << volume << ", surface: " << surfaceArea << "\n";
        }
    }

    // Summary
    cout << "\n================================================================\n";
    cout << "  Test Results: " << passed << "/" << total << " passed\n";
    if (passed == total) {
        cout << "  Status: ALL TESTS PASSED ✓\n";
    } else {
        cout << "  Status: SOME TESTS FAILED ✗\n";
    }
    cout << "================================================================\n\n";

    return (passed == total) ? 0 : 1;
}
