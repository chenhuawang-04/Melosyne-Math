/**
 * @file PerformanceBenchmark.cpp
 * @brief Performance benchmark comparing MMath vs GLM
 *
 * Test scenarios:
 * - Vector operations (add, dot, normalize)
 * - Matrix operations (multiply, transform)
 * - AABB operations (intersect, merge)
 * - Trigonometry (sin/cos)
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>

// MMath headers
#include "fast_math/types.h"
#include "fast_math/vec2.h"
#include "fast_math/mat3.h"
#include "fast_math/aabb2.h"
#include "fast_math/trig.h"

// GLM headers
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

using namespace MMath;

// ============================================================================
// Timing Utilities
// ============================================================================

class Timer {
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_time).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;
};

// ============================================================================
// Test Data Generation
// ============================================================================

std::vector<Vec2> generateMathVec2(int count) {
    std::vector<Vec2> result;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    for (int i = 0; i < count; ++i) {
        result.push_back({dist(rng), dist(rng)});
    }
    return result;
}

std::vector<glm::vec2> generateGlmVec2(int count) {
    std::vector<glm::vec2> result;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    for (int i = 0; i < count; ++i) {
        result.push_back(glm::vec2(dist(rng), dist(rng)));
    }
    return result;
}

std::vector<Mat3> generateMathMat3(int count) {
    std::vector<Mat3> result;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> angle_dist(0.0f, 6.283f);
    std::uniform_real_distribution<float> scale_dist(0.5f, 2.0f);
    std::uniform_real_distribution<float> trans_dist(-100.0f, 100.0f);

    for (int i = 0; i < count; ++i) {
        result.push_back(mat3FromTrs(
            {trans_dist(rng), trans_dist(rng)},
            angle_dist(rng),
            {scale_dist(rng), scale_dist(rng)}
        ));
    }
    return result;
}

std::vector<glm::mat3> generateGlmMat3(int count) {
    std::vector<glm::mat3> result;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> angle_dist(0.0f, 6.283f);
    std::uniform_real_distribution<float> scale_dist(0.5f, 2.0f);
    std::uniform_real_distribution<float> trans_dist(-100.0f, 100.0f);

    for (int i = 0; i < count; ++i) {
        // Build manually for 3x3
        float c = std::cos(angle_dist(rng));
        float s = std::sin(angle_dist(rng));
        float sx = scale_dist(rng);
        float sy = scale_dist(rng);
        float tx = trans_dist(rng);
        float ty = trans_dist(rng);

        glm::mat3 mat(
            sx * c, sx * s, 0.0f,
            -sy * s, sy * c, 0.0f,
            tx, ty, 1.0f
        );
        result.push_back(mat);
    }
    return result;
}

std::vector<Aabb2> generateMathAabb2(int count) {
    std::vector<Aabb2> result;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    for (int i = 0; i < count; ++i) {
        float x = dist(rng);
        float y = dist(rng);
        result.push_back(aabb2FromMinMax({x, y}, {x + 10.0f, y + 10.0f}));
    }
    return result;
}

// ============================================================================
// Benchmark Tests
// ============================================================================

void benchmarkVec2Add() {
    std::cout << "\n=== Vector Addition (1M operations) ===\n";

    const int COUNT = 1000000;
    auto v1_math = generateMathVec2(COUNT);
    auto v2_math = generateMathVec2(COUNT);
    auto v1_glm = generateGlmVec2(COUNT);
    auto v2_glm = generateGlmVec2(COUNT);

    std::vector<Vec2> result_math(COUNT);
    std::vector<glm::vec2> result_glm(COUNT);

    // MMath
    Timer timer;
    timer.start();
    for (int i = 0; i < COUNT; ++i) {
        result_math[i] = vec2Add(v1_math[i], v2_math[i]);
    }
    double mmath_time = timer.elapsed_ms();

    // GLM
    timer.start();
    for (int i = 0; i < COUNT; ++i) {
        result_glm[i] = v1_glm[i] + v2_glm[i];
    }
    double glm_time = timer.elapsed_ms();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "MMath:  " << mmath_time << " ms\n";
    std::cout << "GLM:    " << glm_time << " ms\n";
    std::cout << "Ratio:  " << (glm_time / mmath_time) << "x\n";
}

void benchmarkVec2Dot() {
    std::cout << "\n=== Dot Product (1M operations) ===\n";

    const int COUNT = 1000000;
    auto v1_math = generateMathVec2(COUNT);
    auto v2_math = generateMathVec2(COUNT);
    auto v1_glm = generateGlmVec2(COUNT);
    auto v2_glm = generateGlmVec2(COUNT);

    std::vector<float> result_math(COUNT);
    std::vector<float> result_glm(COUNT);

    // MMath
    Timer timer;
    timer.start();
    for (int i = 0; i < COUNT; ++i) {
        result_math[i] = vec2Dot(v1_math[i], v2_math[i]);
    }
    double mmath_time = timer.elapsed_ms();

    // GLM
    timer.start();
    for (int i = 0; i < COUNT; ++i) {
        result_glm[i] = glm::dot(v1_glm[i], v2_glm[i]);
    }
    double glm_time = timer.elapsed_ms();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "MMath:  " << mmath_time << " ms\n";
    std::cout << "GLM:    " << glm_time << " ms\n";
    std::cout << "Ratio:  " << (glm_time / mmath_time) << "x\n";
}

void benchmarkVec2Normalize() {
    std::cout << "\n=== Vector Normalize (1M operations) ===\n";

    const int COUNT = 1000000;
    auto v_math = generateMathVec2(COUNT);
    auto v_glm = generateGlmVec2(COUNT);

    std::vector<Vec2> result_math(COUNT);
    std::vector<glm::vec2> result_glm(COUNT);

    // MMath
    Timer timer;
    timer.start();
    for (int i = 0; i < COUNT; ++i) {
        result_math[i] = vec2Normalize(v_math[i]);
    }
    double mmath_time = timer.elapsed_ms();

    // GLM
    timer.start();
    for (int i = 0; i < COUNT; ++i) {
        result_glm[i] = glm::normalize(v_glm[i]);
    }
    double glm_time = timer.elapsed_ms();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "MMath:  " << mmath_time << " ms\n";
    std::cout << "GLM:    " << glm_time << " ms\n";
    std::cout << "Ratio:  " << (glm_time / mmath_time) << "x\n";
}

void benchmarkMat3Multiply() {
    std::cout << "\n=== Matrix Multiply (100K operations) ===\n";

    const int COUNT = 100000;
    auto m1_math = generateMathMat3(COUNT);
    auto m2_math = generateMathMat3(COUNT);
    auto m1_glm = generateGlmMat3(COUNT);
    auto m2_glm = generateGlmMat3(COUNT);

    std::vector<Mat3> result_math(COUNT);
    std::vector<glm::mat3> result_glm(COUNT);

    // MMath
    Timer timer;
    timer.start();
    for (int i = 0; i < COUNT; ++i) {
        result_math[i] = mat3MultiplyAffine(m1_math[i], m2_math[i]);
    }
    double mmath_time = timer.elapsed_ms();

    // GLM
    timer.start();
    for (int i = 0; i < COUNT; ++i) {
        result_glm[i] = m1_glm[i] * m2_glm[i];
    }
    double glm_time = timer.elapsed_ms();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "MMath:  " << mmath_time << " ms\n";
    std::cout << "GLM:    " << glm_time << " ms\n";
    std::cout << "Ratio:  " << (glm_time / mmath_time) << "x\n";
}

void benchmarkMat3TransformPoint() {
    std::cout << "\n=== Matrix Transform Point (1M operations) ===\n";

    const int COUNT = 1000000;
    auto matrices_math = generateMathMat3(COUNT);
    auto points_math = generateMathVec2(COUNT);
    auto matrices_glm = generateGlmMat3(COUNT);
    auto points_glm = generateGlmVec2(COUNT);

    std::vector<Vec2> result_math(COUNT);
    std::vector<glm::vec2> result_glm(COUNT);

    // MMath
    Timer timer;
    timer.start();
    for (int i = 0; i < COUNT; ++i) {
        result_math[i] = mat3TransformPoint(matrices_math[i], points_math[i]);
    }
    double mmath_time = timer.elapsed_ms();

    // GLM - Transform point (multiply by 3x3 matrix with homogeneous coords)
    timer.start();
    for (int i = 0; i < COUNT; ++i) {
        glm::vec3 p3 = matrices_glm[i] * glm::vec3(points_glm[i], 1.0f);
        result_glm[i] = glm::vec2(p3.x, p3.y);
    }
    double glm_time = timer.elapsed_ms();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "MMath:  " << mmath_time << " ms\n";
    std::cout << "GLM:    " << glm_time << " ms\n";
    std::cout << "Ratio:  " << (glm_time / mmath_time) << "x\n";
}

void benchmarkAabb2Intersects() {
    std::cout << "\n=== AABB Intersection (10M operations) ===\n";

    const int COUNT = 10000000;
    auto boxes1_math = generateMathAabb2(COUNT / 100);
    auto boxes2_math = generateMathAabb2(COUNT / 100);

    std::vector<bool> result_math(COUNT / 100);

    int actual_count = boxes1_math.size();

    // MMath
    Timer timer;
    timer.start();
    for (int i = 0; i < actual_count; ++i) {
        result_math[i] = aabb2Intersects(boxes1_math[i], boxes2_math[i]);
    }
    double mmath_time = timer.elapsed_ms();

    // Scale to millions for comparison
    mmath_time *= 100;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "MMath:  " << mmath_time << " ms (10M ops)\n";
    std::cout << "Note:   GLM has no built-in AABB (AABB2), only box intersection via SAT\n";
}

void benchmarkAabb2Merge() {
    std::cout << "\n=== AABB Merge (10M operations) ===\n";

    const int COUNT = 10000000;
    auto boxes1_math = generateMathAabb2(COUNT / 100);
    auto boxes2_math = generateMathAabb2(COUNT / 100);

    std::vector<Aabb2> result_math(COUNT / 100);

    int actual_count = boxes1_math.size();

    // MMath
    Timer timer;
    timer.start();
    for (int i = 0; i < actual_count; ++i) {
        result_math[i] = aabb2Merge(boxes1_math[i], boxes2_math[i]);
    }
    double mmath_time = timer.elapsed_ms();

    // Scale to millions for comparison
    mmath_time *= 100;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "MMath:  " << mmath_time << " ms (10M ops)\n";
    std::cout << "Note:   GLM has no built-in AABB merge operation\n";
}

void benchmarkTrigonometry() {
    std::cout << "\n=== Trigonometry - sin/cos (1M operations) ===\n";

    const int COUNT = 1000000;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 6.283f);

    std::vector<float> angles(COUNT);
    for (int i = 0; i < COUNT; ++i) {
        angles[i] = dist(rng);
    }

    std::vector<float> result_sin(COUNT), result_cos(COUNT);

    // MMath
    Timer timer;
    timer.start();
    for (int i = 0; i < COUNT; ++i) {
        result_sin[i] = MMath::sin(angles[i]);
        result_cos[i] = MMath::cos(angles[i]);
    }
    double mmath_time = timer.elapsed_ms();

    // GLM
    timer.start();
    for (int i = 0; i < COUNT; ++i) {
        result_sin[i] = glm::sin(angles[i]);
        result_cos[i] = glm::cos(angles[i]);
    }
    double glm_time = timer.elapsed_ms();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "MMath:  " << mmath_time << " ms\n";
    std::cout << "GLM:    " << glm_time << " ms\n";
    std::cout << "Ratio:  " << (glm_time / mmath_time) << "x\n";
}

// ============================================================================
// Summary Report
// ============================================================================

void printHeader() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         MMath vs GLM Performance Benchmark                 ║\n";
    std::cout << "║         Higher ratio = MMath is faster                     ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";
}

void printFooter() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Summary:                                                  ║\n";
    std::cout << "║  - MMath: Hand-optimized 2D math library with SIMD         ║\n";
    std::cout << "║  - GLM:   Generic linear algebra library                  ║\n";
    std::cout << "║  - Benchmark flags: -O3 -march=native -ffast-math         ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";
}

int main() {
    printHeader();

    try {
        // Vector operations
        benchmarkVec2Add();
        benchmarkVec2Dot();
        benchmarkVec2Normalize();

        // Matrix operations
        benchmarkMat3Multiply();
        benchmarkMat3TransformPoint();

        // AABB operations
        benchmarkAabb2Intersects();
        benchmarkAabb2Merge();

        // Trigonometry
        benchmarkTrigonometry();

        printFooter();

        std::cout << "✓ Benchmark completed successfully!\n";
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
