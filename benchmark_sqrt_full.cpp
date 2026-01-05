/**
 * @file benchmark_sqrt_full.cpp
 * @brief Comprehensive performance benchmarks comparing MMath vs GLM vs DirectXMath vs std::
 *
 * Comparison Matrix:
 * - MMath (our implementation with SSE/AVX)
 * - GLM (OpenGL Mathematics library)
 * - DirectXMath (Microsoft's SIMD math library)
 * - std:: (C++ standard library)
 *
 * Test Scenarios:
 * 1. Scalar operations (tight loop, 10M iterations)
 * 2. Array operations (SIMD batch processing)
 */

#include "include/fast_math/sqrt.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>

// GLM
#define GLM_FORCE_INLINE
#define GLM_FORCE_INTRINSICS
#include "../glm-1.0.2/glm/glm.hpp"
#include "../glm-1.0.2/glm/gtc/constants.hpp"

// DirectXMath
#include "../DirectXMath-apr2025/Inc/DirectXMath.h"
using namespace DirectX;

using namespace MMath;

// ============================================================================
// Timer Class
// ============================================================================

class Timer {
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double elapsed_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff = end_time - start_time;
        return diff.count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;
};

// ============================================================================
// Test Data Generation
// ============================================================================

std::vector<float> generateRandomFloats(size_t count, float min_val, float max_val) {
    std::vector<float> data(count);
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(min_val, max_val);

    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(rng);
    }
    return data;
}

// ============================================================================
// Benchmark: sqrt Scalar
// ============================================================================

void benchmark_sqrt_scalar() {
    constexpr int ITERATIONS = 10000000;
    auto data = generateRandomFloats(ITERATIONS, 0.001f, 10000.0f);
    float result = 0.0f;

    Timer timer;

    std::cout << "sqrt (scalar, " << ITERATIONS << " iterations):\n";

    // MMath
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        result += MMath::sqrt(data[i]);
    }
    double mmath_time = timer.elapsed_ms();
    std::cout << "  MMath:          " << std::setw(8) << std::fixed << std::setprecision(2) << mmath_time << " ms\n";

    // std::
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        result += std::sqrt(data[i]);
    }
    double std_time = timer.elapsed_ms();
    std::cout << "  std::sqrt:      " << std::setw(8) << std_time << " ms  (speedup: " << std::setprecision(2) << (std_time / mmath_time) << "x)\n";

    // GLM
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        result += glm::sqrt(data[i]);
    }
    double glm_time = timer.elapsed_ms();
    std::cout << "  glm::sqrt:      " << std::setw(8) << glm_time << " ms  (speedup: " << (glm_time / mmath_time) << "x)\n";

    // DirectXMath (using scalar conversion)
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        XMVECTOR v = XMVectorReplicate(data[i]);
        result += XMVectorGetX(XMVectorSqrt(v));
    }
    double dxm_time = timer.elapsed_ms();
    std::cout << "  DirectXMath:    " << std::setw(8) << dxm_time << " ms  (speedup: " << (dxm_time / mmath_time) << "x)\n";

    // Prevent optimization
    if (result == 1e30f) std::cout << "";
}

void benchmark_inverseSqrt_scalar() {
    constexpr int ITERATIONS = 10000000;
    auto data = generateRandomFloats(ITERATIONS, 0.001f, 10000.0f);
    float result = 0.0f;

    Timer timer;

    std::cout << "\ninverseSqrt (scalar, " << ITERATIONS << " iterations):\n";

    // MMath (rsqrt + NR)
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        result += MMath::inverseSqrt(data[i]);
    }
    double mmath_time = timer.elapsed_ms();
    std::cout << "  MMath:             " << std::setw(8) << std::fixed << std::setprecision(2) << mmath_time << " ms\n";

    // std:: (1/sqrt)
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        result += 1.0f / std::sqrt(data[i]);
    }
    double std_time = timer.elapsed_ms();
    std::cout << "  1/std::sqrt:       " << std::setw(8) << std_time << " ms  (speedup: " << (std_time / mmath_time) << "x)\n";

    // GLM (1/sqrt, no dedicated inversesqrt for scalar)
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        result += 1.0f / glm::sqrt(data[i]);
    }
    double glm_time = timer.elapsed_ms();
    std::cout << "  1/glm::sqrt:       " << std::setw(8) << glm_time << " ms  (speedup: " << (glm_time / mmath_time) << "x)\n";

    // DirectXMath (rsqrt)
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        XMVECTOR v = XMVectorReplicate(data[i]);
        result += XMVectorGetX(XMVectorReciprocalSqrt(v));
    }
    double dxm_time = timer.elapsed_ms();
    std::cout << "  DirectXMath:       " << std::setw(8) << dxm_time << " ms  (speedup: " << (dxm_time / mmath_time) << "x)\n";

    if (result == 1e30f) std::cout << "";
}

void benchmark_reciprocal_scalar() {
    constexpr int ITERATIONS = 10000000;
    auto data = generateRandomFloats(ITERATIONS, 0.001f, 10000.0f);
    float result = 0.0f;

    Timer timer;

    std::cout << "\nreciprocal (scalar, " << ITERATIONS << " iterations):\n";

    // MMath (rcp + NR)
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        result += MMath::reciprocal(data[i]);
    }
    double mmath_time = timer.elapsed_ms();
    std::cout << "  MMath:          " << std::setw(8) << std::fixed << std::setprecision(2) << mmath_time << " ms\n";

    // std:: (1/x)
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        result += 1.0f / data[i];
    }
    double std_time = timer.elapsed_ms();
    std::cout << "  1/x:            " << std::setw(8) << std_time << " ms  (speedup: " << (std_time / mmath_time) << "x)\n";

    // GLM (1/x)
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        result += 1.0f / data[i];  // GLM doesn't have dedicated reciprocal
    }
    double glm_time = timer.elapsed_ms();
    std::cout << "  GLM (1/x):      " << std::setw(8) << glm_time << " ms  (speedup: " << (glm_time / mmath_time) << "x)\n";

    // DirectXMath (rcp)
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        XMVECTOR v = XMVectorReplicate(data[i]);
        result += XMVectorGetX(XMVectorReciprocal(v));
    }
    double dxm_time = timer.elapsed_ms();
    std::cout << "  DirectXMath:    " << std::setw(8) << dxm_time << " ms  (speedup: " << (dxm_time / mmath_time) << "x)\n";

    if (result == 1e30f) std::cout << "";
}

// ============================================================================
// Benchmark: Array Operations (SIMD)
// ============================================================================

void benchmark_inverseSqrtArray_comparison() {
    std::cout << "\n=== Array Operations: inverseSqrt (SIMD) ===\n";

    std::vector<size_t> sizes = {64, 4096, 1000000};

    for (size_t size : sizes) {
        auto data_mmath = generateRandomFloats(size, 0.001f, 10000.0f);
        auto data_std = data_mmath;
        auto data_dxm = data_mmath;

        Timer timer;

        std::cout << "\nSize: " << size << " elements\n";

        // MMath (SIMD)
        timer.start();
        inverseSqrtArray(data_mmath.data(), static_cast<int32_t>(size));
        double mmath_time = timer.elapsed_ms();
        std::cout << "  MMath SIMD:        " << std::setw(8) << std::fixed << std::setprecision(3) << mmath_time << " ms\n";

        // std:: (scalar loop)
        timer.start();
        for (size_t i = 0; i < size; ++i) {
            data_std[i] = 1.0f / std::sqrt(data_std[i]);
        }
        double std_time = timer.elapsed_ms();
        std::cout << "  std:: loop:        " << std::setw(8) << std_time << " ms  (speedup: " << std::setprecision(2) << (std_time / mmath_time) << "x)\n";

        // DirectXMath (SIMD, 4 floats per iteration)
        timer.start();
        size_t i = 0;
        for (; i + 4 <= size; i += 4) {
            XMVECTOR v = XMLoadFloat4(reinterpret_cast<const XMFLOAT4*>(&data_dxm[i]));
            v = XMVectorReciprocalSqrt(v);
            XMStoreFloat4(reinterpret_cast<XMFLOAT4*>(&data_dxm[i]), v);
        }
        for (; i < size; ++i) {
            data_dxm[i] = 1.0f / std::sqrt(data_dxm[i]);
        }
        double dxm_time = timer.elapsed_ms();
        std::cout << "  DirectXMath SIMD:  " << std::setw(8) << dxm_time << " ms  (speedup: " << (dxm_time / mmath_time) << "x)\n";

        // Verify correctness
        int error_count = 0;
        for (size_t j = 0; j < size; ++j) {
            float rel_error = std::abs((data_mmath[j] - data_std[j]) / data_std[j]);
            if (rel_error > 0.001f) {  // 0.1% threshold
                error_count++;
            }
        }
        if (error_count > 0) {
            std::cout << "  WARNING: " << error_count << " values exceed 0.1% error\n";
        }
    }
}

void benchmark_sqrtArray_comparison() {
    std::cout << "\n=== Array Operations: sqrt (SIMD) ===\n";

    std::vector<size_t> sizes = {64, 4096, 1000000};

    for (size_t size : sizes) {
        auto data_mmath = generateRandomFloats(size, 0.001f, 10000.0f);
        auto data_std = data_mmath;
        auto data_dxm = data_mmath;

        Timer timer;

        std::cout << "\nSize: " << size << " elements\n";

        // MMath (SIMD)
        timer.start();
        sqrtArray(data_mmath.data(), static_cast<int32_t>(size));
        double mmath_time = timer.elapsed_ms();
        std::cout << "  MMath SIMD:        " << std::setw(8) << std::fixed << std::setprecision(3) << mmath_time << " ms\n";

        // std:: (scalar loop)
        timer.start();
        for (size_t i = 0; i < size; ++i) {
            data_std[i] = std::sqrt(data_std[i]);
        }
        double std_time = timer.elapsed_ms();
        std::cout << "  std:: loop:        " << std::setw(8) << std_time << " ms  (speedup: " << std::setprecision(2) << (std_time / mmath_time) << "x)\n";

        // DirectXMath (SIMD, 4 floats per iteration)
        timer.start();
        size_t i = 0;
        for (; i + 4 <= size; i += 4) {
            XMVECTOR v = XMLoadFloat4(reinterpret_cast<const XMFLOAT4*>(&data_dxm[i]));
            v = XMVectorSqrt(v);
            XMStoreFloat4(reinterpret_cast<XMFLOAT4*>(&data_dxm[i]), v);
        }
        for (; i < size; ++i) {
            data_dxm[i] = std::sqrt(data_dxm[i]);
        }
        double dxm_time = timer.elapsed_ms();
        std::cout << "  DirectXMath SIMD:  " << std::setw(8) << dxm_time << " ms  (speedup: " << (dxm_time / mmath_time) << "x)\n";
    }
}

// ============================================================================
// Main Benchmark Runner
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "MMath vs GLM vs DirectXMath vs std::\n";
    std::cout << "Performance Comparison\n";
    std::cout << "========================================\n";

#if defined(__AVX2__)
    std::cout << "SIMD: AVX2 enabled\n";
#elif defined(__SSE4_1__)
    std::cout << "SIMD: SSE4.1 enabled\n";
#else
    std::cout << "SIMD: Disabled (scalar fallback)\n";
#endif

    std::cout << "\n--- Scalar Operations ---\n";
    benchmark_sqrt_scalar();
    benchmark_inverseSqrt_scalar();
    benchmark_reciprocal_scalar();

    std::cout << "\n--- SIMD Array Operations ---";
    benchmark_sqrtArray_comparison();
    benchmark_inverseSqrtArray_comparison();

    std::cout << "\n========================================\n";
    std::cout << "Benchmarks Complete\n";
    std::cout << "========================================\n";

    return 0;
}
