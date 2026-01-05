/**
 * @file benchmark_lerp_full.cpp
 * @brief Comprehensive performance benchmarks comparing MMath vs GLM vs DirectXMath vs std::
 *
 * Comparison Matrix:
 * - MMath (FMA-optimized when available)
 * - GLM (OpenGL Mathematics library)
 * - DirectXMath (Microsoft's SIMD math library)
 * - Manual scalar implementations
 *
 * Focus: Measuring FMA optimization impact (5-30% expected improvement)
 */

#include "include/fast_math/lerp.h"
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
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(min_val, max_val);

    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(rng);
    }
    return data;
}

// ============================================================================
// Benchmark: lerp Scalar
// ============================================================================

void benchmark_lerp_scalar() {
    constexpr int ITERATIONS = 10000000;
    auto data_a = generateRandomFloats(ITERATIONS, 0.0f, 100.0f);
    auto data_b = generateRandomFloats(ITERATIONS, 0.0f, 100.0f);
    float t = 0.3f;
    float result = 0.0f;

    Timer timer;

    std::cout << "lerp (scalar, " << ITERATIONS << " iterations, t=0.3):\n";

    // MMath (FMA-optimized when available)
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        result += MMath::lerp(data_a[i], data_b[i], t);
    }
    double mmath_time = timer.elapsed_ms();
    std::cout << "  MMath (FMA):      " << std::setw(8) << std::fixed << std::setprecision(2) << mmath_time << " ms\n";

    // Standard formula: a + t*(b-a)
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        result += data_a[i] + t * (data_b[i] - data_a[i]);
    }
    double std_time = timer.elapsed_ms();
    std::cout << "  std (a+t*(b-a)):  " << std::setw(8) << std_time << " ms  (speedup: " << std::setprecision(2) << (std_time / mmath_time) << "x)\n";

    // GLM mix
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        result += glm::mix(data_a[i], data_b[i], t);
    }
    double glm_time = timer.elapsed_ms();
    std::cout << "  glm::mix:         " << std::setw(8) << glm_time << " ms  (speedup: " << (glm_time / mmath_time) << "x)\n";

    // DirectXMath
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        XMVECTOR va = XMVectorReplicate(data_a[i]);
        XMVECTOR vb = XMVectorReplicate(data_b[i]);
        result += XMVectorGetX(XMVectorLerp(va, vb, t));
    }
    double dxm_time = timer.elapsed_ms();
    std::cout << "  DirectXMath:      " << std::setw(8) << dxm_time << " ms  (speedup: " << (dxm_time / mmath_time) << "x)\n";

    if (result == 1e30f) std::cout << "";
}

// ============================================================================
// Benchmark: smoothstep Scalar
// ============================================================================

void benchmark_smoothstep_scalar() {
    constexpr int ITERATIONS = 10000000;
    auto data = generateRandomFloats(ITERATIONS, -0.5f, 1.5f);
    float result = 0.0f;

    Timer timer;

    std::cout << "\nsmoothstep (scalar, " << ITERATIONS << " iterations):\n";

    // MMath
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        result += MMath::smoothstep(0.0f, 1.0f, data[i]);
    }
    double mmath_time = timer.elapsed_ms();
    std::cout << "  MMath:            " << std::setw(8) << std::fixed << std::setprecision(2) << mmath_time << " ms\n";

    // GLM
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        result += glm::smoothstep(0.0f, 1.0f, data[i]);
    }
    double glm_time = timer.elapsed_ms();
    std::cout << "  glm::smoothstep:  " << std::setw(8) << glm_time << " ms  (speedup: " << std::setprecision(2) << (glm_time / mmath_time) << "x)\n";

    // Manual implementation
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        float x = std::clamp((data[i] - 0.0f) / (1.0f - 0.0f), 0.0f, 1.0f);
        result += x * x * (3.0f - 2.0f * x);
    }
    double manual_time = timer.elapsed_ms();
    std::cout << "  Manual:           " << std::setw(8) << manual_time << " ms  (speedup: " << (manual_time / mmath_time) << "x)\n";

    if (result == 1e30f) std::cout << "";
}

// ============================================================================
// Benchmark: Array Operations (SIMD)
// ============================================================================

void benchmark_lerpArray_comparison() {
    std::cout << "\n=== Array Operations: lerp (SIMD) ===\n";

    std::vector<size_t> sizes = {64, 4096, 1000000};

    for (size_t size : sizes) {
        auto data_a = generateRandomFloats(size, 0.0f, 100.0f);
        auto data_b = generateRandomFloats(size, 0.0f, 100.0f);
        auto result_mmath = std::vector<float>(size);
        auto result_std = std::vector<float>(size);
        auto result_dxm = std::vector<float>(size);

        float t = 0.3f;

        Timer timer;

        std::cout << "\nSize: " << size << " elements\n";

        // MMath SIMD
        timer.start();
        lerpArray(data_a.data(), data_b.data(), t, result_mmath.data(), static_cast<int32_t>(size));
        double mmath_time = timer.elapsed_ms();
        std::cout << "  MMath SIMD:        " << std::setw(8) << std::fixed << std::setprecision(3) << mmath_time << " ms\n";

        // Scalar loop
        timer.start();
        for (size_t i = 0; i < size; ++i) {
            result_std[i] = data_a[i] + t * (data_b[i] - data_a[i]);
        }
        double std_time = timer.elapsed_ms();
        std::cout << "  std:: loop:        " << std::setw(8) << std_time << " ms  (speedup: " << std::setprecision(2) << (std_time / mmath_time) << "x)\n";

        // DirectXMath SIMD (4 floats per iteration)
        timer.start();
        size_t i = 0;
        XMVECTOR vt = XMVectorReplicate(t);
        for (; i + 4 <= size; i += 4) {
            XMVECTOR va = XMLoadFloat4(reinterpret_cast<const XMFLOAT4*>(&data_a[i]));
            XMVECTOR vb = XMLoadFloat4(reinterpret_cast<const XMFLOAT4*>(&data_b[i]));
            XMVECTOR vresult = XMVectorLerp(va, vb, t);
            XMStoreFloat4(reinterpret_cast<XMFLOAT4*>(&result_dxm[i]), vresult);
        }
        for (; i < size; ++i) {
            result_dxm[i] = data_a[i] + t * (data_b[i] - data_a[i]);
        }
        double dxm_time = timer.elapsed_ms();
        std::cout << "  DirectXMath SIMD:  " << std::setw(8) << dxm_time << " ms  (speedup: " << (dxm_time / mmath_time) << "x)\n";

        // Verify correctness
        bool correct = true;
        for (size_t j = 0; j < size; ++j) {
            if (std::abs(result_mmath[j] - result_std[j]) > 1e-4f) {
                correct = false;
                break;
            }
        }
        if (!correct) {
            std::cout << "  WARNING: Results differ!\n";
        }
    }
}

void benchmark_smoothstepArray_comparison() {
    std::cout << "\n=== Array Operations: smoothstep (SIMD) ===\n";

    std::vector<size_t> sizes = {64, 4096, 1000000};

    for (size_t size : sizes) {
        auto data_mmath = generateRandomFloats(size, -0.5f, 1.5f);
        auto data_std = data_mmath;

        Timer timer;

        std::cout << "\nSize: " << size << " elements\n";

        // MMath SIMD
        timer.start();
        smoothstepArray(0.0f, 1.0f, data_mmath.data(), static_cast<int32_t>(size));
        double mmath_time = timer.elapsed_ms();
        std::cout << "  MMath SIMD:        " << std::setw(8) << std::fixed << std::setprecision(3) << mmath_time << " ms\n";

        // Scalar loop
        timer.start();
        for (size_t i = 0; i < size; ++i) {
            float x = std::clamp((data_std[i] - 0.0f) / (1.0f - 0.0f), 0.0f, 1.0f);
            data_std[i] = x * x * (3.0f - 2.0f * x);
        }
        double std_time = timer.elapsed_ms();
        std::cout << "  std:: loop:        " << std::setw(8) << std_time << " ms  (speedup: " << std::setprecision(2) << (std_time / mmath_time) << "x)\n";

        // Verify correctness
        bool correct = true;
        for (size_t i = 0; i < size; ++i) {
            if (std::abs(data_mmath[i] - data_std[i]) > 1e-4f) {
                correct = false;
                break;
            }
        }
        if (!correct) {
            std::cout << "  WARNING: Results differ!\n";
        }
    }
}

// ============================================================================
// Real-World Use Case: Audio Crossfade
// ============================================================================

void benchmark_audio_crossfade() {
    constexpr size_t SAMPLE_RATE = 48000;
    constexpr size_t DURATION_SEC = 1;
    constexpr size_t TOTAL_SAMPLES = SAMPLE_RATE * DURATION_SEC;

    std::cout << "\n=== Real-World Use Case: Audio Crossfade ===\n";
    std::cout << "Scenario: Crossfade " << TOTAL_SAMPLES << " samples (1s @ " << SAMPLE_RATE << "Hz)\n";

    auto buffer_a = generateRandomFloats(TOTAL_SAMPLES, -1.0f, 1.0f);
    auto buffer_b = generateRandomFloats(TOTAL_SAMPLES, -1.0f, 1.0f);
    auto output_mmath = std::vector<float>(TOTAL_SAMPLES);
    auto output_std = std::vector<float>(TOTAL_SAMPLES);

    float fade = 0.5f;

    Timer timer;

    // MMath SIMD
    timer.start();
    lerpArray(buffer_a.data(), buffer_b.data(), fade, output_mmath.data(), static_cast<int32_t>(TOTAL_SAMPLES));
    double mmath_time = timer.elapsed_ms();

    // Scalar loop
    timer.start();
    for (size_t i = 0; i < TOTAL_SAMPLES; ++i) {
        output_std[i] = buffer_a[i] + fade * (buffer_b[i] - buffer_a[i]);
    }
    double std_time = timer.elapsed_ms();

    std::cout << "Results:\n";
    std::cout << "  MMath SIMD:  " << std::setw(8) << std::fixed << std::setprecision(3) << mmath_time << " ms\n";
    std::cout << "  std:: loop:  " << std::setw(8) << std_time << " ms\n";
    std::cout << "  Speedup:     " << std::setprecision(2) << (std_time / mmath_time) << "x\n";
    std::cout << "  Time saved:  " << std::setprecision(3) << (std_time - mmath_time) << " ms\n";
}

// ============================================================================
// Main Benchmark Runner
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "MMath vs GLM vs DirectXMath vs std::\n";
    std::cout << "Lerp Performance Comparison\n";
    std::cout << "========================================\n";

#if defined(__FMA__)
    std::cout << "FMA: Enabled\n";
#else
    std::cout << "FMA: Disabled\n";
#endif

#if defined(__AVX2__)
    std::cout << "SIMD: AVX2 enabled\n";
#elif defined(__SSE4_1__)
    std::cout << "SIMD: SSE4.1 enabled\n";
#else
    std::cout << "SIMD: Disabled\n";
#endif

    std::cout << "\n--- Scalar Operations ---\n";
    benchmark_lerp_scalar();
    benchmark_smoothstep_scalar();

    std::cout << "\n--- SIMD Array Operations ---";
    benchmark_lerpArray_comparison();
    benchmark_smoothstepArray_comparison();

    benchmark_audio_crossfade();

    std::cout << "\n========================================\n";
    std::cout << "Benchmarks Complete\n";
    std::cout << "========================================\n";

    return 0;
}
