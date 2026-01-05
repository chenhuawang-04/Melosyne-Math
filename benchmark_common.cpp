/**
 * @file benchmark_common.cpp
 * @brief Performance benchmarks for common.h functions
 *
 * Comparison Matrix:
 * - MMath (our implementation)
 * - std:: (C++ standard library)
 * - GLM (if available)
 * - DirectXMath (if available on Windows)
 *
 * Test Scenarios:
 * 1. Scalar operations (single value, tight loop)
 * 2. Small arrays (64 elements - cache friendly)
 * 3. Medium arrays (4096 elements - realistic)
 * 4. Large arrays (1M elements - memory bandwidth bound)
 */

#include "include/fast_math/common.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <algorithm>
#include <cmath>

// Optional: GLM support
#ifdef USE_GLM
#include <glm/glm.hpp>
#include <glm/common.hpp>
#endif

// Optional: DirectXMath support
#ifdef USE_DIRECTXMATH
#include <DirectXMath.h>
#endif

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
// Benchmark: min/max Scalar
// ============================================================================

void benchmark_min_scalar() {
    constexpr int ITERATIONS = 10000000;
    auto data_a = generateRandomFloats(ITERATIONS, -100.0f, 100.0f);
    auto data_b = generateRandomFloats(ITERATIONS, -100.0f, 100.0f);
    float result = 0.0f;

    Timer timer;

    // MMath
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        result += min(data_a[i], data_b[i]);
    }
    double mmath_time = timer.elapsed_ms();

    // std::
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        result += std::min(data_a[i], data_b[i]);
    }
    double std_time = timer.elapsed_ms();

    std::cout << "min (scalar, " << ITERATIONS << " iterations):" << std::endl;
    std::cout << "  MMath:     " << std::setw(8) << std::fixed << std::setprecision(2) << mmath_time << " ms" << std::endl;
    std::cout << "  std::min:  " << std::setw(8) << std_time << " ms";
    std::cout << "  (speedup: " << std::setprecision(2) << (std_time / mmath_time) << "x)" << std::endl;

    // Prevent optimization
    if (result == 1e30f) std::cout << "";
}

void benchmark_max_scalar() {
    constexpr int ITERATIONS = 10000000;
    auto data_a = generateRandomFloats(ITERATIONS, -100.0f, 100.0f);
    auto data_b = generateRandomFloats(ITERATIONS, -100.0f, 100.0f);
    float result = 0.0f;

    Timer timer;

    // MMath
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        result += max(data_a[i], data_b[i]);
    }
    double mmath_time = timer.elapsed_ms();

    // std::
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        result += std::max(data_a[i], data_b[i]);
    }
    double std_time = timer.elapsed_ms();

    std::cout << "max (scalar, " << ITERATIONS << " iterations):" << std::endl;
    std::cout << "  MMath:     " << std::setw(8) << std::fixed << std::setprecision(2) << mmath_time << " ms" << std::endl;
    std::cout << "  std::max:  " << std::setw(8) << std_time << " ms";
    std::cout << "  (speedup: " << std::setprecision(2) << (std_time / mmath_time) << "x)" << std::endl;

    if (result == 1e30f) std::cout << "";
}

void benchmark_clamp_scalar() {
    constexpr int ITERATIONS = 10000000;
    auto data = generateRandomFloats(ITERATIONS, -2.0f, 2.0f);
    float result = 0.0f;

    Timer timer;

    // MMath
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        result += clamp(data[i], 0.0f, 1.0f);
    }
    double mmath_time = timer.elapsed_ms();

    // std::
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        result += std::clamp(data[i], 0.0f, 1.0f);
    }
    double std_time = timer.elapsed_ms();

    std::cout << "clamp (scalar, " << ITERATIONS << " iterations):" << std::endl;
    std::cout << "  MMath:       " << std::setw(8) << std::fixed << std::setprecision(2) << mmath_time << " ms" << std::endl;
    std::cout << "  std::clamp:  " << std::setw(8) << std_time << " ms";
    std::cout << "  (speedup: " << std::setprecision(2) << (std_time / mmath_time) << "x)" << std::endl;

    if (result == 1e30f) std::cout << "";
}

void benchmark_abs_scalar() {
    constexpr int ITERATIONS = 10000000;
    auto data = generateRandomFloats(ITERATIONS, -100.0f, 100.0f);
    float result = 0.0f;

    Timer timer;

    // MMath
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        result += MMath::abs(data[i]);
    }
    double mmath_time = timer.elapsed_ms();

    // std::
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        result += std::abs(data[i]);
    }
    double std_time = timer.elapsed_ms();

    std::cout << "abs (scalar, " << ITERATIONS << " iterations):" << std::endl;
    std::cout << "  MMath:     " << std::setw(8) << std::fixed << std::setprecision(2) << mmath_time << " ms" << std::endl;
    std::cout << "  std::abs:  " << std::setw(8) << std_time << " ms";
    std::cout << "  (speedup: " << std::setprecision(2) << (std_time / mmath_time) << "x)" << std::endl;

    if (result == 1e30f) std::cout << "";
}

// ============================================================================
// Benchmark: Array Operations (SIMD)
// ============================================================================

void benchmark_clampArray() {
    std::vector<size_t> sizes = {64, 4096, 1000000};

    for (size_t size : sizes) {
        auto data_mmath = generateRandomFloats(size, -2.0f, 2.0f);
        auto data_std = data_mmath;  // Copy for std:: version

        Timer timer;

        // MMath (SIMD)
        timer.start();
        clampArray(data_mmath.data(), 0.0f, 1.0f, static_cast<int32_t>(size));
        double mmath_time = timer.elapsed_ms();

        // std:: (scalar loop)
        timer.start();
        for (size_t i = 0; i < size; ++i) {
            data_std[i] = std::clamp(data_std[i], 0.0f, 1.0f);
        }
        double std_time = timer.elapsed_ms();

        std::cout << "clampArray (size=" << size << "):" << std::endl;
        std::cout << "  MMath SIMD: " << std::setw(8) << std::fixed << std::setprecision(2) << mmath_time << " ms" << std::endl;
        std::cout << "  std:: loop: " << std::setw(8) << std_time << " ms";
        std::cout << "  (speedup: " << std::setprecision(2) << (std_time / mmath_time) << "x)" << std::endl;

        // Verify correctness
        bool correct = true;
        for (size_t i = 0; i < size; ++i) {
            if (std::abs(data_mmath[i] - data_std[i]) > 1e-5f) {
                correct = false;
                break;
            }
        }
        if (!correct) {
            std::cout << "  WARNING: Results differ!" << std::endl;
        }
    }
}

void benchmark_absArray() {
    std::vector<size_t> sizes = {64, 4096, 1000000};

    for (size_t size : sizes) {
        auto data_mmath = generateRandomFloats(size, -100.0f, 100.0f);
        auto data_std = data_mmath;

        Timer timer;

        // MMath (SIMD)
        timer.start();
        absArray(data_mmath.data(), static_cast<int32_t>(size));
        double mmath_time = timer.elapsed_ms();

        // std:: (scalar loop)
        timer.start();
        for (size_t i = 0; i < size; ++i) {
            data_std[i] = std::abs(data_std[i]);
        }
        double std_time = timer.elapsed_ms();

        std::cout << "absArray (size=" << size << "):" << std::endl;
        std::cout << "  MMath SIMD: " << std::setw(8) << std::fixed << std::setprecision(2) << mmath_time << " ms" << std::endl;
        std::cout << "  std:: loop: " << std::setw(8) << std_time << " ms";
        std::cout << "  (speedup: " << std::setprecision(2) << (std_time / mmath_time) << "x)" << std::endl;

        // Verify correctness
        bool correct = true;
        for (size_t i = 0; i < size; ++i) {
            if (std::abs(data_mmath[i] - data_std[i]) > 1e-5f) {
                correct = false;
                break;
            }
        }
        if (!correct) {
            std::cout << "  WARNING: Results differ!" << std::endl;
        }
    }
}

void benchmark_minArray() {
    std::vector<size_t> sizes = {64, 4096, 1000000};

    for (size_t size : sizes) {
        auto data_a_mmath = generateRandomFloats(size, -100.0f, 100.0f);
        auto data_b = generateRandomFloats(size, -100.0f, 100.0f);
        auto data_a_std = data_a_mmath;

        Timer timer;

        // MMath (SIMD)
        timer.start();
        minArray(data_a_mmath.data(), data_b.data(), static_cast<int32_t>(size));
        double mmath_time = timer.elapsed_ms();

        // std:: (scalar loop)
        timer.start();
        for (size_t i = 0; i < size; ++i) {
            data_a_std[i] = std::min(data_a_std[i], data_b[i]);
        }
        double std_time = timer.elapsed_ms();

        std::cout << "minArray (size=" << size << "):" << std::endl;
        std::cout << "  MMath SIMD: " << std::setw(8) << std::fixed << std::setprecision(2) << mmath_time << " ms" << std::endl;
        std::cout << "  std:: loop: " << std::setw(8) << std_time << " ms";
        std::cout << "  (speedup: " << std::setprecision(2) << (std_time / mmath_time) << "x)" << std::endl;
    }
}

// ============================================================================
// Real-World Use Case: Audio Volume Limiting
// ============================================================================

void benchmark_audio_volume_limiting() {
    constexpr size_t SAMPLE_RATE = 48000;  // 48kHz
    constexpr size_t DURATION_SEC = 10;     // 10 seconds
    constexpr size_t TOTAL_SAMPLES = SAMPLE_RATE * DURATION_SEC;

    std::cout << "\n=== Real-World Use Case: Audio Volume Limiting ===" << std::endl;
    std::cout << "Scenario: Limit " << TOTAL_SAMPLES << " samples (" << DURATION_SEC << "s @ " << SAMPLE_RATE << "Hz) to [0, 1]" << std::endl;

    auto audio_mmath = generateRandomFloats(TOTAL_SAMPLES, -1.5f, 1.5f);
    auto audio_std = audio_mmath;

    Timer timer;

    // MMath (SIMD)
    timer.start();
    clampArray(audio_mmath.data(), 0.0f, 1.0f, static_cast<int32_t>(TOTAL_SAMPLES));
    double mmath_time = timer.elapsed_ms();

    // std:: (scalar loop)
    timer.start();
    for (size_t i = 0; i < TOTAL_SAMPLES; ++i) {
        audio_std[i] = std::clamp(audio_std[i], 0.0f, 1.0f);
    }
    double std_time = timer.elapsed_ms();

    std::cout << "Results:" << std::endl;
    std::cout << "  MMath SIMD:  " << std::setw(8) << std::fixed << std::setprecision(2) << mmath_time << " ms" << std::endl;
    std::cout << "  std:: loop:  " << std::setw(8) << std_time << " ms" << std::endl;
    std::cout << "  Speedup:     " << std::setprecision(2) << (std_time / mmath_time) << "x" << std::endl;
    std::cout << "  Time saved:  " << std::setprecision(2) << (std_time - mmath_time) << " ms" << std::endl;
}

// ============================================================================
// Main Benchmark Runner
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "MMath common.h Performance Benchmarks" << std::endl;
    std::cout << "========================================" << std::endl;

#if defined(__AVX2__)
    std::cout << "SIMD: AVX2 enabled" << std::endl;
#elif defined(__SSE4_1__)
    std::cout << "SIMD: SSE4.1 enabled" << std::endl;
#else
    std::cout << "SIMD: Disabled (scalar fallback)" << std::endl;
#endif

    std::cout << "\n--- Scalar Operations ---" << std::endl;
    benchmark_min_scalar();
    std::cout << std::endl;
    benchmark_max_scalar();
    std::cout << std::endl;
    benchmark_clamp_scalar();
    std::cout << std::endl;
    benchmark_abs_scalar();

    std::cout << "\n--- Array Operations (SIMD) ---" << std::endl;
    benchmark_minArray();
    std::cout << std::endl;
    benchmark_clampArray();
    std::cout << std::endl;
    benchmark_absArray();

    benchmark_audio_volume_limiting();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmarks Complete" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
