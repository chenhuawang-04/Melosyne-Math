/**
 * @file benchmark_sqrt.cpp
 * @brief Performance benchmarks for sqrt.h functions
 *
 * Comparison Matrix:
 * - MMath (our implementation)
 * - std:: (C++ standard library)
 * - Theoretical (1/sqrt, 1/x)
 *
 * Test Scenarios:
 * 1. Scalar operations (single value, tight loop)
 * 2. Small arrays (64 elements - cache friendly)
 * 3. Medium arrays (4096 elements - realistic)
 * 4. Large arrays (1M elements - memory bandwidth bound)
 */

#include "include/fast_math/sqrt.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>

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

    // MMath
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        result += MMath::sqrt(data[i]);
    }
    double mmath_time = timer.elapsed_ms();

    // std::
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        result += std::sqrt(data[i]);
    }
    double std_time = timer.elapsed_ms();

    std::cout << "sqrt (scalar, " << ITERATIONS << " iterations):" << std::endl;
    std::cout << "  MMath:      " << std::setw(8) << std::fixed << std::setprecision(2) << mmath_time << " ms" << std::endl;
    std::cout << "  std::sqrt:  " << std::setw(8) << std_time << " ms";
    std::cout << "  (speedup: " << std::setprecision(2) << (std_time / mmath_time) << "x)" << std::endl;

    // Prevent optimization
    if (result == 1e30f) std::cout << "";
}

void benchmark_inverseSqrt_scalar() {
    constexpr int ITERATIONS = 10000000;
    auto data = generateRandomFloats(ITERATIONS, 0.001f, 10000.0f);
    float result = 0.0f;

    Timer timer;

    // MMath (rsqrt + NR)
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        result += MMath::inverseSqrt(data[i]);
    }
    double mmath_time = timer.elapsed_ms();

    // std:: (1/sqrt)
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        result += 1.0f / std::sqrt(data[i]);
    }
    double std_time = timer.elapsed_ms();

    std::cout << "inverseSqrt (scalar, " << ITERATIONS << " iterations):" << std::endl;
    std::cout << "  MMath:         " << std::setw(8) << std::fixed << std::setprecision(2) << mmath_time << " ms" << std::endl;
    std::cout << "  1/std::sqrt:   " << std::setw(8) << std_time << " ms";
    std::cout << "  (speedup: " << std::setprecision(2) << (std_time / mmath_time) << "x)" << std::endl;

    if (result == 1e30f) std::cout << "";
}

void benchmark_reciprocal_scalar() {
    constexpr int ITERATIONS = 10000000;
    auto data = generateRandomFloats(ITERATIONS, 0.001f, 10000.0f);
    float result = 0.0f;

    Timer timer;

    // MMath (rcp + NR)
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        result += MMath::reciprocal(data[i]);
    }
    double mmath_time = timer.elapsed_ms();

    // std:: (1/x)
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        result += 1.0f / data[i];
    }
    double std_time = timer.elapsed_ms();

    std::cout << "reciprocal (scalar, " << ITERATIONS << " iterations):" << std::endl;
    std::cout << "  MMath:      " << std::setw(8) << std::fixed << std::setprecision(2) << mmath_time << " ms" << std::endl;
    std::cout << "  1/x:        " << std::setw(8) << std_time << " ms";
    std::cout << "  (speedup: " << std::setprecision(2) << (std_time / mmath_time) << "x)" << std::endl;

    if (result == 1e30f) std::cout << "";
}

// ============================================================================
// Benchmark: Array Operations (SIMD)
// ============================================================================

void benchmark_sqrtArray() {
    std::vector<size_t> sizes = {64, 4096, 1000000};

    for (size_t size : sizes) {
        auto data_mmath = generateRandomFloats(size, 0.001f, 10000.0f);
        auto data_std = data_mmath;  // Copy for std:: version

        Timer timer;

        // MMath (SIMD)
        timer.start();
        sqrtArray(data_mmath.data(), static_cast<int32_t>(size));
        double mmath_time = timer.elapsed_ms();

        // std:: (scalar loop)
        timer.start();
        for (size_t i = 0; i < size; ++i) {
            data_std[i] = std::sqrt(data_std[i]);
        }
        double std_time = timer.elapsed_ms();

        std::cout << "sqrtArray (size=" << size << "):" << std::endl;
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

void benchmark_inverseSqrtArray() {
    std::vector<size_t> sizes = {64, 4096, 1000000};

    for (size_t size : sizes) {
        auto data_mmath = generateRandomFloats(size, 0.001f, 10000.0f);
        auto data_std = data_mmath;

        Timer timer;

        // MMath (SIMD)
        timer.start();
        inverseSqrtArray(data_mmath.data(), static_cast<int32_t>(size));
        double mmath_time = timer.elapsed_ms();

        // std:: (scalar loop)
        timer.start();
        for (size_t i = 0; i < size; ++i) {
            data_std[i] = 1.0f / std::sqrt(data_std[i]);
        }
        double std_time = timer.elapsed_ms();

        std::cout << "inverseSqrtArray (size=" << size << "):" << std::endl;
        std::cout << "  MMath SIMD: " << std::setw(8) << std::fixed << std::setprecision(2) << mmath_time << " ms" << std::endl;
        std::cout << "  std:: loop: " << std::setw(8) << std_time << " ms";
        std::cout << "  (speedup: " << std::setprecision(2) << (std_time / mmath_time) << "x)" << std::endl;

        // Verify correctness (allow small error due to NR approximation)
        bool correct = true;
        for (size_t i = 0; i < size; ++i) {
            float rel_error = std::abs((data_mmath[i] - data_std[i]) / data_std[i]);
            if (rel_error > 0.0001f) {  // 0.01% threshold
                correct = false;
                break;
            }
        }
        if (!correct) {
            std::cout << "  WARNING: Precision issue detected!" << std::endl;
        }
    }
}

void benchmark_reciprocalArray() {
    std::vector<size_t> sizes = {64, 4096, 1000000};

    for (size_t size : sizes) {
        auto data_mmath = generateRandomFloats(size, 0.001f, 10000.0f);
        auto data_std = data_mmath;

        Timer timer;

        // MMath (SIMD)
        timer.start();
        reciprocalArray(data_mmath.data(), static_cast<int32_t>(size));
        double mmath_time = timer.elapsed_ms();

        // std:: (scalar loop)
        timer.start();
        for (size_t i = 0; i < size; ++i) {
            data_std[i] = 1.0f / data_std[i];
        }
        double std_time = timer.elapsed_ms();

        std::cout << "reciprocalArray (size=" << size << "):" << std::endl;
        std::cout << "  MMath SIMD: " << std::setw(8) << std::fixed << std::setprecision(2) << mmath_time << " ms" << std::endl;
        std::cout << "  std:: loop: " << std::setw(8) << std_time << " ms";
        std::cout << "  (speedup: " << std::setprecision(2) << (std_time / mmath_time) << "x)" << std::endl;

        // Verify correctness
        bool correct = true;
        for (size_t i = 0; i < size; ++i) {
            float rel_error = std::abs((data_mmath[i] - data_std[i]) / data_std[i]);
            if (rel_error > 0.0001f) {
                correct = false;
                break;
            }
        }
        if (!correct) {
            std::cout << "  WARNING: Precision issue detected!" << std::endl;
        }
    }
}

// ============================================================================
// Real-World Use Case: Vector Normalization
// ============================================================================

void benchmark_vector_normalization() {
    constexpr size_t NUM_VECTORS = 100000;  // 100k 2D vectors

    std::cout << "\n=== Real-World Use Case: Vector Normalization ===\n";
    std::cout << "Scenario: Normalize " << NUM_VECTORS << " 2D vectors (e.g., velocity directions)\n";

    auto length_squared_mmath = generateRandomFloats(NUM_VECTORS, 0.1f, 100.0f);
    auto length_squared_std = length_squared_mmath;

    Timer timer;

    // MMath (inverseSqrt)
    timer.start();
    inverseSqrtArray(length_squared_mmath.data(), static_cast<int32_t>(NUM_VECTORS));
    double mmath_time = timer.elapsed_ms();

    // std:: (1/sqrt loop)
    timer.start();
    for (size_t i = 0; i < NUM_VECTORS; ++i) {
        length_squared_std[i] = 1.0f / std::sqrt(length_squared_std[i]);
    }
    double std_time = timer.elapsed_ms();

    std::cout << "Results:\n";
    std::cout << "  MMath inverseSqrt:  " << std::setw(8) << std::fixed << std::setprecision(2) << mmath_time << " ms\n";
    std::cout << "  std:: 1/sqrt:       " << std::setw(8) << std_time << " ms\n";
    std::cout << "  Speedup:            " << std::setprecision(2) << (std_time / mmath_time) << "x\n";
    std::cout << "  Time saved:         " << std::setprecision(2) << (std_time - mmath_time) << " ms\n";
}

// ============================================================================
// Main Benchmark Runner
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "MMath sqrt.h Performance Benchmarks" << std::endl;
    std::cout << "========================================" << std::endl;

#if defined(__AVX2__)
    std::cout << "SIMD: AVX2 enabled" << std::endl;
#elif defined(__SSE4_1__)
    std::cout << "SIMD: SSE4.1 enabled" << std::endl;
#else
    std::cout << "SIMD: Disabled (scalar fallback)" << std::endl;
#endif

    std::cout << "\n--- Scalar Operations ---" << std::endl;
    benchmark_sqrt_scalar();
    std::cout << std::endl;
    benchmark_inverseSqrt_scalar();
    std::cout << std::endl;
    benchmark_reciprocal_scalar();

    std::cout << "\n--- Array Operations (SIMD) ---" << std::endl;
    benchmark_sqrtArray();
    std::cout << std::endl;
    benchmark_inverseSqrtArray();
    std::cout << std::endl;
    benchmark_reciprocalArray();

    benchmark_vector_normalization();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmarks Complete" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
