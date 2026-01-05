/**
 * @file benchmark_power_full.cpp
 * @brief Comprehensive performance benchmark for power.h functions
 *
 * Compares MMath implementations against:
 * - std:: (C++ standard library)
 * - GLM (if available)
 *
 * Test scenarios:
 * - Scalar operations (10M iterations)
 * - SIMD array operations (various sizes)
 * - Real-world audio use cases
 */

#include "include/fast_math/power.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <vector>
#include <random>

// ============================================================================
// Timing Utilities
// ============================================================================

class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }

    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

// ============================================================================
// Benchmark Configuration
// ============================================================================

constexpr int32_t kScalarIterations = 10000000;  // 10M iterations
constexpr int32_t kArraySizes[] = {64, 4096, 1000000};  // Small, medium, large

// Random number generator (fixed seed for reproducibility)
std::mt19937 g_rng(12345);
std::uniform_real_distribution<float> g_dist_pos(0.1f, 10.0f);  // Positive values for log
std::uniform_real_distribution<float> g_dist_any(-5.0f, 5.0f);   // Any values for exp
std::uniform_real_distribution<float> g_dist_db(-60.0f, 6.0f);   // dB range for audio

// ============================================================================
// Scalar Benchmarks
// ============================================================================

void benchmark_scalar_exp() {
    std::cout << "exp (scalar, " << kScalarIterations << " iterations):" << std::endl;

    // Generate random inputs
    std::vector<float> inputs(kScalarIterations);
    for (auto& v : inputs) v = g_dist_any(g_rng);

    // MMath::exp
    {
        Timer timer;
        volatile float sink = 0.0f;
        timer.start();
        for (int32_t i = 0; i < kScalarIterations; ++i) {
            sink = MMath::exp(inputs[i]);
        }
        double elapsed = timer.elapsed_ms();
        std::cout << "  MMath::exp:           " << std::fixed << std::setprecision(2)
                  << std::setw(8) << elapsed << " ms" << std::endl;
    }

    // std::exp
    {
        Timer timer;
        volatile float sink = 0.0f;
        timer.start();
        for (int32_t i = 0; i < kScalarIterations; ++i) {
            sink = std::exp(inputs[i]);
        }
        double elapsed = timer.elapsed_ms();
        std::cout << "  std::exp:             " << std::fixed << std::setprecision(2)
                  << std::setw(8) << elapsed << " ms" << std::endl;
    }

    std::cout << std::endl;
}

void benchmark_scalar_log() {
    std::cout << "log (scalar, " << kScalarIterations << " iterations):" << std::endl;

    // Generate random positive inputs
    std::vector<float> inputs(kScalarIterations);
    for (auto& v : inputs) v = g_dist_pos(g_rng);

    // MMath::log
    {
        Timer timer;
        volatile float sink = 0.0f;
        timer.start();
        for (int32_t i = 0; i < kScalarIterations; ++i) {
            sink = MMath::log(inputs[i]);
        }
        double elapsed = timer.elapsed_ms();
        std::cout << "  MMath::log:           " << std::fixed << std::setprecision(2)
                  << std::setw(8) << elapsed << " ms" << std::endl;
    }

    // std::log
    {
        Timer timer;
        volatile float sink = 0.0f;
        timer.start();
        for (int32_t i = 0; i < kScalarIterations; ++i) {
            sink = std::log(inputs[i]);
        }
        double elapsed = timer.elapsed_ms();
        std::cout << "  std::log:             " << std::fixed << std::setprecision(2)
                  << std::setw(8) << elapsed << " ms" << std::endl;
    }

    std::cout << std::endl;
}

void benchmark_scalar_pow() {
    std::cout << "pow (scalar, " << kScalarIterations << " iterations):" << std::endl;

    // Generate random inputs
    std::vector<float> base(kScalarIterations);
    std::vector<float> exp(kScalarIterations);
    for (int32_t i = 0; i < kScalarIterations; ++i) {
        base[i] = g_dist_pos(g_rng);
        exp[i] = g_dist_any(g_rng);
    }

    // MMath::pow
    {
        Timer timer;
        volatile float sink = 0.0f;
        timer.start();
        for (int32_t i = 0; i < kScalarIterations; ++i) {
            sink = MMath::pow(base[i], exp[i]);
        }
        double elapsed = timer.elapsed_ms();
        std::cout << "  MMath::pow:           " << std::fixed << std::setprecision(2)
                  << std::setw(8) << elapsed << " ms" << std::endl;
    }

    // std::pow
    {
        Timer timer;
        volatile float sink = 0.0f;
        timer.start();
        for (int32_t i = 0; i < kScalarIterations; ++i) {
            sink = std::pow(base[i], exp[i]);
        }
        double elapsed = timer.elapsed_ms();
        std::cout << "  std::pow:             " << std::fixed << std::setprecision(2)
                  << std::setw(8) << elapsed << " ms" << std::endl;
    }

    // MMath::powi (integer exponent optimization)
    {
        Timer timer;
        volatile float sink = 0.0f;
        timer.start();
        for (int32_t i = 0; i < kScalarIterations; ++i) {
            sink = MMath::powi(base[i], static_cast<int32_t>(exp[i]));
        }
        double elapsed = timer.elapsed_ms();
        std::cout << "  MMath::powi (int):    " << std::fixed << std::setprecision(2)
                  << std::setw(8) << elapsed << " ms" << std::endl;
    }

    std::cout << std::endl;
}

void benchmark_scalar_pow10_db() {
    std::cout << "pow10 (dB to linear, " << kScalarIterations << " iterations):" << std::endl;

    // Generate dB values (typical audio range: -60 to +6 dB)
    std::vector<float> db_values(kScalarIterations);
    for (auto& v : db_values) v = g_dist_db(g_rng) / 20.0f;  // Convert to exponent

    // MMath::pow10
    {
        Timer timer;
        volatile float sink = 0.0f;
        timer.start();
        for (int32_t i = 0; i < kScalarIterations; ++i) {
            sink = MMath::pow10(db_values[i]);
        }
        double elapsed = timer.elapsed_ms();
        std::cout << "  MMath::pow10:         " << std::fixed << std::setprecision(2)
                  << std::setw(8) << elapsed << " ms" << std::endl;
    }

    // std::pow(10, x)
    {
        Timer timer;
        volatile float sink = 0.0f;
        timer.start();
        for (int32_t i = 0; i < kScalarIterations; ++i) {
            sink = std::pow(10.0f, db_values[i]);
        }
        double elapsed = timer.elapsed_ms();
        std::cout << "  std::pow(10, x):      " << std::fixed << std::setprecision(2)
                  << std::setw(8) << elapsed << " ms" << std::endl;
    }

    std::cout << std::endl;
}

// ============================================================================
// SIMD Array Benchmarks
// ============================================================================

void benchmark_array_exp() {
    std::cout << "=== Array Operations: exp (SIMD) ===" << std::endl << std::endl;

    for (int32_t size : kArraySizes) {
        std::cout << "Size: " << size << " elements" << std::endl;

        // Generate random inputs
        std::vector<float> values(size);
        for (auto& v : values) v = g_dist_any(g_rng);

        // MMath::expArray (SIMD)
        {
            std::vector<float> data = values;
            Timer timer;
            timer.start();
            MMath::expArray(data.data(), size);
            double elapsed = timer.elapsed_ms();
            std::cout << "  MMath SIMD:           " << std::fixed << std::setprecision(3)
                      << elapsed << " ms" << std::endl;
        }

        // std:: scalar loop
        {
            std::vector<float> data = values;
            Timer timer;
            timer.start();
            for (int32_t i = 0; i < size; ++i) {
                data[i] = std::exp(data[i]);
            }
            double elapsed = timer.elapsed_ms();
            std::cout << "  std:: loop:           " << std::fixed << std::setprecision(3)
                      << elapsed << " ms" << std::endl;
        }

        std::cout << std::endl;
    }
}

void benchmark_array_log() {
    std::cout << "=== Array Operations: log (SIMD) ===" << std::endl << std::endl;

    for (int32_t size : kArraySizes) {
        std::cout << "Size: " << size << " elements" << std::endl;

        // Generate random positive inputs
        std::vector<float> values(size);
        for (auto& v : values) v = g_dist_pos(g_rng);

        // MMath::logArray (SIMD)
        {
            std::vector<float> data = values;
            Timer timer;
            timer.start();
            MMath::logArray(data.data(), size);
            double elapsed = timer.elapsed_ms();
            std::cout << "  MMath SIMD:           " << std::fixed << std::setprecision(3)
                      << elapsed << " ms" << std::endl;
        }

        // std:: scalar loop
        {
            std::vector<float> data = values;
            Timer timer;
            timer.start();
            for (int32_t i = 0; i < size; ++i) {
                data[i] = std::log(data[i]);
            }
            double elapsed = timer.elapsed_ms();
            std::cout << "  std:: loop:           " << std::fixed << std::setprecision(3)
                      << elapsed << " ms" << std::endl;
        }

        std::cout << std::endl;
    }
}

void benchmark_array_pow10() {
    std::cout << "=== Array Operations: pow10 (dB to linear) ===" << std::endl << std::endl;

    for (int32_t size : kArraySizes) {
        std::cout << "Size: " << size << " elements" << std::endl;

        // Generate dB values
        std::vector<float> values(size);
        for (auto& v : values) v = g_dist_db(g_rng) / 20.0f;

        // MMath::pow10Array (SIMD)
        {
            std::vector<float> data = values;
            Timer timer;
            timer.start();
            MMath::pow10Array(data.data(), size);
            double elapsed = timer.elapsed_ms();
            std::cout << "  MMath SIMD:           " << std::fixed << std::setprecision(3)
                      << elapsed << " ms" << std::endl;
        }

        // std::pow(10, x) scalar loop
        {
            std::vector<float> data = values;
            Timer timer;
            timer.start();
            for (int32_t i = 0; i < size; ++i) {
                data[i] = std::pow(10.0f, data[i]);
            }
            double elapsed = timer.elapsed_ms();
            std::cout << "  std::pow loop:        " << std::fixed << std::setprecision(3)
                      << elapsed << " ms" << std::endl;
        }

        std::cout << std::endl;
    }
}

void benchmark_array_pow() {
    std::cout << "=== Array Operations: pow (general) ===" << std::endl << std::endl;

    for (int32_t size : kArraySizes) {
        std::cout << "Size: " << size << " elements" << std::endl;

        // Generate random inputs
        std::vector<float> base(size);
        std::vector<float> exp(size);
        for (int32_t i = 0; i < size; ++i) {
            base[i] = g_dist_pos(g_rng);
            exp[i] = g_dist_any(g_rng);
        }

        // MMath::powArray (SIMD)
        {
            std::vector<float> result(size);
            Timer timer;
            timer.start();
            MMath::powArray(base.data(), exp.data(), result.data(), size);
            double elapsed = timer.elapsed_ms();
            std::cout << "  MMath SIMD:           " << std::fixed << std::setprecision(3)
                      << elapsed << " ms" << std::endl;
        }

        // std::pow scalar loop
        {
            std::vector<float> result(size);
            Timer timer;
            timer.start();
            for (int32_t i = 0; i < size; ++i) {
                result[i] = std::pow(base[i], exp[i]);
            }
            double elapsed = timer.elapsed_ms();
            std::cout << "  std::pow loop:        " << std::fixed << std::setprecision(3)
                      << elapsed << " ms" << std::endl;
        }

        std::cout << std::endl;
    }
}

// ============================================================================
// Real-World Audio Use Case
// ============================================================================

void benchmark_audio_db_conversion() {
    std::cout << "=== Real-World Use Case: Audio dB Conversion ===" << std::endl;
    std::cout << "Scenario: Convert 48000 dB values to linear (1s @ 48000Hz)" << std::endl;
    std::cout << "Results:" << std::endl;

    const int32_t num_samples = 48000;

    // Generate dB values (typical fade: -60 to 0 dB)
    std::vector<float> db_values(num_samples);
    for (int32_t i = 0; i < num_samples; ++i) {
        db_values[i] = -60.0f + (60.0f * i / num_samples);  // Linear fade
        db_values[i] /= 20.0f;  // Convert to exponent
    }

    double mmath_time, std_time;

    // MMath::pow10Array
    {
        std::vector<float> data = db_values;
        Timer timer;
        timer.start();
        MMath::pow10Array(data.data(), num_samples);
        mmath_time = timer.elapsed_ms();
        std::cout << "  MMath SIMD:     " << std::fixed << std::setprecision(3)
                  << mmath_time << " ms" << std::endl;
    }

    // std::pow loop
    {
        std::vector<float> data = db_values;
        Timer timer;
        timer.start();
        for (int32_t i = 0; i < num_samples; ++i) {
            data[i] = std::pow(10.0f, data[i]);
        }
        std_time = timer.elapsed_ms();
        std::cout << "  std:: loop:     " << std::fixed << std::setprecision(3)
                  << std_time << " ms" << std::endl;
    }

    double speedup = std_time / mmath_time;
    std::cout << "  Speedup:     " << std::fixed << std::setprecision(2)
              << speedup << "x" << std::endl;
    std::cout << "  Time saved:  " << std::fixed << std::setprecision(3)
              << (std_time - mmath_time) << " ms" << std::endl;

    std::cout << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "MMath vs std::" << std::endl;
    std::cout << "Power Function Performance Comparison" << std::endl;
    std::cout << "========================================" << std::endl;

    // Check SIMD support
    std::cout << "SIMD: ";
#if defined(__AVX2__)
    std::cout << "AVX2 enabled";
#elif defined(__SSE4_1__)
    std::cout << "SSE4.1 enabled";
#else
    std::cout << "Scalar only";
#endif
    std::cout << std::endl << std::endl;

    // Scalar benchmarks
    std::cout << "--- Scalar Operations ---" << std::endl;
    benchmark_scalar_exp();
    benchmark_scalar_log();
    benchmark_scalar_pow();
    benchmark_scalar_pow10_db();

    // SIMD array benchmarks
    std::cout << "--- SIMD Array Operations ---" << std::endl;
    benchmark_array_exp();
    benchmark_array_log();
    benchmark_array_pow10();
    benchmark_array_pow();

    // Real-world use case
    benchmark_audio_db_conversion();

    std::cout << "========================================" << std::endl;
    std::cout << "Benchmarks Complete" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
