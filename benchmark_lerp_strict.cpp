/**
 * @file benchmark_lerp_strict.cpp
 * @brief Strict performance comparison: MMath vs GLM vs std::
 */

#include "include/fast_math/lerp.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <algorithm>

// GLM
#define GLM_FORCE_INLINE
#define GLM_FORCE_INTRINSICS
#include "../glm-1.0.2/glm/glm.hpp"
#include "../glm-1.0.2/glm/gtc/constants.hpp"

using namespace MMath;

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

std::vector<float> generateRandomFloats(size_t count, float min_val, float max_val) {
    std::vector<float> data(count);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(rng);
    }
    return data;
}

void benchmark_lerp_scalar_strict() {
    const int ITERATIONS = 10000000;
    const int WARMUP = 100000;
    const int RUNS = 5;

    auto data_a = generateRandomFloats(ITERATIONS, 0.0f, 100.0f);
    auto data_b = generateRandomFloats(ITERATIONS, 0.0f, 100.0f);
    float t = 0.3f;

    std::cout << "lerp (scalar, " << ITERATIONS << " iterations, " << RUNS << " runs):\n";

    // MMath
    {
        std::vector<double> times;
        for (int run = 0; run < RUNS; ++run) {
            float result = 0.0f;

            // Warmup
            for (int i = 0; i < WARMUP; ++i) {
                result += lerp(data_a[i], data_b[i], t);
            }

            Timer timer;
            timer.start();
            for (int i = 0; i < ITERATIONS; ++i) {
                result += lerp(data_a[i], data_b[i], t);
            }
            double time = timer.elapsed_ms();
            times.push_back(time);

            if (result == 1e30f) std::cout << "";
        }

        std::sort(times.begin(), times.end());
        double median = times[RUNS / 2];
        std::cout << "  MMath:      " << std::fixed << std::setprecision(2)
                  << median << " ms (median)\n";
    }

    // std::
    {
        std::vector<double> times;
        for (int run = 0; run < RUNS; ++run) {
            float result = 0.0f;

            // Warmup
            for (int i = 0; i < WARMUP; ++i) {
                float a = data_a[i];
                float b = data_b[i];
                result += a + t * (b - a);
            }

            Timer timer;
            timer.start();
            for (int i = 0; i < ITERATIONS; ++i) {
                float a = data_a[i];
                float b = data_b[i];
                result += a + t * (b - a);
            }
            double time = timer.elapsed_ms();
            times.push_back(time);

            if (result == 1e30f) std::cout << "";
        }

        std::sort(times.begin(), times.end());
        double median = times[RUNS / 2];
        std::cout << "  std::       " << std::fixed << std::setprecision(2)
                  << median << " ms (median)\n";
    }

    // GLM
    {
        std::vector<double> times;
        for (int run = 0; run < RUNS; ++run) {
            float result = 0.0f;

            // Warmup
            for (int i = 0; i < WARMUP; ++i) {
                result += glm::mix(data_a[i], data_b[i], t);
            }

            Timer timer;
            timer.start();
            for (int i = 0; i < ITERATIONS; ++i) {
                result += glm::mix(data_a[i], data_b[i], t);
            }
            double time = timer.elapsed_ms();
            times.push_back(time);

            if (result == 1e30f) std::cout << "";
        }

        std::sort(times.begin(), times.end());
        double median = times[RUNS / 2];
        std::cout << "  glm::mix:   " << std::fixed << std::setprecision(2)
                  << median << " ms (median)\n";
    }
}

void benchmark_smoothstep_scalar_strict() {
    const int ITERATIONS = 10000000;
    const int WARMUP = 100000;
    const int RUNS = 5;

    auto data = generateRandomFloats(ITERATIONS, 0.0f, 1.0f);

    std::cout << "\nsmoothstep (scalar, " << ITERATIONS << " iterations, " << RUNS << " runs):\n";

    // MMath
    {
        std::vector<double> times;
        for (int run = 0; run < RUNS; ++run) {
            float result = 0.0f;

            // Warmup
            for (int i = 0; i < WARMUP; ++i) {
                result += smoothstep(0.0f, 1.0f, data[i]);
            }

            Timer timer;
            timer.start();
            for (int i = 0; i < ITERATIONS; ++i) {
                result += smoothstep(0.0f, 1.0f, data[i]);
            }
            double time = timer.elapsed_ms();
            times.push_back(time);

            if (result == 1e30f) std::cout << "";
        }

        std::sort(times.begin(), times.end());
        double median = times[RUNS / 2];
        std::cout << "  MMath:          " << std::fixed << std::setprecision(2)
                  << median << " ms (median)\n";
    }

    // GLM
    {
        std::vector<double> times;
        for (int run = 0; run < RUNS; ++run) {
            float result = 0.0f;

            // Warmup
            for (int i = 0; i < WARMUP; ++i) {
                result += glm::smoothstep(0.0f, 1.0f, data[i]);
            }

            Timer timer;
            timer.start();
            for (int i = 0; i < ITERATIONS; ++i) {
                result += glm::smoothstep(0.0f, 1.0f, data[i]);
            }
            double time = timer.elapsed_ms();
            times.push_back(time);

            if (result == 1e30f) std::cout << "";
        }

        std::sort(times.begin(), times.end());
        double median = times[RUNS / 2];
        std::cout << "  glm::smoothstep:" << std::fixed << std::setprecision(2)
                  << median << " ms (median)\n";
    }

    // Manual (like GLM)
    {
        std::vector<double> times;
        for (int run = 0; run < RUNS; ++run) {
            float result = 0.0f;

            // Warmup
            for (int i = 0; i < WARMUP; ++i) {
                float x = std::clamp((data[i] - 0.0f) / (1.0f - 0.0f), 0.0f, 1.0f);
                result += x * x * (3.0f - 2.0f * x);
            }

            Timer timer;
            timer.start();
            for (int i = 0; i < ITERATIONS; ++i) {
                float x = std::clamp((data[i] - 0.0f) / (1.0f - 0.0f), 0.0f, 1.0f);
                result += x * x * (3.0f - 2.0f * x);
            }
            double time = timer.elapsed_ms();
            times.push_back(time);

            if (result == 1e30f) std::cout << "";
        }

        std::sort(times.begin(), times.end());
        double median = times[RUNS / 2];
        std::cout << "  Manual:         " << std::fixed << std::setprecision(2)
                  << median << " ms (median)\n";
    }
}

void benchmark_lerpArray_strict() {
    const size_t SIZE = 1000000;
    const int RUNS = 5;

    auto data_a = generateRandomFloats(SIZE, 0.0f, 100.0f);
    auto data_b = generateRandomFloats(SIZE, 0.0f, 100.0f);
    std::vector<float> result(SIZE);
    float t = 0.3f;

    std::cout << "\nlerpArray (" << SIZE << " elements, " << RUNS << " runs):\n";

    // MMath SIMD
    {
        std::vector<double> times;
        for (int run = 0; run < RUNS; ++run) {
            Timer timer;
            timer.start();
            lerpArray(data_a.data(), data_b.data(), t, result.data(), static_cast<int32_t>(SIZE));
            double time = timer.elapsed_ms();
            times.push_back(time);
        }

        std::sort(times.begin(), times.end());
        double median = times[RUNS / 2];
        std::cout << "  MMath SIMD:  " << std::fixed << std::setprecision(3)
                  << median << " ms (median)\n";
    }

    // std:: loop
    {
        std::vector<double> times;
        for (int run = 0; run < RUNS; ++run) {
            Timer timer;
            timer.start();
            for (size_t i = 0; i < SIZE; ++i) {
                result[i] = data_a[i] + t * (data_b[i] - data_a[i]);
            }
            double time = timer.elapsed_ms();
            times.push_back(time);
        }

        std::sort(times.begin(), times.end());
        double median = times[RUNS / 2];
        std::cout << "  std:: loop:  " << std::fixed << std::setprecision(3)
                  << median << " ms (median)\n";
    }
}

void benchmark_smoothstepArray_strict() {
    const size_t SIZE = 1000000;
    const int RUNS = 5;

    auto data = generateRandomFloats(SIZE, 0.0f, 1.0f);
    std::vector<float> result_copy(SIZE);

    std::cout << "\nsmoothstepArray (" << SIZE << " elements, " << RUNS << " runs):\n";

    // MMath SIMD
    {
        std::vector<double> times;
        for (int run = 0; run < RUNS; ++run) {
            std::copy(data.begin(), data.end(), result_copy.begin());

            Timer timer;
            timer.start();
            smoothstepArray(0.0f, 1.0f, result_copy.data(), static_cast<int32_t>(SIZE));
            double time = timer.elapsed_ms();
            times.push_back(time);
        }

        std::sort(times.begin(), times.end());
        double median = times[RUNS / 2];
        std::cout << "  MMath SIMD:  " << std::fixed << std::setprecision(3)
                  << median << " ms (median)\n";
    }

    // std:: loop
    {
        std::vector<double> times;
        for (int run = 0; run < RUNS; ++run) {
            std::copy(data.begin(), data.end(), result_copy.begin());

            Timer timer;
            timer.start();
            for (size_t i = 0; i < SIZE; ++i) {
                float x = std::clamp((result_copy[i] - 0.0f) / (1.0f - 0.0f), 0.0f, 1.0f);
                result_copy[i] = x * x * (3.0f - 2.0f * x);
            }
            double time = timer.elapsed_ms();
            times.push_back(time);
        }

        std::sort(times.begin(), times.end());
        double median = times[RUNS / 2];
        std::cout << "  std:: loop:  " << std::fixed << std::setprecision(3)
                  << median << " ms (median)\n";
    }
}

int main() {
    std::cout << "========================================\n";
    std::cout << "Strict Lerp Performance Comparison\n";
    std::cout << "MMath vs GLM vs std::\n";
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

    std::cout << "\nNote: Using median of 5 runs for stability\n";
    std::cout << "      100K warmup iterations before each test\n\n";

    std::cout << "--- Scalar Operations ---\n";
    benchmark_lerp_scalar_strict();
    benchmark_smoothstep_scalar_strict();

    std::cout << "\n--- SIMD Array Operations ---\n";
    benchmark_lerpArray_strict();
    benchmark_smoothstepArray_strict();

    std::cout << "\n========================================\n";
    std::cout << "Benchmarks Complete\n";
    std::cout << "========================================\n";

    return 0;
}
