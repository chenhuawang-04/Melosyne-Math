/**
 * @file bench_simd.cpp
 * @brief Benchmark hand-written SIMD vs auto-vectorization
 */

#include <FastMath/trig.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

class Timer {
    std::chrono::high_resolution_clock::time_point start_time;
public:
    void start() { start_time = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_time).count();
    }
};

template<typename Func>
double benchmark_median(Func&& func, int runs = 5) {
    std::vector<double> times;
    times.reserve(runs);

    func();  // Warm-up

    Timer timer;
    for (int r = 0; r < runs; ++r) {
        timer.start();
        func();
        times.push_back(timer.elapsed_ms());
    }

    std::sort(times.begin(), times.end());
    return times[runs / 2];
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "�? fast_math - Hand-written SIMD Benchmark                  ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════╣\n";
#if defined(__AVX2__) && defined(__FMA__)
    std::cout << "�? SIMD: AVX2 + FMA (8 floats per cycle)                    ║\n";
#elif defined(__SSE4_1__)
    std::cout << "�? SIMD: SSE4.1 (4 floats per cycle)                        ║\n";
#else
    std::cout << "�? SIMD: DISABLED (scalar only)                             ║\n";
#endif
    std::cout << "�? Design: Strict POD + Hand-written intrinsics             ║\n";
    std::cout << "�? Reference: DirectXMath 11/10-degree minimax              ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    const int COUNT = 1000000;
    const int ITERS = 100;
    const int64_t TOTAL_OPS = (int64_t)COUNT * ITERS;

    std::cout << "Configuration:\n";
    std::cout << "  Elements:   " << COUNT << "\n";
    std::cout << "  Iterations: " << ITERS << "\n";
    std::cout << "  Total ops:  " << TOTAL_OPS << "\n\n";

    // Allocate memory
    std::vector<float> angles(COUNT);
    std::vector<float> sins(COUNT);
    std::vector<float> coss(COUNT);

    // Generate random angles
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-(float)M_PI * 4, (float)M_PI * 4);
    for (int i = 0; i < COUNT; ++i) {
        angles[i] = dist(rng);
    }

    // ---- Benchmark: Hand-written SIMD ----
    {
        auto bench_func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                FastMath::sincos_array(angles.data(), sins.data(), coss.data(), COUNT);
            }
        };

        double time_ms = benchmark_median(bench_func);
        double ops_per_sec = TOTAL_OPS / time_ms * 1000.0;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  fast_math (hand-written SIMD):\n";
        std::cout << "    Time:       " << std::setw(8) << time_ms << " ms\n";
        std::cout << "    Throughput: " << std::setprecision(1) << (ops_per_sec / 1e6) << " M ops/sec\n\n";
    }

    // ---- Benchmark: std::sin + std::cos (reference) ----
    {
        auto bench_func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                for (int i = 0; i < COUNT; ++i) {
                    sins[i] = std::sin(angles[i]);
                    coss[i] = std::cos(angles[i]);
                }
            }
        };

        double time_ms = benchmark_median(bench_func);
        double ops_per_sec = TOTAL_OPS / time_ms * 1000.0;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  std::sin + std::cos (reference):\n";
        std::cout << "    Time:       " << std::setw(8) << time_ms << " ms\n";
        std::cout << "    Throughput: " << std::setprecision(1) << (ops_per_sec / 1e6) << " M ops/sec\n\n";
    }

    // ---- Accuracy Check ----
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "�? Accuracy Check (vs std::sin/cos)                         ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    double max_sin_err = 0, max_cos_err = 0;
    for (int i = 0; i < std::min(COUNT, 10000); ++i) {
        FastMath::SinCos result;
        FastMath::sincos(FastMath::Angle{angles[i]}, &result);

        double ref_sin = std::sin(angles[i]);
        double ref_cos = std::cos(angles[i]);

        max_sin_err = std::max(max_sin_err, std::abs(result.sin - ref_sin));
        max_cos_err = std::max(max_cos_err, std::abs(result.cos - ref_cos));
    }

    std::cout << std::scientific;
    std::cout << "  Maximum error (10,000 samples):\n";
    std::cout << "    sin: " << max_sin_err << "\n";
    std::cout << "    cos: " << max_cos_err << "\n\n";

    if (max_sin_err < 1e-6 && max_cos_err < 1e-6) {
        std::cout << "  �?Accuracy: PASSED (error < 1e-6)\n\n";
    } else {
        std::cout << "  �?Accuracy: FAILED (error >= 1e-6)\n\n";
    }

    return 0;
}
