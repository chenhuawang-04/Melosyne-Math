/**
 * @file bench_fast_vs_perf.cpp
 * @brief Performance comparison: fast_math vs perf_math
 */

#include <FastMath/trig.h>
#include <perf_math/trig.h>

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

void print_comparison(const std::string& test_name,
                      double perf_time,
                      double fast_time,
                      int64_t ops) {
    double perf_ops = ops / perf_time * 1000.0;
    double fast_ops = ops / fast_time * 1000.0;
    double speedup = perf_time / fast_time;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  " << test_name << ":\n";
    std::cout << "    perf_math: " << std::setw(8) << perf_time << " ms  ("
              << std::setprecision(1) << (perf_ops / 1e6) << " M/s)\n";
    std::cout << "    fast_math: " << std::setw(8) << std::setprecision(2) << fast_time << " ms  ("
              << std::setprecision(1) << (fast_ops / 1e6) << " M/s)"
              << "  [" << std::setprecision(2) << speedup << "x]\n\n";
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "�? fast_math vs perf_math - Performance Comparison          ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════╣\n";
    std::cout << "�? fast_math: MiniMax polynomials (error < 1e-7)            ║\n";
    std::cout << "�? - Degree 7 sine, Degree 6 cosine                         ║\n";
    std::cout << "�? - Horner evaluation with FMA                             ║\n";
    std::cout << "�? - Compiler-friendly for auto-vectorization               ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    const int COUNT = 1000000;
    const int ITERS = 100;
    const int64_t TOTAL_OPS = (int64_t)COUNT * ITERS;

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

    // perf_math batch structures
    perf_math::Angles_Batch perf_angles = {angles.data(), COUNT, COUNT};
    perf_math::SinCos_Batch perf_sc = {sins.data(), coss.data(), 0, COUNT};

    // ---- SinCos Batch ----
    {
        auto perf_func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                perf_math::sincos_batch(&perf_angles, &perf_sc);
            }
        };

        auto fast_func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                FastMath::sincos_array(angles.data(), sins.data(), coss.data(), COUNT);
            }
        };

        double perf_time = benchmark_median(perf_func);
        double fast_time = benchmark_median(fast_func);
        print_comparison("SinCos Batch", perf_time, fast_time, TOTAL_OPS);
    }

    // ---- Accuracy Check ----
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "�? Accuracy Check (max error vs std::sin/cos)               ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    double perf_max_sin_err = 0, perf_max_cos_err = 0;
    double fast_max_sin_err = 0, fast_max_cos_err = 0;

    for (int i = 0; i < std::min(COUNT, 10000); ++i) {
        float perf_s, perf_c;
        perf_math::sincos_scalar(angles[i], &perf_s, &perf_c);

        FastMath::SinCos fast_result;
        FastMath::sincos(FastMath::Angle{angles[i]}, &fast_result);

        double ref_sin = std::sin(angles[i]);
        double ref_cos = std::cos(angles[i]);

        perf_max_sin_err = std::max(perf_max_sin_err, std::abs(perf_s - ref_sin));
        perf_max_cos_err = std::max(perf_max_cos_err, std::abs(perf_c - ref_cos));
        fast_max_sin_err = std::max(fast_max_sin_err, std::abs(fast_result.sin - ref_sin));
        fast_max_cos_err = std::max(fast_max_cos_err, std::abs(fast_result.cos - ref_cos));
    }

    std::cout << std::scientific;
    std::cout << "  perf_math: sin=" << perf_max_sin_err << ", cos=" << perf_max_cos_err << "\n";
    std::cout << "  fast_math: sin=" << fast_max_sin_err << ", cos=" << fast_max_cos_err << "\n\n";

    return 0;
}
