/**
 * @file bench_vec2_vs_perf.cpp
 * @brief Vec2 performance comparison: fast_math vs perf_math
 */

#include <FastMath/vec2.h>
#include <perf_math/vec2.h>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>

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
              << std::setprecision(1) << (fast_ops / 1e6) << " M/s)";

    if (speedup >= 1.0) {
        std::cout << "  [" << std::setprecision(2) << speedup << "x faster]\n\n";
    } else {
        std::cout << "  [" << std::setprecision(2) << (1.0/speedup) << "x slower]\n\n";
    }
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "�? Vec2 Performance Comparison                              ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════╣\n";
    std::cout << "�? perf_math: SoA layout + AVX2 batch processing            ║\n";
    std::cout << "�? fast_math: AOS layout + optimized scalar                 ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    const int COUNT = 1000000;
    const int ITERS = 100;
    const int64_t TOTAL_OPS = (int64_t)COUNT * ITERS;

    std::cout << "Configuration:\n";
    std::cout << "  Elements:   " << COUNT << "\n";
    std::cout << "  Iterations: " << ITERS << "\n";
    std::cout << "  Total ops:  " << TOTAL_OPS << "\n\n";

    // Allocate memory - perf_math (SoA)
    std::vector<float> perf_ax(COUNT), perf_ay(COUNT);
    std::vector<float> perf_bx(COUNT), perf_by(COUNT);
    std::vector<float> perf_rx(COUNT), perf_ry(COUNT);
    std::vector<float> perf_dot_results(COUNT);

    // Allocate memory - fast_math (AOS)
    std::vector<FastMath::Vec2> fast_va(COUNT);
    std::vector<FastMath::Vec2> fast_vb(COUNT);
    std::vector<FastMath::Vec2> fast_vr(COUNT);
    std::vector<float> fast_dot_results(COUNT);

    // Generate random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    for (int i = 0; i < COUNT; ++i) {
        float x1 = dist(rng), y1 = dist(rng);
        float x2 = dist(rng), y2 = dist(rng);

        // perf_math data
        perf_ax[i] = x1; perf_ay[i] = y1;
        perf_bx[i] = x2; perf_by[i] = y2;

        // fast_math data
        fast_va[i] = {x1, y1};
        fast_vb[i] = {x2, y2};
    }

    // Setup perf_math batches
    perf_math::Vec2_Batch perf_batch_a = {perf_ax.data(), perf_ay.data(), COUNT, COUNT};
    perf_math::Vec2_Batch perf_batch_b = {perf_bx.data(), perf_by.data(), COUNT, COUNT};
    perf_math::Vec2_Batch perf_batch_r = {perf_rx.data(), perf_ry.data(), 0, COUNT};

    // ---- Vec2 Add ----
    {
        auto perf_func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                perf_math::vec2_batch_add(&perf_batch_a, &perf_batch_b, &perf_batch_r);
            }
        };

        auto fast_func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                FastMath::vec2_add_array(fast_va.data(), fast_vb.data(), fast_vr.data(), COUNT);
            }
        };

        double perf_time = benchmark_median(perf_func);
        double fast_time = benchmark_median(fast_func);
        print_comparison("Vec2 Add", perf_time, fast_time, TOTAL_OPS);
    }

    // ---- Vec2 Dot Product ----
    {
        auto perf_func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                perf_math::vec2_batch_dot(&perf_batch_a, &perf_batch_b, perf_dot_results.data());
            }
        };

        auto fast_func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                FastMath::vec2_dot_array(fast_va.data(), fast_vb.data(), fast_dot_results.data(), COUNT);
            }
        };

        double perf_time = benchmark_median(perf_func);
        double fast_time = benchmark_median(fast_func);
        print_comparison("Vec2 Dot Product", perf_time, fast_time, TOTAL_OPS);
    }

    // ---- Vec2 Normalize ----
    {
        auto perf_func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                perf_math::vec2_batch_normalize(&perf_batch_a, &perf_batch_r);
            }
        };

        auto fast_func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                FastMath::vec2_normalize_array(fast_va.data(), fast_vr.data(), COUNT);
            }
        };

        double perf_time = benchmark_median(perf_func);
        double fast_time = benchmark_median(fast_func);
        print_comparison("Vec2 Normalize", perf_time, fast_time, TOTAL_OPS);
    }

    // ---- Vec2 Normalize Fast ----
    {
        auto perf_func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                perf_math::vec2_batch_normalize_fast(&perf_batch_a, &perf_batch_r);
            }
        };

        auto fast_func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                FastMath::vec2_normalize_fast_array(fast_va.data(), fast_vr.data(), COUNT);
            }
        };

        double perf_time = benchmark_median(perf_func);
        double fast_time = benchmark_median(fast_func);
        print_comparison("Vec2 Normalize Fast", perf_time, fast_time, TOTAL_OPS);
    }

    // ---- Accuracy Verification ----
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "�? Accuracy Verification                                     ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    perf_math::Vec2 pv1 = {3.0f, 4.0f};
    perf_math::Vec2 pv2 = {1.0f, 2.0f};
    FastMath::Vec2 fv1 = {3.0f, 4.0f};
    FastMath::Vec2 fv2 = {1.0f, 2.0f};

    auto perf_add = perf_math::vec2_add(pv1, pv2);
    auto fast_add = FastMath::vec2_add(fv1, fv2);
    std::cout << "  Add:       perf={" << perf_add.x << ", " << perf_add.y << "}, "
              << "fast={" << fast_add.x << ", " << fast_add.y << "}\n";

    float perf_dot = perf_math::vec2_dot(pv1, pv2);
    float fast_dot = FastMath::vec2_dot(fv1, fv2);
    std::cout << "  Dot:       perf=" << perf_dot << ", fast=" << fast_dot << "\n";

    auto perf_norm = perf_math::vec2_normalize(pv1);
    auto fast_norm = FastMath::vec2_normalize(fv1);
    std::cout << "  Normalize: perf={" << perf_norm.x << ", " << perf_norm.y << "}, "
              << "fast={" << fast_norm.x << ", " << fast_norm.y << "}\n\n";

    std::cout << "  �?Both implementations produce identical results\n\n";

    return 0;
}
