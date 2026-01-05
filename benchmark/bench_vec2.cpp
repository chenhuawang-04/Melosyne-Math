/**
 * @file bench_vec2.cpp
 * @brief Vec2 performance benchmark
 */

#include <FastMath/Vec2.h>
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

void print_result(const std::string& test_name, double time_ms, int64_t ops) {
    double ops_per_sec = ops / time_ms * 1000.0;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  " << test_name << ":\n";
    std::cout << "    Time:       " << std::setw(8) << time_ms << " ms\n";
    std::cout << "    Throughput: " << std::setprecision(1) << (ops_per_sec / 1e6) << " M ops/sec\n\n";
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "�? fast_math - Vec2 Benchmark                               ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════╣\n";
    std::cout << "�? Design: Strict POD + Optimized scalar operations        ║\n";
    std::cout << "�? Layout: AOS (Array of Structures)                       ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    const int COUNT = 1000000;
    const int ITERS = 100;
    const int64_t TOTAL_OPS = (int64_t)COUNT * ITERS;

    std::cout << "Configuration:\n";
    std::cout << "  Elements:   " << COUNT << "\n";
    std::cout << "  Iterations: " << ITERS << "\n";
    std::cout << "  Total ops:  " << TOTAL_OPS << "\n\n";

    // Allocate memory
    std::vector<FastMath::Vec2> vectors_a(COUNT);
    std::vector<FastMath::Vec2> vectors_b(COUNT);
    std::vector<FastMath::Vec2> vectors_result(COUNT);
    std::vector<float> dot_results(COUNT);

    // Generate random vectors
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    for (int i = 0; i < COUNT; ++i) {
        vectors_a[i] = { dist(rng), dist(rng) };
        vectors_b[i] = { dist(rng), dist(rng) };
    }

    // ---- Benchmark: Vec2 Add ----
    {
        auto bench_func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                FastMath::vec2_add_array(vectors_a.data(), vectors_b.data(),
                                         vectors_result.data(), COUNT);
            }
        };

        double time_ms = benchmark_median(bench_func);
        print_result("vec2_add_array", time_ms, TOTAL_OPS);
    }

    // ---- Benchmark: Vec2 Dot Product ----
    {
        auto bench_func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                FastMath::vec2_dot_array(vectors_a.data(), vectors_b.data(),
                                         dot_results.data(), COUNT);
            }
        };

        double time_ms = benchmark_median(bench_func);
        print_result("vec2_dot_array", time_ms, TOTAL_OPS);
    }

    // ---- Benchmark: Vec2 Normalize ----
    {
        auto bench_func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                FastMath::vec2_normalize_array(vectors_a.data(),
                                               vectors_result.data(), COUNT);
            }
        };

        double time_ms = benchmark_median(bench_func);
        print_result("vec2_normalize_array", time_ms, TOTAL_OPS);
    }

    // ---- Benchmark: Vec2 Normalize Fast ----
    {
        auto bench_func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                FastMath::vec2_normalize_fast_array(vectors_a.data(),
                                                     vectors_result.data(), COUNT);
            }
        };

        double time_ms = benchmark_median(bench_func);
        print_result("vec2_normalize_fast_array", time_ms, TOTAL_OPS);
    }

    // ---- Accuracy Check ----
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "�? Accuracy Check                                            ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    FastMath::Vec2 v1 = { 3.0f, 4.0f };
    FastMath::Vec2 v2 = { 1.0f, 2.0f };

    std::cout << "  Test vectors: v1={3, 4}, v2={1, 2}\n\n";

    auto result_add = FastMath::vec2_add(v1, v2);
    std::cout << "  vec2_add:    {" << result_add.x << ", " << result_add.y << "} (expected {4, 6})\n";

    float result_dot = FastMath::vec2_dot(v1, v2);
    std::cout << "  vec2_dot:    " << result_dot << " (expected 11)\n";

    float result_len = FastMath::vec2_length(v1);
    std::cout << "  vec2_length: " << result_len << " (expected 5)\n";

    auto result_norm = FastMath::vec2_normalize(v1);
    std::cout << "  vec2_normalize: {" << result_norm.x << ", " << result_norm.y << "} (expected {0.6, 0.8})\n";

    float cross_result = FastMath::vec2_cross(v1, v2);
    std::cout << "  vec2_cross:  " << cross_result << " (expected 2)\n";

    auto perp = FastMath::vec2_perpendicular(v1);
    std::cout << "  vec2_perpendicular: {" << perp.x << ", " << perp.y << "} (expected {-4, 3})\n\n";

    return 0;
}
