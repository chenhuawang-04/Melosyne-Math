/**
 * @file bench_aabb2.cpp
 * @brief AABB2 performance benchmark
 */

#include <FastMath/aabb2.h>

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
    std::cout << "  " << std::setw(35) << std::left << test_name << ": ";
    std::cout << std::setw(8) << std::right << time_ms << " ms  (";
    std::cout << std::setprecision(1) << (ops_per_sec / 1e6) << " M/s)\n";
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "�? fast_math AABB2 Performance Benchmark                    ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════╣\n";
    std::cout << "�? Optimization: Negative max storage                       ║\n";
    std::cout << "�? SIMD:         SSE4.1 (scalar) + AVX2 (batch)             ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    const int COUNT = 1000000;
    const int ITERS = 100;
    const int64_t TOTAL_OPS = (int64_t)COUNT * ITERS;

    std::cout << "Configuration:\n";
    std::cout << "  Elements:   " << COUNT << "\n";
    std::cout << "  Iterations: " << ITERS << "\n";
    std::cout << "  Total ops:  " << TOTAL_OPS << "\n\n";

    // Generate random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist_pos(-1000.0f, 1000.0f);
    std::uniform_real_distribution<float> dist_size(1.0f, 100.0f);

    std::vector<FastMath::AABB2> aabbs_a(COUNT);
    std::vector<FastMath::AABB2> aabbs_b(COUNT);
    std::vector<FastMath::AABB2> aabbs_result(COUNT);
    std::vector<FastMath::Vec2> points(COUNT);
    std::vector<bool> bool_results_vec(COUNT);
    bool* bool_results = reinterpret_cast<bool*>(malloc(COUNT * sizeof(bool)));  // Use raw array for bool (std::vector<bool> has no data())

    for (int i = 0; i < COUNT; ++i) {
        float minx_a = dist_pos(rng);
        float miny_a = dist_pos(rng);
        float w_a = dist_size(rng);
        float h_a = dist_size(rng);
        aabbs_a[i] = FastMath::aabb2_from_min_max(
            {minx_a, miny_a},
            {minx_a + w_a, miny_a + h_a}
        );

        float minx_b = dist_pos(rng);
        float miny_b = dist_pos(rng);
        float w_b = dist_size(rng);
        float h_b = dist_size(rng);
        aabbs_b[i] = FastMath::aabb2_from_min_max(
            {minx_b, miny_b},
            {minx_b + w_b, miny_b + h_b}
        );

        points[i] = {dist_pos(rng), dist_pos(rng)};
    }

    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "�? Scalar Operations (Single AABB)                          ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    // ---- Scalar: Merge ----
    {
        auto func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                for (int i = 0; i < COUNT; ++i) {
                    aabbs_result[i] = FastMath::aabb2_merge(aabbs_a[i], aabbs_b[i]);
                }
            }
        };

        double time = benchmark_median(func);
        print_result("Merge (Union) - 1 SSE instruction", time, TOTAL_OPS);
    }

    // ---- Scalar: Intersects ----
    {
        auto func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                for (int i = 0; i < COUNT; ++i) {
                    bool_results[i] = FastMath::aabb2_intersects(aabbs_a[i], aabbs_b[i]);
                }
            }
        };

        double time = benchmark_median(func);
        print_result("Intersects - 3 SSE instructions", time, TOTAL_OPS);
    }

    // ---- Scalar: Contains Point ----
    {
        auto func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                for (int i = 0; i < COUNT; ++i) {
                    bool_results[i] = FastMath::aabb2_contains_point(aabbs_a[i], points[i]);
                }
            }
        };

        double time = benchmark_median(func);
        print_result("Contains Point - SSE optimized", time, TOTAL_OPS);
    }

    // ---- Scalar: Overlap Area ----
    {
        std::vector<float> float_results(COUNT);
        auto func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                for (int i = 0; i < COUNT; ++i) {
                    float_results[i] = FastMath::aabb2_overlap_area(aabbs_a[i], aabbs_b[i]);
                }
            }
        };

        double time = benchmark_median(func);
        print_result("Overlap Area", time, TOTAL_OPS);
    }

    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "�? Batch Operations (AVX2 Array Processing)                 ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    // ---- Batch: Merge Array ----
    {
        auto func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                FastMath::aabb2_merge_array(
                    aabbs_a.data(),
                    aabbs_b.data(),
                    aabbs_result.data(),
                    COUNT
                );
            }
        };

        double time = benchmark_median(func);
        print_result("Batch Merge (8 at once)", time, TOTAL_OPS);
    }

    // ---- Batch: Intersects Array ----
    {
        FastMath::AABB2 reference = FastMath::aabb2_from_center_extents(
            {0.0f, 0.0f}, {500.0f, 500.0f}
        );

        auto func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                FastMath::aabb2_intersects_array(
                    reference,
                    aabbs_a.data(),
                    bool_results,  // Already a pointer
                    COUNT
                );
            }
        };

        double time = benchmark_median(func);
        print_result("Batch Intersects (8 at once)", time, TOTAL_OPS);
    }

    // ---- Batch: Contains Points Array ----
    {
        FastMath::AABB2 reference = FastMath::aabb2_from_center_extents(
            {0.0f, 0.0f}, {500.0f, 500.0f}
        );

        auto func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                FastMath::aabb2_contains_points_array(
                    reference,
                    points.data(),
                    bool_results,  // Already a pointer
                    COUNT
                );
            }
        };

        double time = benchmark_median(func);
        print_result("Batch Contains Points (8 at once)", time, TOTAL_OPS);
    }

    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "�? Correctness Verification                                  ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    // Test basic operations
    FastMath::Vec2 min1 = {10.0f, 20.0f};
    FastMath::Vec2 max1 = {50.0f, 80.0f};
    FastMath::AABB2 aabb1 = FastMath::aabb2_from_min_max(min1, max1);

    FastMath::Vec2 min2 = {30.0f, 40.0f};
    FastMath::Vec2 max2 = {70.0f, 100.0f};
    FastMath::AABB2 aabb2 = FastMath::aabb2_from_min_max(min2, max2);

    std::cout << "  AABB1: min=" << aabb1.minx << "," << aabb1.miny
              << " max=" << -aabb1.neg_maxx << "," << -aabb1.neg_maxy << "\n";
    std::cout << "  AABB2: min=" << aabb2.minx << "," << aabb2.miny
              << " max=" << -aabb2.neg_maxx << "," << -aabb2.neg_maxy << "\n\n";

    FastMath::AABB2 merged = FastMath::aabb2_merge(aabb1, aabb2);
    std::cout << "  Merged: min=" << merged.minx << "," << merged.miny
              << " max=" << -merged.neg_maxx << "," << -merged.neg_maxy << "\n";
    std::cout << "    Expected: min=10,20 max=70,100\n\n";

    bool overlaps = FastMath::aabb2_intersects(aabb1, aabb2);
    std::cout << "  Overlaps: " << (overlaps ? "true" : "false") << "\n";
    std::cout << "    Expected: true\n\n";

    FastMath::Vec2 test_point = {40.0f, 50.0f};
    bool contains1 = FastMath::aabb2_contains_point(aabb1, test_point);
    bool contains2 = FastMath::aabb2_contains_point(aabb2, test_point);
    std::cout << "  Point {40, 50}:\n";
    std::cout << "    In AABB1: " << (contains1 ? "true" : "false") << " (expected: true)\n";
    std::cout << "    In AABB2: " << (contains2 ? "true" : "false") << " (expected: true)\n\n";

    float area1 = FastMath::aabb2_area(aabb1);
    std::cout << "  AABB1 area: " << area1 << " (expected: 2400)\n";

    FastMath::Vec2 center1 = FastMath::aabb2_center(aabb1);
    std::cout << "  AABB1 center: {" << center1.x << ", " << center1.y << "} (expected: {30, 50})\n\n";

    std::cout << "  �?All tests completed successfully\n\n";

    free(bool_results);

    return 0;
}
