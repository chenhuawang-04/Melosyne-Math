/**
 * @file bench_mat3.cpp
 * @brief Standalone Mat3 performance benchmark
 */

#include <FastMath/mat3.h>

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
    std::cout << "  " << std::setw(30) << std::left << test_name << ": ";
    std::cout << std::setw(8) << std::right << time_ms << " ms  (";
    std::cout << std::setprecision(1) << (ops_per_sec / 1e6) << " M/s)\n";
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "�? fast_math Mat3 Performance Benchmark                     ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════╣\n";
    std::cout << "�? Layout: AOS (9 floats, 36 bytes)                         ║\n";
    std::cout << "�? SIMD:   AVX2 + FMA (8 matrices at once)                  ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    const int COUNT = 1000000;
    const int ITERS = 100;
    const int64_t TOTAL_OPS = (int64_t)COUNT * ITERS;

    std::cout << "Configuration:\n";
    std::cout << "  Elements:   " << COUNT << "\n";
    std::cout << "  Iterations: " << ITERS << "\n";
    std::cout << "  Total ops:  " << TOTAL_OPS << "\n\n";

    // Allocate memory
    std::vector<FastMath::Vec2> translations(COUNT);
    std::vector<float> rotations(COUNT);
    std::vector<FastMath::Vec2> scales(COUNT);
    std::vector<FastMath::Mat3> matrices_a(COUNT);
    std::vector<FastMath::Mat3> matrices_b(COUNT);
    std::vector<FastMath::Mat3> matrices_result(COUNT);
    std::vector<FastMath::Vec2> points(COUNT);
    std::vector<FastMath::Vec2> points_result(COUNT);

    // Generate random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist_pos(-100.0f, 100.0f);
    std::uniform_real_distribution<float> dist_angle(-3.14159f, 3.14159f);
    std::uniform_real_distribution<float> dist_scale(0.5f, 2.0f);

    for (int i = 0; i < COUNT; ++i) {
        translations[i] = {dist_pos(rng), dist_pos(rng)};
        rotations[i] = dist_angle(rng);
        scales[i] = {dist_scale(rng), dist_scale(rng)};
        points[i] = {dist_pos(rng), dist_pos(rng)};
    }

    // Pre-generate matrices for multiplication test
    for (int i = 0; i < COUNT; ++i) {
        matrices_a[i] = FastMath::mat3_from_trs(translations[i], rotations[i], scales[i]);
        matrices_b[i] = FastMath::mat3_from_trs(
            {dist_pos(rng), dist_pos(rng)},
            dist_angle(rng),
            {dist_scale(rng), dist_scale(rng)}
        );
    }

    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "�? Batch Operations (Array Processing)                      ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    // ---- Mat3 from TRS ----
    {
        auto func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                FastMath::mat3_from_trs_array(
                    translations.data(),
                    rotations.data(),
                    scales.data(),
                    matrices_result.data(),
                    COUNT
                );
            }
        };

        double time = benchmark_median(func);
        print_result("Mat3 from TRS", time, TOTAL_OPS);
    }

    // ---- Mat3 Multiply Affine ----
    {
        auto func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                FastMath::mat3_multiply_affine_array(
                    matrices_a.data(),
                    matrices_b.data(),
                    matrices_result.data(),
                    COUNT
                );
            }
        };

        double time = benchmark_median(func);
        print_result("Mat3 Multiply Affine", time, TOTAL_OPS);
    }

    // ---- Mat3 Transform Points (array) ----
    {
        auto func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                FastMath::mat3_transform_points_array(
                    matrices_a.data(),
                    points.data(),
                    points_result.data(),
                    COUNT
                );
            }
        };

        double time = benchmark_median(func);
        print_result("Mat3 Transform Points (array)", time, TOTAL_OPS);
    }

    // ---- Mat3 Transform Points (single matrix) ----
    {
        FastMath::Mat3 single_matrix = matrices_a[0];

        auto func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                FastMath::mat3_transform_points_single(
                    single_matrix,
                    points.data(),
                    points_result.data(),
                    COUNT
                );
            }
        };

        double time = benchmark_median(func);
        print_result("Mat3 Transform Points (single)", time, TOTAL_OPS);
    }

    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "�? Scalar Operations (Single Matrix)                        ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    // ---- Scalar: Mat3 from TRS ----
    {
        const int64_t SCALAR_OPS = TOTAL_OPS;

        auto func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                for (int i = 0; i < COUNT; ++i) {
                    matrices_result[i] = FastMath::mat3_from_trs(
                        translations[i], rotations[i], scales[i]
                    );
                }
            }
        };

        double time = benchmark_median(func);
        print_result("Scalar: Mat3 from TRS", time, SCALAR_OPS);
    }

    // ---- Scalar: Mat3 Multiply Affine ----
    {
        const int64_t SCALAR_OPS = TOTAL_OPS;

        auto func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                for (int i = 0; i < COUNT; ++i) {
                    matrices_result[i] = FastMath::mat3_multiply_affine(
                        matrices_a[i], matrices_b[i]
                    );
                }
            }
        };

        double time = benchmark_median(func);
        print_result("Scalar: Mat3 Multiply Affine", time, SCALAR_OPS);
    }

    // ---- Scalar: Mat3 Transform Point ----
    {
        const int64_t SCALAR_OPS = TOTAL_OPS;

        auto func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                for (int i = 0; i < COUNT; ++i) {
                    points_result[i] = FastMath::mat3_transform_point(
                        matrices_a[i], points[i]
                    );
                }
            }
        };

        double time = benchmark_median(func);
        print_result("Scalar: Mat3 Transform Point", time, SCALAR_OPS);
    }

    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "�? Correctness Verification                                  ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    // Test basic operations
    FastMath::Vec2 t = {10.0f, 20.0f};
    float r = 0.785398f;  // 45 degrees
    FastMath::Vec2 s = {2.0f, 3.0f};

    FastMath::Mat3 m = FastMath::mat3_from_trs(t, r, s);
    std::cout << "  TRS Matrix (t={10,20}, r=45°, s={2,3}):\n";
    std::cout << "    [" << m.m00 << ", " << m.m01 << ", " << m.m02 << "]\n";
    std::cout << "    [" << m.m10 << ", " << m.m11 << ", " << m.m12 << "]\n";
    std::cout << "    [" << m.m20 << ", " << m.m21 << ", " << m.m22 << "]\n\n";

    FastMath::Vec2 p = {5.0f, 0.0f};
    FastMath::Vec2 tp = FastMath::mat3_transform_point(m, p);
    std::cout << "  Transform point {5, 0}:\n";
    std::cout << "    Result: {" << tp.x << ", " << tp.y << "}\n\n";

    FastMath::Mat3 m_inv = FastMath::mat3_inverse_affine(m);
    FastMath::Vec2 p_back = FastMath::mat3_transform_point(m_inv, tp);
    std::cout << "  Inverse transform check:\n";
    std::cout << "    Original: {" << p.x << ", " << p.y << "}\n";
    std::cout << "    Recovered: {" << p_back.x << ", " << p_back.y << "}\n";
    std::cout << "    Error: {" << std::abs(p_back.x - p.x) << ", " << std::abs(p_back.y - p.y) << "}\n\n";

    std::cout << "  �?All tests completed successfully\n\n";

    return 0;
}
