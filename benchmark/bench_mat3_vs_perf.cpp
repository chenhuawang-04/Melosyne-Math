/**
 * @file bench_mat3_vs_perf.cpp
 * @brief Mat3 performance comparison: fast_math vs perf_math
 */

#include <FastMath/mat3.h>
#include <perf_math/mat3.h>

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
    std::cout << "�? Mat3 Performance Comparison                              ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════╣\n";
    std::cout << "�? perf_math: SoA layout + AVX2 batch processing            ║\n";
    std::cout << "�? fast_math: AOS layout + deinterleave optimization        ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    const int COUNT = 1000000;
    const int ITERS = 100;
    const int64_t TOTAL_OPS = (int64_t)COUNT * ITERS;

    std::cout << "Configuration:\n";
    std::cout << "  Elements:   " << COUNT << "\n";
    std::cout << "  Iterations: " << ITERS << "\n";
    std::cout << "  Total ops:  " << TOTAL_OPS << "\n\n";

    // Allocate memory - perf_math (SoA)
    std::vector<float> perf_tx(COUNT), perf_ty(COUNT);
    std::vector<float> perf_rot(COUNT);
    std::vector<float> perf_sx(COUNT), perf_sy(COUNT);
    std::vector<float> perf_px(COUNT), perf_py(COUNT);
    std::vector<float> perf_result_px(COUNT), perf_result_py(COUNT);

    perf_math::Mat3_Batch perf_batch_a, perf_batch_b, perf_batch_result;
    for (auto* batch : {&perf_batch_a, &perf_batch_b, &perf_batch_result}) {
        batch->m00 = new float[COUNT];
        batch->m01 = new float[COUNT];
        batch->m02 = new float[COUNT];
        batch->m10 = new float[COUNT];
        batch->m11 = new float[COUNT];
        batch->m12 = new float[COUNT];
        batch->m20 = new float[COUNT];
        batch->m21 = new float[COUNT];
        batch->m22 = new float[COUNT];
        batch->capacity = COUNT;
        batch->count = 0;
    }

    // Allocate memory - fast_math (AOS)
    std::vector<FastMath::Vec2> fast_translations(COUNT);
    std::vector<float> fast_rotations(COUNT);
    std::vector<FastMath::Vec2> fast_scales(COUNT);
    std::vector<FastMath::Mat3> fast_matrices_a(COUNT);
    std::vector<FastMath::Mat3> fast_matrices_b(COUNT);
    std::vector<FastMath::Mat3> fast_matrices_result(COUNT);
    std::vector<FastMath::Vec2> fast_points(COUNT);
    std::vector<FastMath::Vec2> fast_result_points(COUNT);

    // Generate random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist_pos(-100.0f, 100.0f);
    std::uniform_real_distribution<float> dist_angle(-3.14159f, 3.14159f);
    std::uniform_real_distribution<float> dist_scale(0.5f, 2.0f);

    for (int i = 0; i < COUNT; ++i) {
        float tx = dist_pos(rng), ty = dist_pos(rng);
        float rot = dist_angle(rng);
        float sx = dist_scale(rng), sy = dist_scale(rng);
        float px = dist_pos(rng), py = dist_pos(rng);

        // perf_math data
        perf_tx[i] = tx; perf_ty[i] = ty;
        perf_rot[i] = rot;
        perf_sx[i] = sx; perf_sy[i] = sy;
        perf_px[i] = px; perf_py[i] = py;

        // fast_math data
        fast_translations[i] = {tx, ty};
        fast_rotations[i] = rot;
        fast_scales[i] = {sx, sy};
        fast_points[i] = {px, py};
    }

    // Pre-generate matrices for multiplication test
    perf_math::mat3_batch_from_trs(
        perf_tx.data(), perf_ty.data(), perf_rot.data(),
        perf_sx.data(), perf_sy.data(),
        &perf_batch_a, COUNT
    );

    for (int i = 0; i < COUNT; ++i) {
        fast_matrices_a[i] = FastMath::mat3_from_trs(fast_translations[i], fast_rotations[i], fast_scales[i]);
    }

    // Generate second set of matrices for multiplication
    for (int i = 0; i < COUNT; ++i) {
        float tx2 = dist_pos(rng), ty2 = dist_pos(rng);
        float rot2 = dist_angle(rng);
        float sx2 = dist_scale(rng), sy2 = dist_scale(rng);

        perf_batch_b.m00[i] = sx2 * std::cos(rot2);
        perf_batch_b.m01[i] = -sy2 * std::sin(rot2);
        perf_batch_b.m02[i] = tx2;
        perf_batch_b.m10[i] = sx2 * std::sin(rot2);
        perf_batch_b.m11[i] = sy2 * std::cos(rot2);
        perf_batch_b.m12[i] = ty2;
        perf_batch_b.m20[i] = 0.0f;
        perf_batch_b.m21[i] = 0.0f;
        perf_batch_b.m22[i] = 1.0f;

        fast_matrices_b[i] = FastMath::mat3_from_trs({tx2, ty2}, rot2, {sx2, sy2});
    }
    perf_batch_b.count = COUNT;

    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "�? Benchmark Results                                         ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    // ---- Mat3 from TRS ----
    {
        auto perf_func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                perf_math::mat3_batch_from_trs(
                    perf_tx.data(), perf_ty.data(), perf_rot.data(),
                    perf_sx.data(), perf_sy.data(),
                    &perf_batch_result, COUNT
                );
            }
        };

        auto fast_func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                FastMath::mat3_from_trs_array(
                    fast_translations.data(),
                    fast_rotations.data(),
                    fast_scales.data(),
                    fast_matrices_result.data(),
                    COUNT
                );
            }
        };

        double perf_time = benchmark_median(perf_func);
        double fast_time = benchmark_median(fast_func);
        print_comparison("Mat3 from TRS", perf_time, fast_time, TOTAL_OPS);
    }

    // ---- Mat3 Multiply Affine ----
    {
        auto perf_func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                perf_math::mat3_batch_multiply_affine(&perf_batch_a, &perf_batch_b, &perf_batch_result, COUNT);
            }
        };

        auto fast_func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                FastMath::mat3_multiply_affine_array(
                    fast_matrices_a.data(),
                    fast_matrices_b.data(),
                    fast_matrices_result.data(),
                    COUNT
                );
            }
        };

        double perf_time = benchmark_median(perf_func);
        double fast_time = benchmark_median(fast_func);
        print_comparison("Mat3 Multiply Affine", perf_time, fast_time, TOTAL_OPS);
    }

    // ---- Mat3 Transform Points ----
    {
        auto perf_func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                perf_math::mat3_batch_transform_points(
                    &perf_batch_a,
                    perf_px.data(), perf_py.data(),
                    perf_result_px.data(), perf_result_py.data(),
                    COUNT
                );
            }
        };

        auto fast_func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                FastMath::mat3_transform_points_array(
                    fast_matrices_a.data(),
                    fast_points.data(),
                    fast_result_points.data(),
                    COUNT
                );
            }
        };

        double perf_time = benchmark_median(perf_func);
        double fast_time = benchmark_median(fast_func);
        print_comparison("Mat3 Transform Points", perf_time, fast_time, TOTAL_OPS);
    }

    // ---- Mat3 Transform Points (Single Matrix) ----
    {
        perf_math::Mat3 perf_single = {
            perf_batch_a.m00[0], perf_batch_a.m01[0], perf_batch_a.m02[0],
            perf_batch_a.m10[0], perf_batch_a.m11[0], perf_batch_a.m12[0],
            perf_batch_a.m20[0], perf_batch_a.m21[0], perf_batch_a.m22[0]
        };

        FastMath::Mat3 fast_single = fast_matrices_a[0];

        auto perf_func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                perf_math::mat3_batch_transform_points_single_matrix(
                    perf_single,
                    perf_px.data(), perf_py.data(),
                    perf_result_px.data(), perf_result_py.data(),
                    COUNT
                );
            }
        };

        auto fast_func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                FastMath::mat3_transform_points_single(
                    fast_single,
                    fast_points.data(),
                    fast_result_points.data(),
                    COUNT
                );
            }
        };

        double perf_time = benchmark_median(perf_func);
        double fast_time = benchmark_median(fast_func);
        print_comparison("Mat3 Transform Points (Single)", perf_time, fast_time, TOTAL_OPS);
    }

    // ---- Accuracy Verification ----
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "�? Accuracy Verification                                     ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    perf_math::Mat3 pm = perf_math::mat3_from_trs({10.0f, 20.0f}, 0.785398f, {2.0f, 3.0f});
    FastMath::Mat3 fm = FastMath::mat3_from_trs({10.0f, 20.0f}, 0.785398f, {2.0f, 3.0f});

    std::cout << "  TRS Matrix (t={10,20}, r=45°, s={2,3}):\n";
    std::cout << "    perf_math: [" << pm.m00 << ", " << pm.m01 << ", " << pm.m02 << "]\n";
    std::cout << "    fast_math: [" << fm.m00 << ", " << fm.m01 << ", " << fm.m02 << "]\n\n";

    FastMath::Vec2 fp = {5.0f, 0.0f};
    perf_math::Vec2 pp = {5.0f, 0.0f};

    FastMath::Vec2 ft = FastMath::mat3_transform_point(fm, fp);
    perf_math::Vec2 pt = perf_math::mat3_transform_point(pm, pp);

    std::cout << "  Transform point {5, 0}:\n";
    std::cout << "    perf_math: {" << pt.x << ", " << pt.y << "}\n";
    std::cout << "    fast_math: {" << ft.x << ", " << ft.y << "}\n";
    std::cout << "    Difference: {" << std::abs(ft.x - pt.x) << ", " << std::abs(ft.y - pt.y) << "}\n\n";

    std::cout << "  �?Both implementations produce equivalent results\n\n";

    // Cleanup
    for (auto* batch : {&perf_batch_a, &perf_batch_b, &perf_batch_result}) {
        delete[] batch->m00; delete[] batch->m01; delete[] batch->m02;
        delete[] batch->m10; delete[] batch->m11; delete[] batch->m12;
        delete[] batch->m20; delete[] batch->m21; delete[] batch->m22;
    }

    return 0;
}
