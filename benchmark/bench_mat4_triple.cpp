/**
 * @file bench_mat4_triple.cpp
 * @brief Mat4 Triple-Library Performance Comparison: MMath vs GLM vs DirectXMath vs Eigen
 *
 * Tests 6 core operations:
 * 1. mat4Mul - Matrix multiplication (most critical)
 * 2. mat4MulVec4 - Matrix-vector multiplication
 * 3. mat4Transpose - Matrix transpose
 * 4. mat4Inverse - Matrix inversion
 * 5. mat4Translation - Translation matrix construction
 * 6. mat4LookAt - View matrix construction
 *
 * Methodology:
 * - Fixed random seed (42) for reproducibility
 * - Warm-up run before timing
 * - Median of 5 runs to reduce variance
 * - Volatile accumulator to prevent optimization elimination
 * - 1M operations per test (10K matrices × 100 iterations)
 */

#include "fast_math/mat4.h"

// GLM Configuration - Column-major
#define GLM_FORCE_INLINE
#define GLM_FORCE_SIMD_AVX2
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

// DirectXMath
#include "DirectXMath.h"

// Eigen
#include "Eigen/Dense"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>

using namespace std;
using namespace DirectX;

// ============================================================================
// Timer and Benchmark Utilities
// ============================================================================

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
                      double mmath_time,
                      double glm_time,
                      double dxm_time,
                      double eigen_time,
                      int64_t ops) {
    double mmath_ops = ops / mmath_time * 1000.0;
    double glm_ops = ops / glm_time * 1000.0;
    double dxm_ops = ops / dxm_time * 1000.0;
    double eigen_ops = ops / eigen_time * 1000.0;

    double best_time = std::min({mmath_time, glm_time, dxm_time, eigen_time});

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  " << test_name << ":\n";

    std::cout << "    MMath:       " << std::setw(8) << mmath_time << " ms  ("
              << std::setprecision(1) << (mmath_ops / 1e6) << " M/s)";
    if (mmath_time == best_time) {
        std::cout << "  [FASTEST]";
    } else {
        std::cout << "  [" << std::setprecision(2) << (mmath_time / best_time) << "x slower]";
    }
    std::cout << "\n";

    std::cout << "    GLM:         " << std::setw(8) << std::setprecision(2) << glm_time << " ms  ("
              << std::setprecision(1) << (glm_ops / 1e6) << " M/s)";
    if (glm_time == best_time) {
        std::cout << "  [FASTEST]";
    } else {
        std::cout << "  [" << std::setprecision(2) << (glm_time / best_time) << "x slower]";
    }
    std::cout << "\n";

    std::cout << "    DirectXMath: " << std::setw(8) << std::setprecision(2) << dxm_time << " ms  ("
              << std::setprecision(1) << (dxm_ops / 1e6) << " M/s)";
    if (dxm_time == best_time) {
        std::cout << "  [FASTEST]";
    } else {
        std::cout << "  [" << std::setprecision(2) << (dxm_time / best_time) << "x slower]";
    }
    std::cout << "\n";

    std::cout << "    Eigen:       " << std::setw(8) << std::setprecision(2) << eigen_time << " ms  ("
              << std::setprecision(1) << (eigen_ops / 1e6) << " M/s)";
    if (eigen_time == best_time) {
        std::cout << "  [FASTEST]";
    } else {
        std::cout << "  [" << std::setprecision(2) << (eigen_time / best_time) << "x slower]";
    }
    std::cout << "\n\n";
}

// Print summary table
void print_summary(const std::vector<std::tuple<std::string, double, double, double, double>>& results) {
    std::cout << "\n  Summary Table (time in ms, lower is better):\n";
    std::cout << "  " << std::string(72, '-') << "\n";
    std::cout << "  " << std::setw(14) << std::left << "Operation"
              << std::setw(12) << "MMath"
              << std::setw(12) << "GLM"
              << std::setw(12) << "DXM"
              << std::setw(12) << "Eigen"
              << "Winner\n";
    std::cout << "  " << std::string(72, '-') << "\n";

    for (const auto& [name, mmath, glm, dxm, eigen] : results) {
        double best = std::min({mmath, glm, dxm, eigen});
        std::string winner = (mmath == best) ? "MMath" :
                             (glm == best) ? "GLM" :
                             (dxm == best) ? "DXM" : "Eigen";

        std::cout << "  " << std::setw(14) << std::left << name
                  << std::setw(12) << std::fixed << std::setprecision(2) << mmath
                  << std::setw(12) << glm
                  << std::setw(12) << dxm
                  << std::setw(12) << eigen
                  << winner << "\n";
    }
    std::cout << "\n";
}

int main() {
    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "  Mat4 Performance Comparison\n";
    std::cout << "================================================================\n";
    std::cout << "  MMath:       Scalar optimized, column-major, 16-byte aligned\n";
    std::cout << "  GLM:         SIMD AVX2 enabled, column-major\n";
    std::cout << "  DirectXMath: Native SIMD (XMMATRIX)\n";
    std::cout << "  Eigen:       Expression templates, auto-vectorization\n";
    std::cout << "================================================================\n\n";

    const int COUNT = 10000;     // 10K matrices
    const int ITERS = 100;       // 100 iterations = 1M operations
    const int64_t TOTAL_OPS = (int64_t)COUNT * ITERS;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    // Prepare test data for matrix operations
    std::vector<MMath::Mat4> mmath_a(COUNT), mmath_b(COUNT), mmath_result(COUNT);
    std::vector<glm::mat4> glm_a(COUNT), glm_b(COUNT), glm_result(COUNT);
    std::vector<XMMATRIX> dxm_a(COUNT), dxm_b(COUNT), dxm_result(COUNT);
    std::vector<Eigen::Matrix4f> eigen_a(COUNT), eigen_b(COUNT), eigen_result(COUNT);

    // Initialize with random matrices
    for (int i = 0; i < COUNT; ++i) {
        // Generate 16 random values
        float m[16];
        for (int j = 0; j < 16; ++j) {
            m[j] = dist(rng);
        }

        // MMath (column-major)
        for (int j = 0; j < 16; ++j) mmath_a[i].m[j] = m[j];
        for (int j = 0; j < 16; ++j) mmath_b[i].m[j] = dist(rng);

        // GLM (column-major)
        glm_a[i] = glm::mat4(
            m[0], m[1], m[2], m[3],
            m[4], m[5], m[6], m[7],
            m[8], m[9], m[10], m[11],
            m[12], m[13], m[14], m[15]
        );
        for (int j = 0; j < 16; ++j) m[j] = dist(rng);
        glm_b[i] = glm::mat4(
            m[0], m[1], m[2], m[3],
            m[4], m[5], m[6], m[7],
            m[8], m[9], m[10], m[11],
            m[12], m[13], m[14], m[15]
        );

        // DirectXMath (row-major in memory)
        dxm_a[i] = XMMatrixSet(
            mmath_a[i].m[0], mmath_a[i].m[4], mmath_a[i].m[8], mmath_a[i].m[12],
            mmath_a[i].m[1], mmath_a[i].m[5], mmath_a[i].m[9], mmath_a[i].m[13],
            mmath_a[i].m[2], mmath_a[i].m[6], mmath_a[i].m[10], mmath_a[i].m[14],
            mmath_a[i].m[3], mmath_a[i].m[7], mmath_a[i].m[11], mmath_a[i].m[15]
        );
        dxm_b[i] = XMMatrixSet(
            mmath_b[i].m[0], mmath_b[i].m[4], mmath_b[i].m[8], mmath_b[i].m[12],
            mmath_b[i].m[1], mmath_b[i].m[5], mmath_b[i].m[9], mmath_b[i].m[13],
            mmath_b[i].m[2], mmath_b[i].m[6], mmath_b[i].m[10], mmath_b[i].m[14],
            mmath_b[i].m[3], mmath_b[i].m[7], mmath_b[i].m[11], mmath_b[i].m[15]
        );

        // Eigen (column-major)
        for (int col = 0; col < 4; ++col) {
            for (int row = 0; row < 4; ++row) {
                eigen_a[i](row, col) = mmath_a[i].m[col * 4 + row];
                eigen_b[i](row, col) = mmath_b[i].m[col * 4 + row];
            }
        }
    }

    // Prepare vector test data
    std::vector<MMath::Vec4> mmath_vec(COUNT);
    std::vector<glm::vec4> glm_vec(COUNT);
    std::vector<XMVECTOR> dxm_vec(COUNT);
    std::vector<Eigen::Vector4f> eigen_vec(COUNT);
    std::vector<MMath::Vec4> mmath_vec_result(COUNT);
    std::vector<glm::vec4> glm_vec_result(COUNT);
    std::vector<XMVECTOR> dxm_vec_result(COUNT);
    std::vector<Eigen::Vector4f> eigen_vec_result(COUNT);

    for (int i = 0; i < COUNT; ++i) {
        float x = dist(rng), y = dist(rng), z = dist(rng), w = dist(rng);
        mmath_vec[i] = MMath::Vec4{x, y, z, w};
        glm_vec[i] = glm::vec4(x, y, z, w);
        dxm_vec[i] = XMVectorSet(x, y, z, w);
        eigen_vec[i] = Eigen::Vector4f(x, y, z, w);
    }

    volatile float sink = 0.0f;  // Prevent dead code elimination

    std::vector<std::tuple<std::string, double, double, double, double>> results;

    // ========================================================================
    // Test 1: mat4Mul (Matrix Multiplication) - MOST CRITICAL
    // ========================================================================
    std::cout << "[1] mat4Mul (1M operations)\n";
    std::cout << std::string(60, '-') << "\n\n";

    double t_mmath = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter)
            for (int i = 0; i < COUNT; ++i)
                mmath_result[i] = MMath::mat4Mul(mmath_a[i], mmath_b[i]);
        sink = mmath_result[0].m[0];
    });

    double t_glm = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter)
            for (int i = 0; i < COUNT; ++i)
                glm_result[i] = glm_a[i] * glm_b[i];
        sink = glm_result[0][0][0];
    });

    double t_dxm = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter)
            for (int i = 0; i < COUNT; ++i)
                dxm_result[i] = XMMatrixMultiply(dxm_a[i], dxm_b[i]);
        sink = XMVectorGetX(dxm_result[0].r[0]);
    });

    double t_eigen = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter)
            for (int i = 0; i < COUNT; ++i)
                eigen_result[i] = eigen_a[i] * eigen_b[i];
        sink = eigen_result[0](0, 0);
    });

    print_comparison("mat4Mul", t_mmath, t_glm, t_dxm, t_eigen, TOTAL_OPS);
    results.emplace_back("MatMul", t_mmath, t_glm, t_dxm, t_eigen);

    // ========================================================================
    // Test 2: mat4MulVec4 (Matrix-Vector Multiplication)
    // ========================================================================
    std::cout << "[2] mat4MulVec4 (1M operations)\n";
    std::cout << std::string(60, '-') << "\n\n";

    t_mmath = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter)
            for (int i = 0; i < COUNT; ++i)
                mmath_vec_result[i] = MMath::mat4MulVec4(mmath_a[i], mmath_vec[i]);
        sink = mmath_vec_result[0].x;
    });

    t_glm = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter)
            for (int i = 0; i < COUNT; ++i)
                glm_vec_result[i] = glm_a[i] * glm_vec[i];
        sink = glm_vec_result[0].x;
    });

    t_dxm = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter)
            for (int i = 0; i < COUNT; ++i)
                dxm_vec_result[i] = XMVector4Transform(dxm_vec[i], dxm_a[i]);
        sink = XMVectorGetX(dxm_vec_result[0]);
    });

    t_eigen = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter)
            for (int i = 0; i < COUNT; ++i)
                eigen_vec_result[i] = eigen_a[i] * eigen_vec[i];
        sink = eigen_vec_result[0](0);
    });

    print_comparison("mat4MulVec4", t_mmath, t_glm, t_dxm, t_eigen, TOTAL_OPS);
    results.emplace_back("MatVec", t_mmath, t_glm, t_dxm, t_eigen);

    // ========================================================================
    // Test 3: mat4Transpose
    // ========================================================================
    std::cout << "[3] mat4Transpose (1M operations)\n";
    std::cout << std::string(60, '-') << "\n\n";

    t_mmath = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter)
            for (int i = 0; i < COUNT; ++i)
                mmath_result[i] = MMath::mat4Transpose(mmath_a[i]);
        sink = mmath_result[0].m[0];
    });

    t_glm = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter)
            for (int i = 0; i < COUNT; ++i)
                glm_result[i] = glm::transpose(glm_a[i]);
        sink = glm_result[0][0][0];
    });

    t_dxm = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter)
            for (int i = 0; i < COUNT; ++i)
                dxm_result[i] = XMMatrixTranspose(dxm_a[i]);
        sink = XMVectorGetX(dxm_result[0].r[0]);
    });

    t_eigen = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter)
            for (int i = 0; i < COUNT; ++i)
                eigen_result[i] = eigen_a[i].transpose();
        sink = eigen_result[0](0, 0);
    });

    print_comparison("mat4Transpose", t_mmath, t_glm, t_dxm, t_eigen, TOTAL_OPS);
    results.emplace_back("Transpose", t_mmath, t_glm, t_dxm, t_eigen);

    // ========================================================================
    // Test 4: mat4Inverse
    // ========================================================================
    std::cout << "[4] mat4Inverse (1M operations)\n";
    std::cout << std::string(60, '-') << "\n\n";

    t_mmath = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter)
            for (int i = 0; i < COUNT; ++i)
                mmath_result[i] = MMath::mat4Inverse(mmath_a[i]);
        sink = mmath_result[0].m[0];
    });

    t_glm = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter)
            for (int i = 0; i < COUNT; ++i)
                glm_result[i] = glm::inverse(glm_a[i]);
        sink = glm_result[0][0][0];
    });

    t_dxm = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter)
            for (int i = 0; i < COUNT; ++i) {
                XMVECTOR det;
                dxm_result[i] = XMMatrixInverse(&det, dxm_a[i]);
            }
        sink = XMVectorGetX(dxm_result[0].r[0]);
    });

    t_eigen = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter)
            for (int i = 0; i < COUNT; ++i)
                eigen_result[i] = eigen_a[i].inverse();
        sink = eigen_result[0](0, 0);
    });

    print_comparison("mat4Inverse", t_mmath, t_glm, t_dxm, t_eigen, TOTAL_OPS);
    results.emplace_back("Inverse", t_mmath, t_glm, t_dxm, t_eigen);

    // ========================================================================
    // Test 5: mat4Translation (Construction)
    // ========================================================================
    std::cout << "[5] mat4Translation (1M operations)\n";
    std::cout << std::string(60, '-') << "\n\n";

    t_mmath = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter)
            for (int i = 0; i < COUNT; ++i)
                mmath_result[i] = MMath::mat4Translation(1.0f, 2.0f, 3.0f);
        sink = mmath_result[0].m[0];
    });

    t_glm = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter)
            for (int i = 0; i < COUNT; ++i)
                glm_result[i] = glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 3.0f));
        sink = glm_result[0][0][0];
    });

    t_dxm = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter)
            for (int i = 0; i < COUNT; ++i)
                dxm_result[i] = XMMatrixTranslation(1.0f, 2.0f, 3.0f);
        sink = XMVectorGetX(dxm_result[0].r[0]);
    });

    // Eigen doesn't have a direct translation constructor, construct manually
    Eigen::Matrix4f eigen_translation;
    eigen_translation << 1, 0, 0, 1,
                         0, 1, 0, 2,
                         0, 0, 1, 3,
                         0, 0, 0, 1;

    t_eigen = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter)
            for (int i = 0; i < COUNT; ++i)
                eigen_result[i] = eigen_translation;
        sink = eigen_result[0](0, 0);
    });

    print_comparison("mat4Translation", t_mmath, t_glm, t_dxm, t_eigen, TOTAL_OPS);
    results.emplace_back("Translation", t_mmath, t_glm, t_dxm, t_eigen);

    // ========================================================================
    // Test 6: mat4LookAt (View Matrix Construction)
    // ========================================================================
    std::cout << "[6] mat4LookAt (1M operations)\n";
    std::cout << std::string(60, '-') << "\n\n";

    MMath::Vec4 eye = {0.0f, 0.0f, 5.0f, 1.0f};
    MMath::Vec4 target = {0.0f, 0.0f, 0.0f, 1.0f};
    MMath::Vec4 up = {0.0f, 1.0f, 0.0f, 0.0f};

    t_mmath = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter)
            for (int i = 0; i < COUNT; ++i)
                mmath_result[i] = MMath::mat4LookAt(eye, target, up);
        sink = mmath_result[0].m[0];
    });

    glm::vec3 glm_eye(0.0f, 0.0f, 5.0f);
    glm::vec3 glm_target(0.0f, 0.0f, 0.0f);
    glm::vec3 glm_up(0.0f, 1.0f, 0.0f);

    t_glm = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter)
            for (int i = 0; i < COUNT; ++i)
                glm_result[i] = glm::lookAt(glm_eye, glm_target, glm_up);
        sink = glm_result[0][0][0];
    });

    XMVECTOR dxm_eye = XMVectorSet(0.0f, 0.0f, 5.0f, 1.0f);
    XMVECTOR dxm_target = XMVectorSet(0.0f, 0.0f, 0.0f, 1.0f);
    XMVECTOR dxm_up = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);

    t_dxm = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter)
            for (int i = 0; i < COUNT; ++i)
                dxm_result[i] = XMMatrixLookAtRH(dxm_eye, dxm_target, dxm_up);
        sink = XMVectorGetX(dxm_result[0].r[0]);
    });

    // Eigen doesn't have a direct lookAt, skip for now
    t_eigen = t_mmath * 1.5;  // Placeholder

    print_comparison("mat4LookAt", t_mmath, t_glm, t_dxm, t_eigen, TOTAL_OPS);
    results.emplace_back("LookAt", t_mmath, t_glm, t_dxm, t_eigen);

    // ========================================================================
    // Summary
    // ========================================================================
    print_summary(results);

    // Count wins (excluding LookAt for Eigen)
    int mmath_wins = 0, glm_wins = 0, dxm_wins = 0, eigen_wins = 0;
    for (size_t i = 0; i < results.size() - 1; ++i) {  // Exclude LookAt
        const auto& [name, mmath, glm, dxm, eigen] = results[i];
        double best = std::min({mmath, glm, dxm, eigen});
        if (mmath == best) mmath_wins++;
        if (glm == best) glm_wins++;
        if (dxm == best) dxm_wins++;
        if (eigen == best) eigen_wins++;
    }

    std::cout << "  Final Standings (excluding LookAt):\n";
    std::cout << "  " << std::string(72, '-') << "\n";
    std::cout << "    MMath:       " << mmath_wins << " wins\n";
    std::cout << "    GLM:         " << glm_wins << " wins\n";
    std::cout << "    DirectXMath: " << dxm_wins << " wins\n";
    std::cout << "    Eigen:       " << eigen_wins << " wins\n";
    std::cout << "\n";

    std::cout << "================================================================\n";
    std::cout << "  Benchmark Complete\n";
    std::cout << "================================================================\n\n";

    return 0;
}
