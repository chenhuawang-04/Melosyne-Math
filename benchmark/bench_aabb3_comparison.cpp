/**
 * @file bench_aabb3_comparison.cpp
 * @brief Detailed comparison benchmark: MMath vs GLM vs DirectXMath
 *
 * Methodology:
 * - Same test data for all libraries
 * - Same compiler flags (-O3 -march=native -mavx2 -mfma)
 * - Median of 5 runs to reduce variance
 * - 10M operations (10K AABBs × 1000 iterations)
 *
 * Tested Operations:
 * 1. Union/Merge: Combine two AABBs
 * 2. Overlap/Intersects: Test if two AABBs overlap
 * 3. Contains: Test if AABB contains a point
 * 4. Transform: Apply 4x4 matrix transformation
 */

#include "fast_math/aabb3.h"

// GLM Configuration
#define GLM_FORCE_INLINE
#define GLM_FORCE_SIMD_AVX2
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/glm.hpp"
#include "glm/gtx/component_wise.hpp"
#include "glm/gtc/matrix_transform.hpp"

// DirectXMath
#include "DirectXMath.h"
#include "DirectXCollision.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <random>
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
                      int64_t ops) {
    double best_time = std::min({mmath_time, glm_time, dxm_time});

    std::cout << "  " << test_name << ":\n";

    std::cout << "    MMath:       " << std::setw(8) << std::fixed << std::setprecision(2) << mmath_time << " ms  ("
              << std::setprecision(1) << (ops / mmath_time / 1e6) << " M/s)";
    if (mmath_time == best_time) {
        std::cout << "  [FASTEST]";
    } else {
        std::cout << "  [" << std::setprecision(2) << (mmath_time / best_time) << "x slower]";
    }
    std::cout << "\n";

    std::cout << "    GLM:         " << std::setw(8) << std::fixed << std::setprecision(2) << glm_time << " ms  ("
              << std::setprecision(1) << (ops / glm_time / 1e6) << " M/s)";
    if (glm_time == best_time) {
        std::cout << "  [FASTEST]";
    } else {
        std::cout << "  [" << std::setprecision(2) << (glm_time / best_time) << "x slower]";
    }
    std::cout << "\n";

    std::cout << "    DirectXMath: " << std::setw(8) << std::fixed << std::setprecision(2) << dxm_time << " ms  ("
              << std::setprecision(1) << (ops / dxm_time / 1e6) << " M/s)";
    if (dxm_time == best_time) {
        std::cout << "  [FASTEST]";
    } else {
        std::cout << "  [" << std::setprecision(2) << (dxm_time / best_time) << "x slower]";
    }
    std::cout << "\n\n";
}

// ============================================================================
// GLM AABB Wrapper (GLM has no built-in AABB, implement manually)
// ============================================================================

struct GLM_AABB {
    glm::vec3 min;
    glm::vec3 max;
};

GLM_AABB glm_aabb_from_min_max(glm::vec3 min, glm::vec3 max) {
    return GLM_AABB{min, max};
}

GLM_AABB glm_aabb_union(const GLM_AABB& a, const GLM_AABB& b) {
    return GLM_AABB{
        glm::min(a.min, b.min),
        glm::max(a.max, b.max)
    };
}

bool glm_aabb_overlap(const GLM_AABB& a, const GLM_AABB& b) {
    return glm::all(glm::lessThanEqual(a.min, b.max)) &&
           glm::all(glm::greaterThanEqual(a.max, b.min));
}

bool glm_aabb_contains(const GLM_AABB& aabb, glm::vec3 point) {
    return glm::all(glm::greaterThanEqual(point, aabb.min)) &&
           glm::all(glm::lessThanEqual(point, aabb.max));
}

GLM_AABB glm_aabb_transform(const GLM_AABB& aabb, const glm::mat4& mat) {
    // Transform all 8 corners
    glm::vec3 corners[8] = {
        glm::vec3(mat * glm::vec4(aabb.min.x, aabb.min.y, aabb.min.z, 1.0f)),
        glm::vec3(mat * glm::vec4(aabb.max.x, aabb.min.y, aabb.min.z, 1.0f)),
        glm::vec3(mat * glm::vec4(aabb.min.x, aabb.max.y, aabb.min.z, 1.0f)),
        glm::vec3(mat * glm::vec4(aabb.max.x, aabb.max.y, aabb.min.z, 1.0f)),
        glm::vec3(mat * glm::vec4(aabb.min.x, aabb.min.y, aabb.max.z, 1.0f)),
        glm::vec3(mat * glm::vec4(aabb.max.x, aabb.min.y, aabb.max.z, 1.0f)),
        glm::vec3(mat * glm::vec4(aabb.min.x, aabb.max.y, aabb.max.z, 1.0f)),
        glm::vec3(mat * glm::vec4(aabb.max.x, aabb.max.y, aabb.max.z, 1.0f))
    };

    glm::vec3 new_min = corners[0];
    glm::vec3 new_max = corners[0];
    for (int i = 1; i < 8; ++i) {
        new_min = glm::min(new_min, corners[i]);
        new_max = glm::max(new_max, corners[i]);
    }

    return GLM_AABB{new_min, new_max};
}

// ============================================================================
// Main Benchmark
// ============================================================================

int main() {
    cout << "\n";
    cout << "================================================================\n";
    cout << "  Aabb3 Detailed Comparison Benchmark\n";
    cout << "================================================================\n";
    cout << "  MMath:       SIMD optimized (SSE4.1, negative max storage)\n";
    cout << "  GLM:         AVX2 enabled, manual AABB implementation\n";
    cout << "  DirectXMath: Native BoundingBox (Center+Extents)\n";
    cout << "================================================================\n";
    cout << "  Compiler: Clang -O3 -march=native -mavx2 -mfma\n";
    cout << "  Operations: 10M (10K AABBs × 1000 iterations)\n";
    cout << "  Runs: Median of 5 runs\n";
    cout << "================================================================\n\n";

    const int COUNT = 10000;
    const int ITERS = 1000;
    const int64_t TOTAL_OPS = (int64_t)COUNT * ITERS;

    // Setup random test data (same for all libraries)
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    std::vector<float> test_data_min_x(COUNT), test_data_min_y(COUNT), test_data_min_z(COUNT);
    std::vector<float> test_data_max_x(COUNT), test_data_max_y(COUNT), test_data_max_z(COUNT);
    std::vector<float> test_data_min_x2(COUNT), test_data_min_y2(COUNT), test_data_min_z2(COUNT);
    std::vector<float> test_data_max_x2(COUNT), test_data_max_y2(COUNT), test_data_max_z2(COUNT);
    std::vector<float> test_point_x(COUNT), test_point_y(COUNT), test_point_z(COUNT);

    for (int i = 0; i < COUNT; ++i) {
        test_data_min_x[i] = dist(rng);
        test_data_min_y[i] = dist(rng);
        test_data_min_z[i] = dist(rng);
        test_data_max_x[i] = test_data_min_x[i] + std::abs(dist(rng));
        test_data_max_y[i] = test_data_min_y[i] + std::abs(dist(rng));
        test_data_max_z[i] = test_data_min_z[i] + std::abs(dist(rng));

        test_data_min_x2[i] = dist(rng);
        test_data_min_y2[i] = dist(rng);
        test_data_min_z2[i] = dist(rng);
        test_data_max_x2[i] = test_data_min_x2[i] + std::abs(dist(rng));
        test_data_max_y2[i] = test_data_min_y2[i] + std::abs(dist(rng));
        test_data_max_z2[i] = test_data_min_z2[i] + std::abs(dist(rng));

        test_point_x[i] = dist(rng);
        test_point_y[i] = dist(rng);
        test_point_z[i] = dist(rng);
    }

    // Prepare test data structures for each library
    std::vector<MMath::Aabb3> mmath_a(COUNT), mmath_b(COUNT), mmath_result(COUNT);
    std::vector<GLM_AABB> glm_a(COUNT), glm_b(COUNT), glm_result(COUNT);
    std::vector<BoundingBox> dxm_a(COUNT), dxm_b(COUNT), dxm_result(COUNT);
    std::vector<MMath::Vec3> mmath_points(COUNT);
    std::vector<glm::vec3> glm_points(COUNT);
    std::vector<XMFLOAT3> dxm_points(COUNT);

    for (int i = 0; i < COUNT; ++i) {
        // MMath
        mmath_a[i] = MMath::aabb3FromMinMax(
            {test_data_min_x[i], test_data_min_y[i], test_data_min_z[i]},
            {test_data_max_x[i], test_data_max_y[i], test_data_max_z[i]}
        );
        mmath_b[i] = MMath::aabb3FromMinMax(
            {test_data_min_x2[i], test_data_min_y2[i], test_data_min_z2[i]},
            {test_data_max_x2[i], test_data_max_y2[i], test_data_max_z2[i]}
        );
        mmath_points[i] = {test_point_x[i], test_point_y[i], test_point_z[i]};

        // GLM
        glm_a[i] = glm_aabb_from_min_max(
            glm::vec3(test_data_min_x[i], test_data_min_y[i], test_data_min_z[i]),
            glm::vec3(test_data_max_x[i], test_data_max_y[i], test_data_max_z[i])
        );
        glm_b[i] = glm_aabb_from_min_max(
            glm::vec3(test_data_min_x2[i], test_data_min_y2[i], test_data_min_z2[i]),
            glm::vec3(test_data_max_x2[i], test_data_max_y2[i], test_data_max_z2[i])
        );
        glm_points[i] = glm::vec3(test_point_x[i], test_point_y[i], test_point_z[i]);

        // DirectXMath (uses Center+Extents representation)
        glm::vec3 center_a = (glm_a[i].min + glm_a[i].max) * 0.5f;
        glm::vec3 extents_a = (glm_a[i].max - glm_a[i].min) * 0.5f;
        dxm_a[i].Center = XMFLOAT3(center_a.x, center_a.y, center_a.z);
        dxm_a[i].Extents = XMFLOAT3(extents_a.x, extents_a.y, extents_a.z);

        glm::vec3 center_b = (glm_b[i].min + glm_b[i].max) * 0.5f;
        glm::vec3 extents_b = (glm_b[i].max - glm_b[i].min) * 0.5f;
        dxm_b[i].Center = XMFLOAT3(center_b.x, center_b.y, center_b.z);
        dxm_b[i].Extents = XMFLOAT3(extents_b.x, extents_b.y, extents_b.z);

        dxm_points[i] = XMFLOAT3(test_point_x[i], test_point_y[i], test_point_z[i]);
    }

    std::vector<bool> bool_results(COUNT);
    volatile float sink = 0.0f;

    // ========================================================================
    // Benchmark 1: Union/Merge
    // ========================================================================

    cout << "[1] Union/Merge (combine two AABBs)\n";
    cout << "------------------------------------------------------------\n\n";

    double mmath_union_time = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter) {
            for (int i = 0; i < COUNT; ++i) {
                mmath_result[i] = MMath::aabb3Union(mmath_a[i], mmath_b[i]);
            }
        }
        sink = mmath_result[0].minx;
    });

    double glm_union_time = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter) {
            for (int i = 0; i < COUNT; ++i) {
                glm_result[i] = glm_aabb_union(glm_a[i], glm_b[i]);
            }
        }
        sink = glm_result[0].min.x;
    });

    double dxm_union_time = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter) {
            for (int i = 0; i < COUNT; ++i) {
                BoundingBox::CreateMerged(dxm_result[i], dxm_a[i], dxm_b[i]);
            }
        }
        sink = dxm_result[0].Center.x;
    });

    print_comparison("aabb3Union", mmath_union_time, glm_union_time, dxm_union_time, TOTAL_OPS);

    // ========================================================================
    // Benchmark 2: Overlap/Intersects
    // ========================================================================

    cout << "[2] Overlap/Intersects (test if two AABBs overlap)\n";
    cout << "------------------------------------------------------------\n\n";

    double mmath_overlap_time = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter) {
            for (int i = 0; i < COUNT; ++i) {
                bool_results[i] = MMath::aabb3Overlap(mmath_a[i], mmath_b[i]);
            }
        }
        sink = bool_results[0] ? 1.0f : 0.0f;
    });

    double glm_overlap_time = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter) {
            for (int i = 0; i < COUNT; ++i) {
                bool_results[i] = glm_aabb_overlap(glm_a[i], glm_b[i]);
            }
        }
        sink = bool_results[0] ? 1.0f : 0.0f;
    });

    double dxm_overlap_time = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter) {
            for (int i = 0; i < COUNT; ++i) {
                bool_results[i] = dxm_a[i].Intersects(dxm_b[i]);
            }
        }
        sink = bool_results[0] ? 1.0f : 0.0f;
    });

    print_comparison("aabb3Overlap", mmath_overlap_time, glm_overlap_time, dxm_overlap_time, TOTAL_OPS);

    // ========================================================================
    // Benchmark 3: Contains Point
    // ========================================================================

    cout << "[3] Contains Point (test if AABB contains a point)\n";
    cout << "------------------------------------------------------------\n\n";

    double mmath_contains_time = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter) {
            for (int i = 0; i < COUNT; ++i) {
                bool_results[i] = MMath::aabb3Contains(mmath_a[i], mmath_points[i]);
            }
        }
        sink = bool_results[0] ? 1.0f : 0.0f;
    });

    double glm_contains_time = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter) {
            for (int i = 0; i < COUNT; ++i) {
                bool_results[i] = glm_aabb_contains(glm_a[i], glm_points[i]);
            }
        }
        sink = bool_results[0] ? 1.0f : 0.0f;
    });

    double dxm_contains_time = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter) {
            for (int i = 0; i < COUNT; ++i) {
                XMVECTOR pt = XMLoadFloat3(&dxm_points[i]);
                bool_results[i] = dxm_a[i].Contains(pt) != DISJOINT;
            }
        }
        sink = bool_results[0] ? 1.0f : 0.0f;
    });

    print_comparison("aabb3Contains", mmath_contains_time, glm_contains_time, dxm_contains_time, TOTAL_OPS);

    // ========================================================================
    // Benchmark 4: Transform (expensive operation, fewer iterations)
    // ========================================================================

    cout << "[4] Transform (apply 4x4 matrix, 1M ops)\n";
    cout << "------------------------------------------------------------\n\n";

    const int TRANSFORM_ITERS = 100;
    const int64_t TRANSFORM_OPS = (int64_t)COUNT * TRANSFORM_ITERS;

    MMath::Mat4 mmath_mat = MMath::mat4Translation(10.0f, 20.0f, 30.0f);
    glm::mat4 glm_mat = glm::translate(glm::mat4(1.0f), glm::vec3(10.0f, 20.0f, 30.0f));
    XMMATRIX dxm_mat = XMMatrixTranslation(10.0f, 20.0f, 30.0f);

    double mmath_transform_time = benchmark_median([&]() {
        for (int iter = 0; iter < TRANSFORM_ITERS; ++iter) {
            for (int i = 0; i < COUNT; ++i) {
                mmath_result[i] = MMath::aabb3Transform(mmath_a[i], mmath_mat);
            }
        }
        sink = mmath_result[0].minx;
    });

    double glm_transform_time = benchmark_median([&]() {
        for (int iter = 0; iter < TRANSFORM_ITERS; ++iter) {
            for (int i = 0; i < COUNT; ++i) {
                glm_result[i] = glm_aabb_transform(glm_a[i], glm_mat);
            }
        }
        sink = glm_result[0].min.x;
    });

    double dxm_transform_time = benchmark_median([&]() {
        for (int iter = 0; iter < TRANSFORM_ITERS; ++iter) {
            for (int i = 0; i < COUNT; ++i) {
                dxm_a[i].Transform(dxm_result[i], dxm_mat);
            }
        }
        sink = dxm_result[0].Center.x;
    });

    print_comparison("aabb3Transform", mmath_transform_time, glm_transform_time, dxm_transform_time, TRANSFORM_OPS);

    // ========================================================================
    // Summary
    // ========================================================================

    cout << "================================================================\n";
    cout << "  Summary Table (time in ms, lower is better)\n";
    cout << "================================================================\n";
    cout << "  Operation       MMath       GLM         DXM         Winner\n";
    cout << "  --------------  ----------  ----------  ----------  ----------\n";

    auto print_row = [](const string& name, double mmath, double glm_time, double dxm) {
        double best = std::min({mmath, glm_time, dxm});
        string winner = (mmath == best) ? "MMath" : (glm_time == best) ? "GLM" : "DXM";
        cout << "  " << std::setw(14) << std::left << name
             << std::setw(12) << std::fixed << std::setprecision(2) << mmath
             << std::setw(12) << glm_time
             << std::setw(12) << dxm
             << winner << "\n";
    };

    print_row("Union", mmath_union_time, glm_union_time, dxm_union_time);
    print_row("Overlap", mmath_overlap_time, glm_overlap_time, dxm_overlap_time);
    print_row("Contains", mmath_contains_time, glm_contains_time, dxm_contains_time);
    print_row("Transform", mmath_transform_time, glm_transform_time, dxm_transform_time);

    cout << "\n  Final Standings:\n";
    int mmath_wins = 0, glm_wins = 0, dxm_wins = 0;

    auto count_winner = [&](double mmath, double glm_time, double dxm) {
        double best = std::min({mmath, glm_time, dxm});
        if (mmath == best) mmath_wins++;
        else if (glm_time == best) glm_wins++;
        else dxm_wins++;
    };

    count_winner(mmath_union_time, glm_union_time, dxm_union_time);
    count_winner(mmath_overlap_time, glm_overlap_time, dxm_overlap_time);
    count_winner(mmath_contains_time, glm_contains_time, dxm_contains_time);
    count_winner(mmath_transform_time, glm_transform_time, dxm_transform_time);

    cout << "    MMath:       " << mmath_wins << " wins\n";
    cout << "    GLM:         " << glm_wins << " wins\n";
    cout << "    DirectXMath: " << dxm_wins << " wins\n";
    cout << "\n================================================================\n\n";

    cout << "Key Insights:\n";
    cout << "  - MMath uses negative max storage for single-instruction union\n";
    cout << "  - DirectXMath uses Center+Extents (different representation)\n";
    cout << "  - GLM manual AABB implementation (no built-in support)\n";
    cout << "  - All implementations use SIMD optimization strategies\n\n";

    return 0;
}
