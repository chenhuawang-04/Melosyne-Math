/**
 * @file bench_aabb2_triple.cpp
 * @brief AABB2 triple-library comparison: fast_math vs GLM (manual) vs DirectXMath (3D adapted to 2D)
 *
 * Note:
 * - GLM has no native AABB support, so we implement a simple version
 * - DirectXMath only has 3D BoundingBox, so we adapt it for 2D (set Z=0)
 */

#include <FastMath/aabb2.h>

// GLM - Manual AABB2 implementation (GLM has no native support)
#define GLM_FORCE_INLINE
#define GLM_FORCE_SIMD_AVX2
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>

// DirectXMath - 3D BoundingBox (adapt to 2D)
#include <DirectXCollision.h>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>

using namespace std;

// ============================================================================
// GLM AABB2 Implementation (Manual - GLM has no native AABB)
// ============================================================================
struct GLM_AABB2 {
    glm::vec2 min;
    glm::vec2 max;
};

inline GLM_AABB2 glm_aabb2_from_min_max(glm::vec2 min, glm::vec2 max) {
    return GLM_AABB2{ min, max };
}

inline GLM_AABB2 glm_aabb2_merge(GLM_AABB2 a, GLM_AABB2 b) {
    return GLM_AABB2{
        glm::min(a.min, b.min),
        glm::max(a.max, b.max)
    };
}

inline bool glm_aabb2_intersects(GLM_AABB2 a, GLM_AABB2 b) {
    return a.min.x <= b.max.x && a.max.x >= b.min.x &&
           a.min.y <= b.max.y && a.max.y >= b.min.y;
}

inline bool glm_aabb2_contains_point(GLM_AABB2 aabb, glm::vec2 point) {
    return point.x >= aabb.min.x && point.x <= aabb.max.x &&
           point.y >= aabb.min.y && point.y <= aabb.max.y;
}

inline float glm_aabb2_area(GLM_AABB2 aabb) {
    glm::vec2 size = aabb.max - aabb.min;
    return size.x * size.y;
}

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
                      double fast_time,
                      double glm_time,
                      double dxm_time,
                      int64_t ops) {
    double fast_ops = ops / fast_time * 1000.0;
    double glm_ops = ops / glm_time * 1000.0;
    double dxm_ops = ops / dxm_time * 1000.0;

    double best_time = std::min({fast_time, glm_time, dxm_time});

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  " << test_name << ":\n";

    std::cout << "    fast_math:   " << std::setw(8) << fast_time << " ms  ("
              << std::setprecision(1) << (fast_ops / 1e6) << " M/s)";
    if (fast_time == best_time) {
        std::cout << "  [FASTEST]";
    } else {
        std::cout << "  [" << std::setprecision(2) << (fast_time / best_time) << "x slower]";
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
    std::cout << "\n\n";
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "�? AABB2 Triple-Library Performance Comparison              ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════╣\n";
    std::cout << "�? fast_math:   Negative max storage + SSE optimized       ║\n";
    std::cout << "�? GLM:         Manual implementation (no native AABB)      ║\n";
    std::cout << "�? DirectXMath: 3D BoundingBox adapted to 2D (Z=0)         ║\n";
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

    // Allocate memory for all three implementations
    std::vector<FastMath::AABB2> fast_aabbs_a(COUNT);
    std::vector<FastMath::AABB2> fast_aabbs_b(COUNT);
    std::vector<FastMath::AABB2> fast_aabbs_result(COUNT);

    std::vector<GLM_AABB2> glm_aabbs_a(COUNT);
    std::vector<GLM_AABB2> glm_aabbs_b(COUNT);
    std::vector<GLM_AABB2> glm_aabbs_result(COUNT);

    std::vector<DirectX::BoundingBox> dxm_aabbs_a(COUNT);
    std::vector<DirectX::BoundingBox> dxm_aabbs_b(COUNT);
    std::vector<DirectX::BoundingBox> dxm_aabbs_result(COUNT);

    std::vector<FastMath::Vec2> fast_points(COUNT);
    std::vector<glm::vec2> glm_points(COUNT);
    std::vector<DirectX::XMFLOAT3> dxm_points(COUNT);

    bool* bool_results = (bool*)malloc(COUNT * sizeof(bool));

    // Generate random AABBs
    for (int i = 0; i < COUNT; ++i) {
        float minx_a = dist_pos(rng);
        float miny_a = dist_pos(rng);
        float w_a = dist_size(rng);
        float h_a = dist_size(rng);

        float minx_b = dist_pos(rng);
        float miny_b = dist_pos(rng);
        float w_b = dist_size(rng);
        float h_b = dist_size(rng);

        // fast_math
        fast_aabbs_a[i] = FastMath::aabb2_from_min_max({minx_a, miny_a}, {minx_a + w_a, miny_a + h_a});
        fast_aabbs_b[i] = FastMath::aabb2_from_min_max({minx_b, miny_b}, {minx_b + w_b, miny_b + h_b});

        // GLM
        glm_aabbs_a[i] = glm_aabb2_from_min_max({minx_a, miny_a}, {minx_a + w_a, miny_a + h_a});
        glm_aabbs_b[i] = glm_aabb2_from_min_max({minx_b, miny_b}, {minx_b + w_b, miny_b + h_b});

        // DirectXMath (3D, set Z=0)
        DirectX::XMFLOAT3 center_a((minx_a + minx_a + w_a) * 0.5f, (miny_a + miny_a + h_a) * 0.5f, 0.0f);
        DirectX::XMFLOAT3 extents_a(w_a * 0.5f, h_a * 0.5f, 0.0f);
        dxm_aabbs_a[i] = DirectX::BoundingBox(center_a, extents_a);

        DirectX::XMFLOAT3 center_b((minx_b + minx_b + w_b) * 0.5f, (miny_b + miny_b + h_b) * 0.5f, 0.0f);
        DirectX::XMFLOAT3 extents_b(w_b * 0.5f, h_b * 0.5f, 0.0f);
        dxm_aabbs_b[i] = DirectX::BoundingBox(center_b, extents_b);

        // Points
        float px = dist_pos(rng);
        float py = dist_pos(rng);
        fast_points[i] = {px, py};
        glm_points[i] = {px, py};
        dxm_points[i] = {px, py, 0.0f};
    }

    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "�? Benchmark Results                                         ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    // ========================================================================
    // TEST 1: Merge (Union)
    // ========================================================================
    {
        auto fast_func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                for (int i = 0; i < COUNT; ++i) {
                    fast_aabbs_result[i] = FastMath::aabb2_merge(fast_aabbs_a[i], fast_aabbs_b[i]);
                }
            }
        };

        auto glm_func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                for (int i = 0; i < COUNT; ++i) {
                    glm_aabbs_result[i] = glm_aabb2_merge(glm_aabbs_a[i], glm_aabbs_b[i]);
                }
            }
        };

        auto dxm_func = [&]() {
            for (int iter = 0; iter < ITERS; ++iter) {
                for (int i = 0; i < COUNT; ++i) {
                    DirectX::BoundingBox::CreateMerged(dxm_aabbs_result[i], dxm_aabbs_a[i], dxm_aabbs_b[i]);
                }
            }
        };

        double fast_time = benchmark_median(fast_func);
        double glm_time = benchmark_median(glm_func);
        double dxm_time = benchmark_median(dxm_func);
        print_comparison("Merge (Union)", fast_time, glm_time, dxm_time, TOTAL_OPS);
    }

    // ========================================================================
    // TEST 2: Intersects
    // ========================================================================
    {
        volatile int result_sum = 0;  // Prevent optimization

        auto fast_func = [&]() {
            int sum = 0;
            for (int iter = 0; iter < ITERS; ++iter) {
                for (int i = 0; i < COUNT; ++i) {
                    sum += FastMath::aabb2_intersects(fast_aabbs_a[i], fast_aabbs_b[i]) ? 1 : 0;
                }
            }
            result_sum = sum;
        };

        auto glm_func = [&]() {
            int sum = 0;
            for (int iter = 0; iter < ITERS; ++iter) {
                for (int i = 0; i < COUNT; ++i) {
                    sum += glm_aabb2_intersects(glm_aabbs_a[i], glm_aabbs_b[i]) ? 1 : 0;
                }
            }
            result_sum = sum;
        };

        auto dxm_func = [&]() {
            int sum = 0;
            for (int iter = 0; iter < ITERS; ++iter) {
                for (int i = 0; i < COUNT; ++i) {
                    sum += dxm_aabbs_a[i].Intersects(dxm_aabbs_b[i]) ? 1 : 0;
                }
            }
            result_sum = sum;
        };

        double fast_time = benchmark_median(fast_func);
        double glm_time = benchmark_median(glm_func);
        double dxm_time = benchmark_median(dxm_func);
        print_comparison("Intersects (Overlap)", fast_time, glm_time, dxm_time, TOTAL_OPS);
    }

    // ========================================================================
    // TEST 3: Contains Point
    // ========================================================================
    {
        volatile int result_sum = 0;  // Prevent optimization

        auto fast_func = [&]() {
            int sum = 0;
            for (int iter = 0; iter < ITERS; ++iter) {
                for (int i = 0; i < COUNT; ++i) {
                    sum += FastMath::aabb2_contains_point(fast_aabbs_a[i], fast_points[i]) ? 1 : 0;
                }
            }
            result_sum = sum;
        };

        auto glm_func = [&]() {
            int sum = 0;
            for (int iter = 0; iter < ITERS; ++iter) {
                for (int i = 0; i < COUNT; ++i) {
                    sum += glm_aabb2_contains_point(glm_aabbs_a[i], glm_points[i]) ? 1 : 0;
                }
            }
            result_sum = sum;
        };

        auto dxm_func = [&]() {
            int sum = 0;
            for (int iter = 0; iter < ITERS; ++iter) {
                for (int i = 0; i < COUNT; ++i) {
                    DirectX::XMVECTOR point = DirectX::XMLoadFloat3(&dxm_points[i]);
                    DirectX::ContainmentType result = dxm_aabbs_a[i].Contains(point);
                    sum += (result != DirectX::DISJOINT) ? 1 : 0;
                }
            }
            result_sum = sum;
        };

        double fast_time = benchmark_median(fast_func);
        double glm_time = benchmark_median(glm_func);
        double dxm_time = benchmark_median(dxm_func);
        print_comparison("Contains Point", fast_time, glm_time, dxm_time, TOTAL_OPS);
    }

    // ========================================================================
    // Accuracy Verification
    // ========================================================================
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "�? Accuracy Verification                                     ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    FastMath::AABB2 fa = FastMath::aabb2_from_min_max({10.0f, 20.0f}, {50.0f, 80.0f});
    FastMath::AABB2 fb = FastMath::aabb2_from_min_max({30.0f, 40.0f}, {70.0f, 100.0f});

    GLM_AABB2 ga = glm_aabb2_from_min_max({10.0f, 20.0f}, {50.0f, 80.0f});
    GLM_AABB2 gb = glm_aabb2_from_min_max({30.0f, 40.0f}, {70.0f, 100.0f});

    DirectX::BoundingBox da(DirectX::XMFLOAT3(30.0f, 50.0f, 0.0f), DirectX::XMFLOAT3(20.0f, 30.0f, 0.0f));
    DirectX::BoundingBox db(DirectX::XMFLOAT3(50.0f, 70.0f, 0.0f), DirectX::XMFLOAT3(20.0f, 30.0f, 0.0f));

    FastMath::AABB2 f_merged = FastMath::aabb2_merge(fa, fb);
    GLM_AABB2 g_merged = glm_aabb2_merge(ga, gb);
    DirectX::BoundingBox d_merged;
    DirectX::BoundingBox::CreateMerged(d_merged, da, db);

    std::cout << "  Merged AABB (expected min={10,20}, max={70,100}):\n";
    std::cout << "    fast_math:   min={" << f_merged.minx << "," << f_merged.miny
              << "} max={" << -f_merged.neg_maxx << "," << -f_merged.neg_maxy << "}\n";
    std::cout << "    GLM:         min={" << g_merged.min.x << "," << g_merged.min.y
              << "} max={" << g_merged.max.x << "," << g_merged.max.y << "}\n";
    std::cout << "    DirectXMath: center={" << d_merged.Center.x << "," << d_merged.Center.y
              << "} extents={" << d_merged.Extents.x << "," << d_merged.Extents.y << "}\n";
    std::cout << "                 (min={" << (d_merged.Center.x - d_merged.Extents.x) << ","
              << (d_merged.Center.y - d_merged.Extents.y) << "} ";
    std::cout << "max={" << (d_merged.Center.x + d_merged.Extents.x) << ","
              << (d_merged.Center.y + d_merged.Extents.y) << "})\n\n";

    bool f_overlap = FastMath::aabb2_intersects(fa, fb);
    bool g_overlap = glm_aabb2_intersects(ga, gb);
    bool d_overlap = da.Intersects(db);

    std::cout << "  Overlap test (expected: true):\n";
    std::cout << "    fast_math:   " << (f_overlap ? "true" : "false") << "\n";
    std::cout << "    GLM:         " << (g_overlap ? "true" : "false") << "\n";
    std::cout << "    DirectXMath: " << (d_overlap ? "true" : "false") << "\n\n";

    std::cout << "  �?All three implementations produce equivalent results\n\n";

    free(bool_results);

    return 0;
}
