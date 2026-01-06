/**
 * @file bench_aabb3_contains_optimization.cpp
 * @brief 详细对比测试：aabb3Contains的5个优化版本
 *
 * 测试目标：缩小与DirectXMath的47%性能差距
 *
 * 测试版本：
 * - Baseline: 当前实现（使用_mm_setr_ps，对point取负）
 * - V1: 使用_mm_mul_ps取负（模仿DirectXMath）
 * - V2: 模仿DirectXMath InBounds策略（Center+Extents）
 * - V3: 优化load策略（unaligned load）
 * - V4: 使用_mm_cmpge_ps减少逻辑反转
 *
 * 对比：
 * - DirectXMath BoundingBox::Contains
 */

#include "fast_math/aabb3.h"
#include "fast_math/aabb3_contains_optimized.h"

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
using namespace MMath;

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
    cout << "\n";
    cout << "================================================================\n";
    cout << "  aabb3Contains 深度优化分析\n";
    cout << "================================================================\n";
    cout << "  目标：缩小与DirectXMath的47%性能差距\n";
    cout << "  当前：MMath 33.24ms vs DirectXMath 22.68ms\n";
    cout << "================================================================\n";
    cout << "  测试配置：10M operations (10K × 1000 iterations)\n";
    cout << "  编译器：Clang -O3 -march=native -mavx2 -mfma\n";
    cout << "  统计：中位数取5次运行\n";
    cout << "================================================================\n\n";

    const int COUNT = 10000;
    const int ITERS = 1000;
    const int64_t TOTAL_OPS = (int64_t)COUNT * ITERS;

    // Setup test data (same for all versions)
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    std::vector<Aabb3> aabbs(COUNT);
    std::vector<Vec3> points(COUNT);
    std::vector<BoundingBox> dxm_boxes(COUNT);
    std::vector<XMVECTOR> dxm_points(COUNT);

    for (int i = 0; i < COUNT; ++i) {
        Vec3 min = {dist(rng), dist(rng), dist(rng)};
        Vec3 max = {min.x + std::abs(dist(rng)),
                    min.y + std::abs(dist(rng)),
                    min.z + std::abs(dist(rng))};
        aabbs[i] = aabb3FromMinMax(min, max);
        points[i] = {dist(rng), dist(rng), dist(rng)};

        // DirectXMath (Center+Extents)
        Vec3 center = {(min.x + max.x) * 0.5f, (min.y + max.y) * 0.5f, (min.z + max.z) * 0.5f};
        Vec3 extents = {(max.x - min.x) * 0.5f, (max.y - min.y) * 0.5f, (max.z - min.z) * 0.5f};
        dxm_boxes[i].Center = XMFLOAT3(center.x, center.y, center.z);
        dxm_boxes[i].Extents = XMFLOAT3(extents.x, extents.y, extents.z);
        dxm_points[i] = XMVectorSet(points[i].x, points[i].y, points[i].z, 0.0f);
    }

    std::vector<bool> results(COUNT);
    volatile float sink = 0.0f;

    cout << "指令分析：\n";
    cout << "  Baseline:  2 load + 1 sub (negate point) + 2 cmp + 1 and\n";
    cout << "  V1:        2 load + 1 mul (negate max) + 2 cmp + 1 and\n";
    cout << "  V2:        2 load + InBounds (3 sub + 2 mul + 2 cmp + 1 and)\n";
    cout << "  V3:        1 loadu + extract + 1 mul + 2 cmp + 1 and\n";
    cout << "  V4:        2 load + 1 sub + 2 cmp + 1 and\n";
    cout << "  DirectXMath: 2 load + 1 sub + 1 mul + 2 cmp + 1 and\n";
    cout << "================================================================\n\n";

    // Benchmark Baseline
    double baseline_time = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter) {
            for (int i = 0; i < COUNT; ++i) {
                results[i] = aabb3Contains_baseline(aabbs[i], points[i]);
            }
        }
        sink = results[0] ? 1.0f : 0.0f;
    });

    // Benchmark V1: mul instead of sub
    double v1_time = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter) {
            for (int i = 0; i < COUNT; ++i) {
                results[i] = aabb3Contains_v1(aabbs[i], points[i]);
            }
        }
        sink = results[0] ? 1.0f : 0.0f;
    });

    // Benchmark V2: InBounds strategy
    double v2_time = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter) {
            for (int i = 0; i < COUNT; ++i) {
                results[i] = aabb3Contains_v2(aabbs[i], points[i]);
            }
        }
        sink = results[0] ? 1.0f : 0.0f;
    });

    // Benchmark V3: optimized load
    double v3_time = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter) {
            for (int i = 0; i < COUNT; ++i) {
                results[i] = aabb3Contains_v3(aabbs[i], points[i]);
            }
        }
        sink = results[0] ? 1.0f : 0.0f;
    });

    // Benchmark V4: cmpge
    double v4_time = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter) {
            for (int i = 0; i < COUNT; ++i) {
                results[i] = aabb3Contains_v4(aabbs[i], points[i]);
            }
        }
        sink = results[0] ? 1.0f : 0.0f;
    });

    // Benchmark DirectXMath
    double dxm_time = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter) {
            for (int i = 0; i < COUNT; ++i) {
                results[i] = (dxm_boxes[i].Contains(dxm_points[i]) != DISJOINT);
            }
        }
        sink = results[0] ? 1.0f : 0.0f;
    });

    // Results
    cout << "性能结果：\n";
    cout << "------------------------------------------------------------\n";

    double best = std::min({baseline_time, v1_time, v2_time, v3_time, v4_time, dxm_time});

    auto print_result = [&](const string& name, double time) {
        cout << "  " << setw(20) << left << name
             << fixed << setprecision(2) << setw(10) << time << " ms  ("
             << setprecision(1) << (TOTAL_OPS / time / 1e6) << " M/s)";
        if (time == best) {
            cout << "  [FASTEST]";
        } else {
            double ratio = time / best;
            cout << "  [" << setprecision(2) << ratio << "x slower]";
        }

        // vs DirectXMath
        if (time != dxm_time) {
            double vs_dxm = (time / dxm_time - 1.0) * 100.0;
            if (vs_dxm > 0) {
                cout << "  [+" << setprecision(1) << vs_dxm << "% vs DXM]";
            } else {
                cout << "  [" << setprecision(1) << vs_dxm << "% vs DXM]";
            }
        }
        cout << "\n";
    };

    print_result("Baseline (current)", baseline_time);
    print_result("V1 (mul negate)", v1_time);
    print_result("V2 (InBounds)", v2_time);
    print_result("V3 (loadu)", v3_time);
    print_result("V4 (cmpge)", v4_time);
    print_result("DirectXMath", dxm_time);

    cout << "\n================================================================\n";
    cout << "  详细分析\n";
    cout << "================================================================\n\n";

    double improvement_v1 = (baseline_time - v1_time) / baseline_time * 100.0;
    double improvement_v2 = (baseline_time - v2_time) / baseline_time * 100.0;
    double improvement_v3 = (baseline_time - v3_time) / baseline_time * 100.0;
    double improvement_v4 = (baseline_time - v4_time) / baseline_time * 100.0;

    double gap_baseline = (baseline_time - dxm_time) / dxm_time * 100.0;
    double gap_best = (best - dxm_time) / dxm_time * 100.0;

    cout << "相对Baseline的改进：\n";
    cout << "  V1 (mul negate):   " << showpos << fixed << setprecision(1) << improvement_v1 << "%\n";
    cout << "  V2 (InBounds):     " << improvement_v2 << "%\n";
    cout << "  V3 (loadu):        " << improvement_v3 << "%\n";
    cout << "  V4 (cmpge):        " << improvement_v4 << "%\n";
    cout << noshowpos;

    cout << "\n与DirectXMath的差距：\n";
    cout << "  Baseline差距:      +" << fixed << setprecision(1) << gap_baseline << "%\n";
    cout << "  最优版本差距:      " << showpos << gap_best << "%\n" << noshowpos;

    if (gap_best < 0) {
        cout << "\n  ✅ 成功！MMath已超越DirectXMath！\n";
    } else if (gap_best < 10) {
        cout << "\n  ⚡ 接近！差距已缩小到10%以内\n";
    } else {
        cout << "\n  ⚠️  仍有优化空间\n";
    }

    cout << "\n关键洞察：\n";
    cout << "  1. DirectXMath使用Center+Extents表示，天然适合Contains测试\n";
    cout << "  2. DirectXMath的Point参数在XMVECTOR中（寄存器传递）\n";
    cout << "  3. MMath使用Min+NegMax表示，优化了Union但Contains不利\n";
    cout << "  4. 这是设计权衡：不同数据表示适合不同操作\n";

    cout << "\n建议：\n";
    if (best < baseline_time) {
        cout << "  ✓ 采用最优版本替换当前实现\n";
        cout << "  ✓ 可获得" << fixed << setprecision(1)
             << (baseline_time - best) / baseline_time * 100.0 << "%性能提升\n";
    } else {
        cout << "  ✓ 当前实现已是最优\n";
        cout << "  ✓ 进一步优化需改变数据结构（不推荐）\n";
    }

    cout << "\n================================================================\n\n";

    return 0;
}
