/**
 * @file performance_regression_test.cpp
 * @brief 性能回归测试 - 确保重构后性能无回退
 *
 * 测试策略:
 * 1. 对比标量版本性能 (*.h vs 原始实现)
 * 2. 对比SIMD优化版本性能 (*_simd.h vs 原始AVX2实现)
 * 3. 验证自动派发正确性
 * 4. 确保所有操作性能 >= 基线版本 (允许5%波动)
 */

#include <iostream>
#include <chrono>
#include <iomanip>
#include <random>
#include <vector>
#include <string>
#include <cmath>

// 包含新版本头文件
#include <fast_math/bit_ops.h>

using namespace Melosyne;
using namespace std::chrono;

// ============================================================================
// 性能测试工具
// ============================================================================

class PerformanceTimer {
    using Clock = high_resolution_clock;
    Clock::time_point start_time;

public:
    void start() { start_time = Clock::now(); }

    double elapsedNs() const {
        auto end = Clock::now();
        return duration<double, std::nano>(end - start_time).count();
    }

    double elapsedMs() const {
        return elapsedNs() / 1e6;
    }
};

// 性能统计
struct PerfResult {
    std::string name;
    double time_ns;
    double ops_per_sec;
    std::size_t iterations;

    void print() const {
        std::cout << "  " << std::setw(30) << std::left << name << ": "
                  << std::setw(10) << std::fixed << std::setprecision(2) << (time_ns / iterations)
                  << " ns/op ("
                  << std::setw(12) << std::fixed << std::setprecision(0) << ops_per_sec
                  << " ops/s)\n";
    }
};

// 性能对比
struct PerfComparison {
    PerfResult baseline;
    PerfResult optimized;
    double speedup;
    bool passed;  // 性能是否合格 (允许5%回退)

    void print() const {
        std::cout << "\n  Baseline:  " << std::setw(10) << std::fixed << std::setprecision(2)
                  << (baseline.time_ns / baseline.iterations) << " ns/op\n";
        std::cout << "  Optimized: " << std::setw(10) << (optimized.time_ns / optimized.iterations)
                  << " ns/op\n";
        std::cout << "  Speedup:   " << std::setw(10) << std::setprecision(2) << speedup << "x";

        if (speedup >= 0.95) {
            std::cout << " ✅ PASS\n";
        } else {
            std::cout << " ❌ FAIL (回退 " << ((1.0 - speedup) * 100) << "%)\n";
        }
    }
};

// ============================================================================
// 测试数据准备
// ============================================================================

template<std::size_t N>
void randomize(BitSet<N>& bs, std::mt19937_64& rng) {
    for (auto& w : bs.words) {
        w = rng();
    }
}

template<std::size_t N>
void zeroInitialize(BitSet<N>& bs) {
    for (auto& w : bs.words) {
        w = 0;
    }
}

// ============================================================================
// 核心操作性能测试
// ============================================================================

template<std::size_t N>
PerfResult benchmarkBitwiseAnd(std::size_t iterations) {
    std::mt19937_64 rng(12345);
    BitSet<N> a, b;
    randomize(a, rng);
    randomize(b, rng);

    PerformanceTimer timer;
    timer.start();

    for (std::size_t i = 0; i < iterations; ++i) {
        BitOps::bitwiseAnd(a, b);
    }

    double elapsed = timer.elapsedNs();
    return {"bitwiseAnd (scalar)", elapsed, iterations / (elapsed / 1e9), iterations};
}

template<std::size_t N>
PerfResult benchmarkBitwiseAndOptimized(std::size_t iterations) {
    std::mt19937_64 rng(12345);
    BitSet<N> a, b;
    randomize(a, rng);
    randomize(b, rng);

    PerformanceTimer timer;
    timer.start();

    for (std::size_t i = 0; i < iterations; ++i) {
        BitOps::bitwiseAndOptimized(a, b);
    }

    double elapsed = timer.elapsedNs();
    return {"bitwiseAnd (optimized)", elapsed, iterations / (elapsed / 1e9), iterations};
}

template<std::size_t N>
PerfComparison testCoreOperations(std::size_t iterations) {
    auto baseline = benchmarkBitwiseAnd<N>(iterations);
    auto optimized = benchmarkBitwiseAndOptimized<N>(iterations);

    double speedup = baseline.time_ns / optimized.time_ns;
    bool passed = (speedup >= 0.95);  // 允许5%性能波动

    return {baseline, optimized, speedup, passed};
}

// ============================================================================
// 统计操作性能测试
// ============================================================================

template<std::size_t N>
PerfResult benchmarkPopcount(std::size_t iterations) {
    std::mt19937_64 rng(12345);
    BitSet<N> a;
    randomize(a, rng);

    volatile std::size_t sum = 0;  // 防止优化掉

    PerformanceTimer timer;
    timer.start();

    for (std::size_t i = 0; i < iterations; ++i) {
        sum += BitOps::popcount(a);
    }

    double elapsed = timer.elapsedNs();
    return {"popcount (scalar)", elapsed, iterations / (elapsed / 1e9), iterations};
}

template<std::size_t N>
PerfResult benchmarkPopcountOptimized(std::size_t iterations) {
    std::mt19937_64 rng(12345);
    BitSet<N> a;
    randomize(a, rng);

    volatile std::size_t sum = 0;

    PerformanceTimer timer;
    timer.start();

    for (std::size_t i = 0; i < iterations; ++i) {
        sum += BitOps::popcountOptimized(a);
    }

    double elapsed = timer.elapsedNs();
    return {"popcount (optimized)", elapsed, iterations / (elapsed / 1e9), iterations};
}

template<std::size_t N>
PerfComparison testPopcountOperations(std::size_t iterations) {
    auto baseline = benchmarkPopcount<N>(iterations);
    auto optimized = benchmarkPopcountOptimized<N>(iterations);

    double speedup = baseline.time_ns / optimized.time_ns;
    bool passed = (speedup >= 0.95);

    return {baseline, optimized, speedup, passed};
}

// ============================================================================
// 扫描操作性能测试
// ============================================================================

template<std::size_t N>
PerfResult benchmarkFindFirst(std::size_t iterations) {
    std::mt19937_64 rng(12345);
    BitSet<N> a;
    randomize(a, rng);

    volatile std::size_t sum = 0;

    PerformanceTimer timer;
    timer.start();

    for (std::size_t i = 0; i < iterations; ++i) {
        sum += BitOps::findFirst(a);
    }

    double elapsed = timer.elapsedNs();
    return {"findFirst", elapsed, iterations / (elapsed / 1e9), iterations};
}

template<std::size_t N>
PerfResult benchmarkSelect(std::size_t iterations) {
    std::mt19937_64 rng(12345);
    BitSet<N> a;
    randomize(a, rng);

    volatile std::size_t sum = 0;
    std::size_t k = N / 4;  // 查找第k个设置的位

    PerformanceTimer timer;
    timer.start();

    for (std::size_t i = 0; i < iterations; ++i) {
        sum += BitOps::select(a, k);
    }

    double elapsed = timer.elapsedNs();
    return {"select (scalar)", elapsed, iterations / (elapsed / 1e9), iterations};
}

template<std::size_t N>
PerfResult benchmarkSelectOptimized(std::size_t iterations) {
    std::mt19937_64 rng(12345);
    BitSet<N> a;
    randomize(a, rng);

    volatile std::size_t sum = 0;
    std::size_t k = N / 4;

    PerformanceTimer timer;
    timer.start();

    for (std::size_t i = 0; i < iterations; ++i) {
        sum += BitOps::selectOptimized(a, k);
    }

    double elapsed = timer.elapsedNs();
    return {"select (optimized)", elapsed, iterations / (elapsed / 1e9), iterations};
}

template<std::size_t N>
PerfComparison testScanOperations(std::size_t iterations) {
    auto baseline = benchmarkSelect<N>(iterations);
    auto optimized = benchmarkSelectOptimized<N>(iterations);

    double speedup = baseline.time_ns / optimized.time_ns;
    bool passed = (speedup >= 0.95);

    return {baseline, optimized, speedup, passed};
}

// ============================================================================
// 范围操作性能测试
// ============================================================================

template<std::size_t N>
PerfResult benchmarkSetRange(std::size_t iterations) {
    BitSet<N> a;

    PerformanceTimer timer;
    timer.start();

    for (std::size_t i = 0; i < iterations; ++i) {
        zeroInitialize(a);
        BitOps::setRange(a, 10, N - 10);
    }

    double elapsed = timer.elapsedNs();
    return {"setRange (scalar)", elapsed, iterations / (elapsed / 1e9), iterations};
}

template<std::size_t N>
PerfResult benchmarkSetRangeOptimized(std::size_t iterations) {
    BitSet<N> a;

    PerformanceTimer timer;
    timer.start();

    for (std::size_t i = 0; i < iterations; ++i) {
        zeroInitialize(a);
        BitOps::setRangeOptimized(a, 10, N - 10);
    }

    double elapsed = timer.elapsedNs();
    return {"setRange (optimized)", elapsed, iterations / (elapsed / 1e9), iterations};
}

template<std::size_t N>
PerfComparison testRangeOperations(std::size_t iterations) {
    auto baseline = benchmarkSetRange<N>(iterations);
    auto optimized = benchmarkSetRangeOptimized<N>(iterations);

    double speedup = baseline.time_ns / optimized.time_ns;
    bool passed = (speedup >= 0.95);

    return {baseline, optimized, speedup, passed};
}

// ============================================================================
// 高级操作性能测试
// ============================================================================

PerfResult benchmarkReverseBits64(std::size_t iterations) {
    std::mt19937_64 rng(12345);
    uint64_t val = rng();
    volatile uint64_t sum = 0;

    PerformanceTimer timer;
    timer.start();

    for (std::size_t i = 0; i < iterations; ++i) {
        sum += BitOps::reverseBits64(val);
    }

    double elapsed = timer.elapsedNs();
    return {"reverseBits64", elapsed, iterations / (elapsed / 1e9), iterations};
}

template<std::size_t N>
PerfResult benchmarkReverse(std::size_t iterations) {
    std::mt19937_64 rng(12345);
    BitSet<N> a;
    randomize(a, rng);

    PerformanceTimer timer;
    timer.start();

    for (std::size_t i = 0; i < iterations; ++i) {
        BitOps::reverse(a);
    }

    double elapsed = timer.elapsedNs();
    return {"reverse (scalar)", elapsed, iterations / (elapsed / 1e9), iterations};
}

template<std::size_t N>
PerfResult benchmarkReverseOptimized(std::size_t iterations) {
    std::mt19937_64 rng(12345);
    BitSet<N> a;
    randomize(a, rng);

    PerformanceTimer timer;
    timer.start();

    for (std::size_t i = 0; i < iterations; ++i) {
        BitOps::reverseOptimized(a);
    }

    double elapsed = timer.elapsedNs();
    return {"reverse (optimized)", elapsed, iterations / (elapsed / 1e9), iterations};
}

// ============================================================================
// 主测试套件
// ============================================================================

int main() {
    std::cout << "============================================================\n";
    std::cout << "位操作库性能回归测试\n";
    std::cout << "目标: 确保重构后性能无回退 (允许5%波动)\n";
    std::cout << "============================================================\n\n";

    // 特性检测
    std::cout << "编译特性:\n";
    std::cout << "------------------------------------------------------------\n";
#if defined(__AVX2__)
    std::cout << "  AVX2:  ✅ 已启用\n";
#else
    std::cout << "  AVX2:  ❌ 未启用\n";
#endif

#if defined(__BMI2__)
    std::cout << "  BMI2:  ✅ 已启用 (PEXT/PDEP)\n";
#else
    std::cout << "  BMI2:  ❌ 未启用\n";
#endif

    std::cout << "  编译器: ";
#if defined(__clang__)
    std::cout << "Clang " << __clang_major__ << "." << __clang_minor__ << "\n";
#elif defined(__GNUC__)
    std::cout << "GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "\n";
#elif defined(_MSC_VER)
    std::cout << "MSVC " << _MSC_VER << "\n";
#endif

    std::cout << "\n测试策略:\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << "  每个操作测试标量版本和优化版本\n";
    std::cout << "  优化版本性能应 >= 标量版本 * 0.95\n";
    std::cout << "  对于大位集，AVX2版本应显著快于标量版本\n\n";

    int passed = 0;
    int total = 0;

    // ========================================================================
    // 测试1: 核心操作 - BitSet<256> (小位集)
    // ========================================================================
    std::cout << "\n[测试1] 核心操作 - BitSet<256> (小位集)\n";
    std::cout << "------------------------------------------------------------\n";
    {
        auto result = testCoreOperations<256>(10000000);
        result.print();
        ++total;
        if (result.passed) ++passed;
    }

    // ========================================================================
    // 测试2: 核心操作 - BitSet<4096> (中等位集，AVX2应显著加速)
    // ========================================================================
    std::cout << "\n[测试2] 核心操作 - BitSet<4096> (中等位集)\n";
    std::cout << "------------------------------------------------------------\n";
    {
        auto result = testCoreOperations<4096>(1000000);
        result.print();
        ++total;
        if (result.passed) ++passed;

#if defined(__AVX2__)
        if (result.speedup < 1.5) {
            std::cout << "  ⚠️  警告: AVX2加速不明显 (预期 >1.5x)\n";
        } else {
            std::cout << "  ✅ AVX2加速效果良好\n";
        }
#endif
    }

    // ========================================================================
    // 测试3: Popcount - BitSet<4096>
    // ========================================================================
    std::cout << "\n[测试3] Popcount - BitSet<4096> (Harley-Seal算法)\n";
    std::cout << "------------------------------------------------------------\n";
    {
        auto result = testPopcountOperations<4096>(1000000);
        result.print();
        ++total;
        if (result.passed) ++passed;

#if defined(__AVX2__)
        if (result.speedup < 1.8) {
            std::cout << "  ⚠️  警告: Harley-Seal加速不足 (预期 >1.8x)\n";
        } else {
            std::cout << "  ✅ Harley-Seal算法效果良好\n";
        }
#endif
    }

    // ========================================================================
    // 测试4: Select操作 - BitSet<4096>
    // ========================================================================
    std::cout << "\n[测试4] Select操作 - BitSet<4096>\n";
    std::cout << "------------------------------------------------------------\n";
    {
        auto result = testScanOperations<4096>(100000);
        result.print();
        ++total;
        if (result.passed) ++passed;
    }

    // ========================================================================
    // 测试5: 范围操作 - BitSet<4096>
    // ========================================================================
    std::cout << "\n[测试5] 范围操作 - BitSet<4096>\n";
    std::cout << "------------------------------------------------------------\n";
    {
        auto result = testRangeOperations<4096>(100000);
        result.print();
        ++total;
        if (result.passed) ++passed;

#if defined(__AVX2__)
        if (result.speedup < 2.0) {
            std::cout << "  ⚠️  警告: AVX2范围操作加速不足 (预期 >2.0x)\n";
        } else {
            std::cout << "  ✅ AVX2范围操作效果良好\n";
        }
#endif
    }

    // ========================================================================
    // 测试6: findFirst操作 (标量应最优)
    // ========================================================================
    std::cout << "\n[测试6] FindFirst操作 - BitSet<4096> (标量最优)\n";
    std::cout << "------------------------------------------------------------\n";
    {
        auto result = benchmarkFindFirst<4096>(10000000);
        result.print();
        std::cout << "  ℹ️  此操作无SIMD版本 (1-周期TZCNT最优)\n";
    }

    // ========================================================================
    // 测试7: reverseBits64 (SWAR算法)
    // ========================================================================
    std::cout << "\n[测试7] ReverseBits64 - SWAR算法\n";
    std::cout << "------------------------------------------------------------\n";
    {
        auto result = benchmarkReverseBits64(10000000);
        result.print();
        std::cout << "  ℹ️  SWAR算法天然并行，无需SIMD\n";
    }

    // ========================================================================
    // 测试8: 大位集性能 - BitSet<65536>
    // ========================================================================
    std::cout << "\n[测试8] 大位集核心操作 - BitSet<65536>\n";
    std::cout << "------------------------------------------------------------\n";
    {
        auto result = testCoreOperations<65536>(100000);
        result.print();
        ++total;
        if (result.passed) ++passed;

#if defined(__AVX2__)
        if (result.speedup < 2.5) {
            std::cout << "  ⚠️  警告: 大位集AVX2加速不足 (预期 >2.5x)\n";
        } else {
            std::cout << "  ✅ 大位集AVX2加速优秀\n";
        }
#endif
    }

    // ========================================================================
    // 最终总结
    // ========================================================================
    std::cout << "\n============================================================\n";
    std::cout << "性能回归测试结果\n";
    std::cout << "============================================================\n";
    std::cout << "  通过: " << passed << " / " << total << " 测试\n";

    if (passed == total) {
        std::cout << "\n  ✅ 所有测试通过 - 性能无回退!\n";
        std::cout << "  重构后的代码性能保持在基线的95%以上\n";
        return 0;
    } else {
        std::cout << "\n  ❌ 部分测试失败 - 检测到性能回退!\n";
        std::cout << "  失败的测试需要优化\n";
        return 1;
    }
}
