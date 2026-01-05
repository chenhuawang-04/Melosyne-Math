/**
 * @file diagnose_simd_bottleneck.cpp
 * @brief 诊断SIMD性能瓶颈
 */

#include <iostream>
#include <chrono>
#include <iomanip>
#include <random>
#include <fast_math/bit_ops.h>

using namespace Melosyne;
using namespace std::chrono;

class PerfTimer {
    high_resolution_clock::time_point start;
public:
    void begin() { start = high_resolution_clock::now(); }
    double elapsedNs() const {
        auto end = high_resolution_clock::now();
        return duration<double, std::nano>(end - start).count();
    }
};

// 测试1: 直接调用标量版本
template<std::size_t N>
double testScalarDirect(std::size_t iterations) {
    std::mt19937_64 rng(12345);
    BitSet<N> a, b;
    for (auto& w : a.words) w = rng();
    for (auto& w : b.words) w = rng();

    PerfTimer timer;
    timer.begin();

    for (std::size_t i = 0; i < iterations; ++i) {
        BitOps::bitwiseAnd(a, b);
    }

    return timer.elapsedNs() / iterations;
}

// 测试2: 通过Optimized包装调用（会走SIMD）
template<std::size_t N>
double testOptimizedWrapper(std::size_t iterations) {
    std::mt19937_64 rng(12345);
    BitSet<N> a, b;
    for (auto& w : a.words) w = rng();
    for (auto& w : b.words) w = rng();

    PerfTimer timer;
    timer.begin();

    for (std::size_t i = 0; i < iterations; ++i) {
        BitOps::bitwiseAndOptimized(a, b);
    }

    return timer.elapsedNs() / iterations;
}

// 测试3: 直接调用SIMD版本（绕过wrapper）
template<std::size_t N>
double testSimdDirect(std::size_t iterations) {
    std::mt19937_64 rng(12345);
    BitSet<N> a, b;
    for (auto& w : a.words) w = rng();
    for (auto& w : b.words) w = rng();

    PerfTimer timer;
    timer.begin();

    for (std::size_t i = 0; i < iterations; ++i) {
#if defined(__AVX2__)
        BitOps::detail::bitwiseAndSimd(BitSetView(a), ConstBitSetView(b));
#else
        BitOps::bitwiseAnd(a, b);
#endif
    }

    return timer.elapsedNs() / iterations;
}

// 测试4: 手工内联的SIMD代码
template<std::size_t N>
double testSimdInlined(std::size_t iterations) {
    std::mt19937_64 rng(12345);
    BitSet<N> a, b;
    for (auto& w : a.words) w = rng();
    for (auto& w : b.words) w = rng();

    // 防止编译器优化掉
    volatile uint64_t sink = 0;

    PerfTimer timer;
    timer.begin();

    for (std::size_t i = 0; i < iterations; ++i) {
#if defined(__AVX2__)
        // 手工内联SIMD代码
        constexpr std::size_t word_count = BitSet<N>::kWords;
        constexpr std::size_t avx2_words = word_count & ~std::size_t{3};

        std::size_t j = 0;
        for (; j < avx2_words; j += 4) {
            __m256i av = _mm256_load_si256(reinterpret_cast<const __m256i*>(a.words + j));
            __m256i bv = _mm256_load_si256(reinterpret_cast<const __m256i*>(b.words + j));
            __m256i result = _mm256_and_si256(av, bv);
            _mm256_store_si256(reinterpret_cast<__m256i*>(a.words + j), result);
        }

        for (; j < word_count; ++j) {
            a.words[j] &= b.words[j];
        }
#else
        for (std::size_t j = 0; j < BitSet<N>::kWords; ++j) {
            a.words[j] &= b.words[j];
        }
#endif
        // 防止优化
        sink = a.words[0];
    }

    return timer.elapsedNs() / iterations;
}

// 测试5: 纯标量循环（编译器自动矢量化）
template<std::size_t N>
double testScalarAutoVec(std::size_t iterations) {
    std::mt19937_64 rng(12345);
    BitSet<N> a, b;
    for (auto& w : a.words) w = rng();
    for (auto& w : b.words) w = rng();

    // 防止编译器优化掉
    volatile uint64_t sink = 0;

    PerfTimer timer;
    timer.begin();

    for (std::size_t i = 0; i < iterations; ++i) {
        // 简单循环，让编译器自动矢量化
        for (std::size_t j = 0; j < BitSet<N>::kWords; ++j) {
            a.words[j] &= b.words[j];
        }
        // 防止优化
        sink = a.words[0];
    }

    return timer.elapsedNs() / iterations;
}

// 测试6: 冷数据测试（每次操作不同数据）
template<std::size_t N>
double testColdDataSimd(std::size_t iterations) {
    std::mt19937_64 rng(12345);
    std::vector<BitSet<N>> data_a(iterations);
    std::vector<BitSet<N>> data_b(iterations);

    for (std::size_t i = 0; i < iterations; ++i) {
        for (auto& w : data_a[i].words) w = rng();
        for (auto& w : data_b[i].words) w = rng();
    }

    PerfTimer timer;
    timer.begin();

    for (std::size_t i = 0; i < iterations; ++i) {
        BitOps::bitwiseAndOptimized(data_a[i], data_b[i]);
    }

    return timer.elapsedNs() / iterations;
}

template<std::size_t N>
void runDiagnostics(std::size_t iterations) {
    std::cout << "\n============================================================\n";
    std::cout << "BitSet<" << N << "> (" << BitSet<N>::kWords << " words, "
              << BitSet<N>::kBytes << " bytes)\n";
    std::cout << "============================================================\n";

    double t1 = testScalarDirect<N>(iterations);
    double t2 = testOptimizedWrapper<N>(iterations);
    double t3 = testSimdDirect<N>(iterations);
    double t4 = testSimdInlined<N>(iterations);
    double t5 = testScalarAutoVec<N>(iterations);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  1. 标量版本（直接调用）:     " << std::setw(8) << t1 << " ns/op\n";
    std::cout << "  2. Optimized包装（走SIMD）:  " << std::setw(8) << t2 << " ns/op";
    if (t2 > t1) {
        std::cout << "  ❌ 慢了 " << std::setprecision(1) << ((t2/t1 - 1)*100) << "%\n";
    } else {
        std::cout << "  ✅ 快了 " << std::setprecision(1) << ((1 - t2/t1)*100) << "%\n";
    }

    std::cout << "  3. SIMD版本（直接调用）:     " << std::setw(8) << t3 << " ns/op";
    if (t3 > t1) {
        std::cout << "  ❌ 慢了 " << std::setprecision(1) << ((t3/t1 - 1)*100) << "%\n";
    } else {
        std::cout << "  ✅ 快了 " << std::setprecision(1) << ((1 - t3/t1)*100) << "%\n";
    }

    std::cout << "  4. SIMD手工内联:             " << std::setw(8) << t4 << " ns/op";
    if (t4 > t1) {
        std::cout << "  ❌ 慢了 " << std::setprecision(1) << ((t4/t1 - 1)*100) << "%\n";
    } else {
        std::cout << "  ✅ 快了 " << std::setprecision(1) << ((1 - t4/t1)*100) << "%\n";
    }

    std::cout << "  5. 标量循环（自动矢量化）:   " << std::setw(8) << t5 << " ns/op";
    if (t5 > t1) {
        std::cout << "  ❌ 慢了 " << std::setprecision(1) << ((t5/t1 - 1)*100) << "%\n";
    } else {
        std::cout << "  ✅ 快了 " << std::setprecision(1) << ((1 - t5/t1)*100) << "%\n";
    }

    // 分析瓶颈
    std::cout << "\n  瓶颈分析:\n";

    if (t2 > t3 + 0.5) {
        std::cout << "  ⚠️  包装函数开销: " << (t2 - t3) << " ns ("
                  << std::setprecision(1) << ((t2 - t3)/t1 * 100) << "% of total)\n";
    }

    if (t3 > t4 + 0.5) {
        std::cout << "  ⚠️  函数调用开销: " << (t3 - t4) << " ns\n";
    }

    if (t4 > t5 + 0.5) {
        std::cout << "  ⚠️  SIMD慢于自动矢量化: " << (t4 - t5) << " ns\n";
        std::cout << "      可能原因: L1缓存已足够快，SIMD设置开销抵消收益\n";
    }

    if (t5 > t1 + 0.5) {
        std::cout << "  ⚠️  BitSetView构造开销: " << (t5 - t1) << " ns\n";
    }

    // 冷数据测试（仅对中等和大位集）
    if (N >= 4096) {
        std::cout << "\n  冷数据测试 (1000次不同数据):\n";
        double t6 = testColdDataSimd<N>(1000);
        std::cout << "  6. 冷数据SIMD:                " << std::setw(8) << t6 << " ns/op";
        if (t6 < t1) {
            std::cout << "  ✅ 快了 " << std::setprecision(1) << ((1 - t6/t1)*100) << "%\n";
            std::cout << "      结论: SIMD在冷数据场景下有优势！\n";
        } else {
            std::cout << "  ❌ 仍然慢\n";
        }
    }
}

int main() {
    std::cout << "============================================================\n";
    std::cout << "SIMD性能瓶颈诊断\n";
    std::cout << "============================================================\n";

#if defined(__AVX2__)
    std::cout << "✅ AVX2已启用\n";
#else
    std::cout << "❌ AVX2未启用\n";
#endif

    // 测试不同大小的BitSet
    runDiagnostics<256>(10000000);   // 4 words (小位集)
    runDiagnostics<4096>(1000000);   // 64 words (中等位集)
    runDiagnostics<65536>(100000);   // 1024 words (大位集)

    std::cout << "\n============================================================\n";
    std::cout << "诊断完成\n";
    std::cout << "============================================================\n";

    return 0;
}
