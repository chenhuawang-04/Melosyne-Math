/**
 * @file test_guaranteed_vectorization.cpp
 * @brief 测试如何保证编译器矢量化
 */

#include <iostream>
#include <chrono>
#include <iomanip>
#include <fast_math/bit_ops.h>

using namespace Melosyne;
using namespace std::chrono;

class Timer {
    high_resolution_clock::time_point start;
public:
    void begin() { start = high_resolution_clock::now(); }
    double ns() const {
        return duration<double, std::nano>(high_resolution_clock::now() - start).count();
    }
};

// 方案1: 原始手工AVX2
template<std::size_t N>
double testManualAVX2(std::size_t iters) {
    BitSet<N> a, b;
    for (auto& w : a.words) w = 0xAAAAAAAAAAAAAAAA;
    for (auto& w : b.words) w = 0x5555555555555555;

    volatile uint64_t sink = 0;
    Timer t;
    t.begin();

    for (std::size_t i = 0; i < iters; ++i) {
        BitOps::bitwiseAndOptimized(a, b);
        sink = a.words[0];
    }

    return t.ns() / iters;
}

// 方案2: 纯标量（让编译器决定）
template<std::size_t N>
double testAutoVectorize(std::size_t iters) {
    BitSet<N> a, b;
    for (auto& w : a.words) w = 0xAAAAAAAAAAAAAAAA;
    for (auto& w : b.words) w = 0x5555555555555555;

    volatile uint64_t sink = 0;
    Timer t;
    t.begin();

    for (std::size_t i = 0; i < iters; ++i) {
        for (std::size_t j = 0; j < BitSet<N>::kWords; ++j) {
            a.words[j] &= b.words[j];
        }
        sink = a.words[0];
    }

    return t.ns() / iters;
}

// 方案3: 带编译器提示（推荐）
template<std::size_t N>
double testWithHints(std::size_t iters) {
    BitSet<N> a, b;
    for (auto& w : a.words) w = 0xAAAAAAAAAAAAAAAA;
    for (auto& w : b.words) w = 0x5555555555555555;

    volatile uint64_t sink = 0;
    Timer t;
    t.begin();

    for (std::size_t i = 0; i < iters; ++i) {
        uint64_t* __restrict dst = a.words;
        const uint64_t* __restrict src = b.words;

#if defined(_MSC_VER)
        #pragma loop(ivdep)
        #pragma loop(hint_parallel(4))
#elif defined(__clang__)
        #pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUC__)
        #pragma GCC ivdep
#endif
        for (std::size_t j = 0; j < BitSet<N>::kWords; ++j) {
            dst[j] &= src[j];
        }
        sink = a.words[0];
    }

    return t.ns() / iters;
}

// 方案4: 改进的手工AVX2（减少开销）
template<std::size_t N>
double testImprovedAVX2(std::size_t iters) {
    BitSet<N> a, b;
    for (auto& w : a.words) w = 0xAAAAAAAAAAAAAAAA;
    for (auto& w : b.words) w = 0x5555555555555555;

    volatile uint64_t sink = 0;
    Timer t;
    t.begin();

    for (std::size_t i = 0; i < iters; ++i) {
#if defined(__AVX2__)
        constexpr std::size_t count = BitSet<N>::kWords;
        uint64_t* dst = a.words;
        const uint64_t* src = b.words;

        // 完全展开小循环
        if constexpr (count == 4) {
            __m256i av = _mm256_load_si256((__m256i*)dst);
            __m256i bv = _mm256_load_si256((__m256i*)src);
            _mm256_store_si256((__m256i*)dst, _mm256_and_si256(av, bv));
        } else {
            // 大循环使用SIMD
            constexpr std::size_t avx2_count = count & ~std::size_t{3};
            for (std::size_t j = 0; j < avx2_count; j += 4) {
                __m256i av = _mm256_load_si256((__m256i*)(dst + j));
                __m256i bv = _mm256_load_si256((__m256i*)(src + j));
                _mm256_store_si256((__m256i*)(dst + j), _mm256_and_si256(av, bv));
            }
            // 标量余数
            for (std::size_t j = avx2_count; j < count; ++j) {
                dst[j] &= src[j];
            }
        }
#else
        for (std::size_t j = 0; j < BitSet<N>::kWords; ++j) {
            a.words[j] &= b.words[j];
        }
#endif
        sink = a.words[0];
    }

    return t.ns() / iters;
}

template<std::size_t N>
void compare() {
    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "BitSet<" << N << "> (" << BitSet<N>::kWords << " words)\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";

    constexpr std::size_t iters = N < 1000 ? 10000000 : (N < 10000 ? 1000000 : 100000);

    double t1 = testManualAVX2<N>(iters);
    double t2 = testAutoVectorize<N>(iters);
    double t3 = testWithHints<N>(iters);
    double t4 = testImprovedAVX2<N>(iters);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "1. 当前手工AVX2:        " << std::setw(8) << t1 << " ns/op\n";
    std::cout << "2. 纯标量自动矢量化:    " << std::setw(8) << t2 << " ns/op  ";
    if (t2 < t1) std::cout << "✅ 快 " << std::setw(4) << std::setprecision(1)
                           << (1-t2/t1)*100 << "%\n";
    else std::cout << "❌ 慢 " << std::setw(4) << (t2/t1-1)*100 << "%\n";

    std::cout << "3. 带编译器提示:        " << std::setw(8) << t3 << " ns/op  ";
    if (t3 < t1) std::cout << "✅ 快 " << std::setw(4) << (1-t3/t1)*100 << "%\n";
    else std::cout << "❌ 慢 " << std::setw(4) << (t3/t1-1)*100 << "%\n";

    std::cout << "4. 改进手工AVX2:        " << std::setw(8) << t4 << " ns/op  ";
    if (t4 < t1) std::cout << "✅ 快 " << std::setw(4) << (1-t4/t1)*100 << "%\n";
    else std::cout << "❌ 慢 " << std::setw(4) << (t4/t1-1)*100 << "%\n";

    // 推荐最佳方案
    double best = std::min({t1, t2, t3, t4});
    std::cout << "\n推荐方案: ";
    if (best == t2) std::cout << "2 (纯标量自动矢量化) - 简单且快\n";
    else if (best == t3) std::cout << "3 (带编译器提示) - 可靠且快\n";
    else if (best == t4) std::cout << "4 (改进手工AVX2) - 最快\n";
    else std::cout << "1 (保持当前实现)\n";
}

int main() {
    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << "    保证矢量化的方案对比测试\n";
    std::cout << "═══════════════════════════════════════════════════════\n";

#if defined(__AVX2__)
    std::cout << "✅ AVX2已启用\n";
#else
    std::cout << "❌ AVX2未启用\n";
#endif

    compare<256>();
    compare<4096>();
    compare<65536>();

    std::cout << "\n═══════════════════════════════════════════════════════\n";
    std::cout << "结论和建议\n";
    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << "\n方案对比:\n";
    std::cout << "1. 当前手工AVX2: 跨平台一致，但可能有额外开销\n";
    std::cout << "2. 纯标量:       依赖编译器，不保证矢量化\n";
    std::cout << "3. 带编译器提示: 平衡方案，强烈建议矢量化\n";
    std::cout << "4. 改进AVX2:     最可控，但需要更多维护\n";

    return 0;
}
