#include "bench/framework/benchmark_framework.h"

#include "fast_math/bit_ops.h"

#include <bitset>
#include <cstdint>
#include <random>
#include <vector>

namespace {

using namespace MMath;
using namespace MMath::BitOps;

template <typename BitsetT>
void seedPattern(BitsetT& bs_, std::size_t bits_, uint32_t seed_) {
    uint32_t x = seed_;
    for (std::size_t i = 0; i < bits_; ++i) {
        x = x * 1664525u + 1013904223u;
        if ((x >> 31) & 1u) {
            set(bs_, i);
        }
    }
}

std::vector<std::size_t> makeIndices(std::size_t count_, std::size_t limit_, uint32_t seed_) {
    std::vector<std::size_t> out(count_);
    std::mt19937 rng(seed_);
    std::uniform_int_distribution<std::size_t> dist(0, limit_ - 1);
    for (auto& v : out) {
        v = dist(rng);
    }
    return out;
}

template <std::size_t N>
void seedPatternStd(std::bitset<N>& bs_, uint32_t seed_) {
    uint32_t x = seed_;
    for (std::size_t i = 0; i < N; ++i) {
        x = x * 1664525u + 1013904223u;
        if ((x >> 31) & 1u) {
            bs_.set(i);
        }
    }
}

} // namespace

FM_BENCH(DataStructure, DynamicBitSetLifecycle_4K) {
    constexpr std::size_t bits = 4096;
    constexpr int iters = 2048;

    FmBench::runComparisonCase(
        "dynamic bitset lifecycle (4K bits)",
        static_cast<std::size_t>(iters),
        {
            {"fast_math", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < iters; ++i) {
                     DynamicBitSet bs(bits);
                     const std::size_t p0 = static_cast<std::size_t>((i * 17) & (bits - 1));
                     const std::size_t p1 = static_cast<std::size_t>((i * 47) & (bits - 1));
                     bs.set(p0).set(p1);
                     acc += bs.test(p0) ? 1.0 : 0.0;
                     acc += static_cast<double>(bs.wordCount());
                 }
                 FmBench::consume(acc);
             }},
        },
        options.config);
}

FM_BENCH(DataStructure, DynamicBitSetCopyMove_64K) {
    constexpr std::size_t bits = 65536;
    constexpr int iters = 192;

    DynamicBitSet src(bits);
    seedPattern(src, bits, 0xA1B2C3D4u);

    FmBench::runComparisonCase(
        "dynamic bitset copy/move (64K bits)",
        static_cast<std::size_t>(iters * 4),
        {
            {"fast_math", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < iters; ++i) {
                     DynamicBitSet c0(src);               // copy ctor
                     DynamicBitSet c1(std::move(c0));     // move ctor
                     DynamicBitSet c2(bits);
                     c2 = c1;                             // copy assign
                     DynamicBitSet c3;
                     c3 = std::move(c2);                  // move assign
                     acc += static_cast<double>(test(c3, static_cast<std::size_t>((i * 73) & (bits - 1))));
                 }
                 FmBench::consume(acc);
             }},
        },
        options.config);
}

FM_BENCH(DataStructure, DynamicBitSetBitAccessRandom_16K) {
    constexpr std::size_t bits = 16384;
    constexpr int rounds = 256;
    constexpr std::size_t indices = 4096;

    DynamicBitSet bs(bits);
    const auto idx = makeIndices(indices, bits, 0xCAFEBABEu);

    FmBench::runComparisonCase(
        "dynamic bitset random set/reset/test (16K bits)",
        static_cast<std::size_t>(rounds * indices * 3),
        {
            {"fast_math", true, [&]() {
                 double acc = 0.0;
                 for (int r = 0; r < rounds; ++r) {
                     for (std::size_t p : idx) {
                         set(bs, p);
                     }
                     for (std::size_t p : idx) {
                         acc += test(bs, p) ? 1.0 : 0.0;
                     }
                     for (std::size_t p : idx) {
                         reset(bs, p);
                     }
                 }
                 FmBench::consume(acc);
             }},
        },
        options.config);
}

FM_BENCH(DataStructure, DynamicBitSetBulkFlipCount_64K) {
    constexpr std::size_t bits = 65536;
    constexpr int rounds = 512;

    DynamicBitSet bs(bits);
    seedPattern(bs, bits, 0x13579BDFu);

    FmBench::runComparisonCase(
        "dynamic bitset setAll/resetAll/flipAll/popcount (64K bits)",
        static_cast<std::size_t>(rounds * 4),
        {
            {"fast_math", true, [&]() {
                 double acc = 0.0;
                 DynamicBitSet x = bs;
                 for (int r = 0; r < rounds; ++r) {
                     setAll(x);
                     acc += static_cast<double>(popcount(x));
                     flipAll(x);
                     acc += static_cast<double>(popcount(x));
                     resetAll(x);
                     acc += static_cast<double>(popcount(x));
                     flipAll(x);
                     acc += static_cast<double>(popcount(x));
                 }
                 FmBench::consume(acc);
             }},
        },
        options.config);
}

FM_BENCH(BitOps, DynamicThroughputScalarVsOptimized_4K) {
    constexpr std::size_t n = 4096;
    constexpr int rounds = 2048;

    DynamicBitSet a(n), b(n);
    for (std::size_t i = 0; i < n; i += 3) set(a, i);
    for (std::size_t i = 0; i < n; i += 5) set(b, i);

    FmBench::runComparisonCase(
        "dynamic bitset and+count+rank kernel (4K bits)",
        static_cast<std::size_t>(rounds * 3),
        {
            {"scalar", true, [&]() {
                 double acc = 0.0;
                 DynamicBitSet x = a;
                 for (int r = 0; r < rounds; ++r) {
                     bitwiseAnd(x, b);
                     acc += static_cast<double>(popcount(x));
                     acc += static_cast<double>(rank(x, static_cast<std::size_t>((r * 17) % static_cast<int>(n))));
                     flipRange(x, static_cast<std::size_t>((r * 13) % static_cast<int>(n)),
                               static_cast<std::size_t>(((r * 13) % static_cast<int>(n)) + 7));
                 }
                 FmBench::consume(acc);
             }},
            {"scalar_fused", true, [&]() {
                 double acc = 0.0;
                 DynamicBitSet x = a;
                 for (int r = 0; r < rounds; ++r) {
                     acc += static_cast<double>(bitwiseAndCount(x, b));
                     acc += static_cast<double>(rank(x, static_cast<std::size_t>((r * 17) % static_cast<int>(n))));
                     flipRange(x, static_cast<std::size_t>((r * 13) % static_cast<int>(n)),
                               static_cast<std::size_t>(((r * 13) % static_cast<int>(n)) + 7));
                 }
                 FmBench::consume(acc);
             }},
#if defined(__AVX2__) || defined(__SSE4_1__)
            {"optimized", true, [&]() {
                 double acc = 0.0;
                 DynamicBitSet x = a;
                 for (int r = 0; r < rounds; ++r) {
                     acc += static_cast<double>(bitwiseAndCountOptimized(x, b));
                     acc += static_cast<double>(rankOptimized(x, static_cast<std::size_t>((r * 17) % static_cast<int>(n))));
                     flipRangeOptimized(x, static_cast<std::size_t>((r * 13) % static_cast<int>(n)),
                                        static_cast<std::size_t>(((r * 13) % static_cast<int>(n)) + 7));
                 }
                 FmBench::consume(acc);
             }},
#else
            {"optimized", false, {}},
#endif
        },
        options.config);
}

FM_BENCH(BitOps, DynamicAndPopcountLarge_64K) {
    constexpr std::size_t n = 65536;
    constexpr int rounds = 384;

    DynamicBitSet a(n), b(n);
    seedPattern(a, n, 0x10203040u);
    seedPattern(b, n, 0x55667788u);

    FmBench::runComparisonCase(
        "dynamic bitset and+popcount (64K bits)",
        static_cast<std::size_t>(rounds * 2),
        {
            {"scalar", true, [&]() {
                 double acc = 0.0;
                 DynamicBitSet x = a;
                 for (int r = 0; r < rounds; ++r) {
                     bitwiseAnd(x, b);
                     acc += static_cast<double>(popcount(x));
                     flipAll(x);
                 }
                 FmBench::consume(acc);
             }},
            {"scalar_fused", true, [&]() {
                 double acc = 0.0;
                 DynamicBitSet x = a;
                 for (int r = 0; r < rounds; ++r) {
                     acc += static_cast<double>(bitwiseAndCount(x, b));
                     flipAll(x);
                 }
                 FmBench::consume(acc);
             }},
#if defined(__AVX2__) || defined(__SSE4_1__)
            {"optimized", true, [&]() {
                 double acc = 0.0;
                 DynamicBitSet x = a;
                 for (int r = 0; r < rounds; ++r) {
                     acc += static_cast<double>(bitwiseAndCountOptimized(x, b));
                     flipAll(x);
                 }
                 FmBench::consume(acc);
             }},
#else
            {"optimized", false, {}},
#endif
        },
        options.config);
}

FM_BENCH(StdBitsetCmp, DynamicVsStdBitset_AndCount_4K) {
    constexpr std::size_t n = 4096;
    constexpr int rounds = 3072;

    DynamicBitSet a_dyn(n), b_dyn(n);
    for (std::size_t i = 0; i < n; i += 3) set(a_dyn, i);
    for (std::size_t i = 0; i < n; i += 5) set(b_dyn, i);

    std::bitset<n> a_std, b_std;
    seedPatternStd(a_std, 0x10293847u);
    seedPatternStd(b_std, 0x55667788u);

    FmBench::runComparisonCase(
        "dynamic vs std::bitset and+count (4K bits)",
        static_cast<std::size_t>(rounds * 2),
        {
            {"dynamic_bitset", true, [&]() {
                 double acc = 0.0;
                 DynamicBitSet x = a_dyn;
                 for (int r = 0; r < rounds; ++r) {
                     acc += static_cast<double>(bitwiseAndCountOptimized(x, b_dyn));
                     flipRange(x, static_cast<std::size_t>((r * 13) & (n - 1)),
                               static_cast<std::size_t>(((r * 13) & (n - 1)) + 9));
                 }
                 FmBench::consume(acc);
             }},
            {"std::bitset", true, [&]() {
                 double acc = 0.0;
                 std::bitset<n> x = a_std;
                 for (int r = 0; r < rounds; ++r) {
                     x &= b_std;
                     acc += static_cast<double>(x.count());

                     const std::size_t base = static_cast<std::size_t>((r * 13) & (n - 1));
                     for (std::size_t k = 0; k < 9; ++k) {
                         x.flip((base + k) & (n - 1));
                     }
                 }
                 FmBench::consume(acc);
             }},
        },
        options.config);
}

FM_BENCH(StdBitsetCmp, DynamicVsStdBitset_AndCount_64K) {
    constexpr std::size_t n = 65536;
    constexpr int rounds = 512;

    DynamicBitSet a_dyn(n), b_dyn(n);
    seedPattern(a_dyn, n, 0xA1B2C3D4u);
    seedPattern(b_dyn, n, 0x0F1E2D3Cu);

    std::bitset<n> a_std, b_std;
    seedPatternStd(a_std, 0xA1B2C3D4u);
    seedPatternStd(b_std, 0x0F1E2D3Cu);

    FmBench::runComparisonCase(
        "dynamic vs std::bitset and+count (64K bits)",
        static_cast<std::size_t>(rounds * 2),
        {
            {"dynamic_bitset", true, [&]() {
                 double acc = 0.0;
                 DynamicBitSet x = a_dyn;
                 for (int r = 0; r < rounds; ++r) {
                     acc += static_cast<double>(bitwiseAndCountOptimized(x, b_dyn));
                     flipAll(x);
                 }
                 FmBench::consume(acc);
             }},
            {"std::bitset", true, [&]() {
                 double acc = 0.0;
                 std::bitset<n> x = a_std;
                 for (int r = 0; r < rounds; ++r) {
                     x &= b_std;
                     acc += static_cast<double>(x.count());
                     x.flip();
                 }
                 FmBench::consume(acc);
             }},
        },
        options.config);
}

FM_BENCH(StdBitsetCmp, DynamicVsStdBitset_RandomBitAccess_16K) {
    constexpr std::size_t n = 16384;
    constexpr int rounds = 256;
    constexpr std::size_t indices = 4096;

    const auto idx = makeIndices(indices, n, 0xD00DFEEDu);
    DynamicBitSet dyn(n);
    std::bitset<n> st;

    FmBench::runComparisonCase(
        "dynamic vs std::bitset random set/test/reset (16K bits)",
        static_cast<std::size_t>(rounds * indices * 3),
        {
            {"dynamic_bitset", true, [&]() {
                 double acc = 0.0;
                 for (int r = 0; r < rounds; ++r) {
                     for (std::size_t p : idx) set(dyn, p);
                     for (std::size_t p : idx) acc += test(dyn, p) ? 1.0 : 0.0;
                     for (std::size_t p : idx) reset(dyn, p);
                 }
                 FmBench::consume(acc);
             }},
            {"std::bitset", true, [&]() {
                 double acc = 0.0;
                 for (int r = 0; r < rounds; ++r) {
                     for (std::size_t p : idx) st.set(p);
                     for (std::size_t p : idx) acc += st.test(p) ? 1.0 : 0.0;
                     for (std::size_t p : idx) st.reset(p);
                 }
                 FmBench::consume(acc);
             }},
        },
        options.config);
}
