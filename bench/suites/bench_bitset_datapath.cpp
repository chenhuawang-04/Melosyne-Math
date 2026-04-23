#include "bench/framework/benchmark_framework.h"

#include "fast_math/bit_ops.h"

#include <cstdint>
#include <random>
#include <vector>

namespace {

using namespace Melosyne;
using namespace Melosyne::BitOps;

template <typename BitsetT>
void seed_pattern(BitsetT& bs, std::size_t bits, uint32_t seed) {
    uint32_t x = seed;
    for (std::size_t i = 0; i < bits; ++i) {
        x = x * 1664525u + 1013904223u;
        if ((x >> 31) & 1u) {
            set(bs, i);
        }
    }
}

std::vector<std::size_t> make_indices(std::size_t count, std::size_t limit, uint32_t seed) {
    std::vector<std::size_t> out(count);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::size_t> dist(0, limit - 1);
    for (auto& v : out) {
        v = dist(rng);
    }
    return out;
}

} // namespace

FM_BENCH(DataStructure, DynamicBitSetLifecycle_4K) {
    constexpr std::size_t BITS = 4096;
    constexpr int ITERS = 2048;

    fmbench::run_comparison_case(
        "dynamic bitset lifecycle (4K bits)",
        static_cast<std::size_t>(ITERS),
        {
            {"fast_math", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < ITERS; ++i) {
                     DynamicBitSet bs(BITS);
                     const std::size_t p0 = static_cast<std::size_t>((i * 17) & (BITS - 1));
                     const std::size_t p1 = static_cast<std::size_t>((i * 47) & (BITS - 1));
                     bs.set(p0).set(p1);
                     acc += bs.test(p0) ? 1.0 : 0.0;
                     acc += static_cast<double>(bs.wordCount());
                 }
                 fmbench::consume(acc);
             }},
        },
        options.config);
}

FM_BENCH(DataStructure, DynamicBitSetCopyMove_64K) {
    constexpr std::size_t BITS = 65536;
    constexpr int ITERS = 192;

    DynamicBitSet src(BITS);
    seed_pattern(src, BITS, 0xA1B2C3D4u);

    fmbench::run_comparison_case(
        "dynamic bitset copy/move (64K bits)",
        static_cast<std::size_t>(ITERS * 4),
        {
            {"fast_math", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < ITERS; ++i) {
                     DynamicBitSet c0(src);               // copy ctor
                     DynamicBitSet c1(std::move(c0));     // move ctor
                     DynamicBitSet c2(BITS);
                     c2 = c1;                             // copy assign
                     DynamicBitSet c3;
                     c3 = std::move(c2);                  // move assign
                     acc += static_cast<double>(test(c3, static_cast<std::size_t>((i * 73) & (BITS - 1))));
                 }
                 fmbench::consume(acc);
             }},
        },
        options.config);
}

FM_BENCH(DataStructure, DynamicBitSetBitAccessRandom_16K) {
    constexpr std::size_t BITS = 16384;
    constexpr int ROUNDS = 256;
    constexpr std::size_t INDICES = 4096;

    DynamicBitSet bs(BITS);
    const auto idx = make_indices(INDICES, BITS, 0xCAFEBABEu);

    fmbench::run_comparison_case(
        "dynamic bitset random set/reset/test (16K bits)",
        static_cast<std::size_t>(ROUNDS * INDICES * 3),
        {
            {"fast_math", true, [&]() {
                 double acc = 0.0;
                 for (int r = 0; r < ROUNDS; ++r) {
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
                 fmbench::consume(acc);
             }},
        },
        options.config);
}

FM_BENCH(DataStructure, DynamicBitSetBulkFlipCount_64K) {
    constexpr std::size_t BITS = 65536;
    constexpr int ROUNDS = 512;

    DynamicBitSet bs(BITS);
    seed_pattern(bs, BITS, 0x13579BDFu);

    fmbench::run_comparison_case(
        "dynamic bitset setAll/resetAll/flipAll/popcount (64K bits)",
        static_cast<std::size_t>(ROUNDS * 4),
        {
            {"fast_math", true, [&]() {
                 double acc = 0.0;
                 DynamicBitSet x = bs;
                 for (int r = 0; r < ROUNDS; ++r) {
                     setAll(x);
                     acc += static_cast<double>(popcount(x));
                     flipAll(x);
                     acc += static_cast<double>(popcount(x));
                     resetAll(x);
                     acc += static_cast<double>(popcount(x));
                     flipAll(x);
                     acc += static_cast<double>(popcount(x));
                 }
                 fmbench::consume(acc);
             }},
        },
        options.config);
}

FM_BENCH(BitOps, DynamicThroughputScalarVsOptimized_4K) {
    constexpr std::size_t N = 4096;
    constexpr int ROUNDS = 2048;

    DynamicBitSet a(N), b(N);
    for (std::size_t i = 0; i < N; i += 3) set(a, i);
    for (std::size_t i = 0; i < N; i += 5) set(b, i);

    fmbench::run_comparison_case(
        "dynamic bitset and+count+rank kernel (4K bits)",
        static_cast<std::size_t>(ROUNDS * 3),
        {
            {"scalar", true, [&]() {
                 double acc = 0.0;
                 DynamicBitSet x = a;
                 for (int r = 0; r < ROUNDS; ++r) {
                     bitwiseAnd(x, b);
                     acc += static_cast<double>(popcount(x));
                     acc += static_cast<double>(rank(x, static_cast<std::size_t>((r * 17) % static_cast<int>(N))));
                     flipRange(x, static_cast<std::size_t>((r * 13) % static_cast<int>(N)),
                               static_cast<std::size_t>(((r * 13) % static_cast<int>(N)) + 7));
                 }
                 fmbench::consume(acc);
             }},
#if defined(__AVX2__) || defined(__SSE4_1__)
            {"optimized", true, [&]() {
                 double acc = 0.0;
                 DynamicBitSet x = a;
                 for (int r = 0; r < ROUNDS; ++r) {
                     bitwiseAndOptimized(x, b);
                     acc += static_cast<double>(popcountOptimized(x));
                     acc += static_cast<double>(rankOptimized(x, static_cast<std::size_t>((r * 17) % static_cast<int>(N))));
                     flipRangeOptimized(x, static_cast<std::size_t>((r * 13) % static_cast<int>(N)),
                                        static_cast<std::size_t>(((r * 13) % static_cast<int>(N)) + 7));
                 }
                 fmbench::consume(acc);
             }},
#else
            {"optimized", false, {}},
#endif
        },
        options.config);
}

FM_BENCH(BitOps, DynamicAndPopcountLarge_64K) {
    constexpr std::size_t N = 65536;
    constexpr int ROUNDS = 384;

    DynamicBitSet a(N), b(N);
    seed_pattern(a, N, 0x10203040u);
    seed_pattern(b, N, 0x55667788u);

    fmbench::run_comparison_case(
        "dynamic bitset and+popcount (64K bits)",
        static_cast<std::size_t>(ROUNDS * 2),
        {
            {"scalar", true, [&]() {
                 double acc = 0.0;
                 DynamicBitSet x = a;
                 for (int r = 0; r < ROUNDS; ++r) {
                     bitwiseAnd(x, b);
                     acc += static_cast<double>(popcount(x));
                     flipAll(x);
                 }
                 fmbench::consume(acc);
             }},
#if defined(__AVX2__) || defined(__SSE4_1__)
            {"optimized", true, [&]() {
                 double acc = 0.0;
                 DynamicBitSet x = a;
                 for (int r = 0; r < ROUNDS; ++r) {
                     bitwiseAndOptimized(x, b);
                     acc += static_cast<double>(popcountOptimized(x));
                     flipAll(x);
                 }
                 fmbench::consume(acc);
             }},
#else
            {"optimized", false, {}},
#endif
        },
        options.config);
}
