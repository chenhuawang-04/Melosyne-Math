#include "test/framework/test_framework.h"

#include "fast_math/bit_ops.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <vector>

namespace {

using namespace Melosyne;
using namespace Melosyne::BitOps;

std::vector<uint8_t> to_bits(ConstBitSetView view) {
    std::vector<uint8_t> out(view.bit_count, 0);
    for (std::size_t i = 0; i < view.bit_count; ++i) {
        out[i] = test(view, i) ? 1 : 0;
    }
    return out;
}

std::size_t ref_popcount(ConstBitSetView view) {
    std::size_t c = 0;
    for (std::size_t i = 0; i < view.word_count; ++i) {
        c += std::popcount(view.data[i]);
    }
    return c;
}

template <std::size_t N>
BitSet<N> make_pattern(uint32_t seed) {
    BitSet<N> bs{};
    uint32_t x = seed;
    for (std::size_t i = 0; i < N; ++i) {
        x = x * 1664525u + 1013904223u;
        if ((x >> 31) & 1u) {
            set(bs, i);
        }
    }
    return bs;
}

} // namespace

FM_TEST(BitOps, SingleBitAndCoreOps) {
    BitSet<256> bs{};

    set(bs, 5);
    set(bs, 127);
    set(bs, 255);
    FM_REQUIRE(test(bs, 5));
    FM_REQUIRE(test(bs, 127));
    FM_REQUIRE(test(bs, 255));

    reset(bs, 127);
    FM_REQUIRE(!test(bs, 127));

    flip(bs, 127);
    FM_REQUIRE(test(bs, 127));

    BitSet<256> other{};
    set(other, 5);
    set(other, 6);
    set(other, 255);

    BitSet<256> x = bs;
    bitwiseAnd(x, other);
    FM_REQUIRE(test(x, 5));
    FM_REQUIRE(!test(x, 127));
    FM_REQUIRE(test(x, 255));

    BitSet<256> y = bs;
    bitwiseOr(y, other);
    FM_REQUIRE(test(y, 6));
    FM_REQUIRE(test(y, 127));

    BitSet<256> z = bs;
    bitwiseXor(z, other);
    FM_REQUIRE(!test(z, 5));
    FM_REQUIRE(test(z, 6));

    BitSet<256> cpy{};
    copy(cpy, bs);
    FM_REQUIRE(equal(cpy, bs));
}

FM_TEST(BitOps, CountMetricsAgainstReference) {
    BitSet<1024> a = make_pattern<1024>(0x12345678u);
    BitSet<1024> b = make_pattern<1024>(0x9ABCDEF0u);

    const std::size_t pa = ref_popcount(a);
    const std::size_t pb = ref_popcount(b);

    FM_REQUIRE_EQ_U64(popcount(a), pa);
    FM_REQUIRE_EQ_U64(popcount(b), pb);

    const std::size_t hd = hammingDistance(a, b);

    BitSet<1024> t = a;
    bitwiseXor(t, b);
    FM_REQUIRE_EQ_U64(hd, popcount(t));

    const std::size_t uni = unionCount(a, b);
    const std::size_t inter = intersectionCount(a, b);

    FM_REQUIRE(uni >= inter);
    FM_REQUIRE_EQ_U64(intersectionCount(a, b), inter);
    FM_REQUIRE_NEAR(jaccardSimilarity(a, b), uni ? static_cast<float>(inter) / static_cast<float>(uni) : 1.0f, 1e-6f);
    FM_REQUIRE_NEAR(diceCoefficient(a, b), (pa + pb) ? (2.0f * inter) / (pa + pb) : 1.0f, 1e-6f);
}

FM_TEST(BitOps, ScanRankSelectIterator) {
    BitSet<130> bs{};
    std::vector<std::size_t> positions = {0, 1, 2, 64, 65, 66, 129};

    for (auto p : positions) set(bs, p);

    FM_REQUIRE_EQ_U64(findFirst(bs), 0);
    FM_REQUIRE_EQ_U64(findLast(bs), 129);
    FM_REQUIRE_EQ_U64(findNext(bs, 2), 64);
    FM_REQUIRE_EQ_U64(findPrev(bs, 66), 65);
    FM_REQUIRE_EQ_U64(findFirstZero(bs), 3);

    for (std::size_t k = 0; k < positions.size(); ++k) {
        FM_REQUIRE_EQ_U64(select(bs, k), positions[k]);
    }
    FM_REQUIRE_EQ_U64(select(bs, positions.size()), bs.kBits);

    FM_REQUIRE_EQ_U64(rank(bs, 0), 0);
    FM_REQUIRE_EQ_U64(rank(bs, 64), 3);
    FM_REQUIRE_EQ_U64(rank(bs, 130), positions.size());

    std::size_t idx = 0;
    for (std::size_t p : BitPositionIterator(bs)) {
        FM_REQUIRE_EQ_U64(p, positions[idx]);
        ++idx;
    }
    FM_REQUIRE_EQ_U64(idx, positions.size());
}

FM_TEST(BitOps, RangeAndAdvancedOps) {
    BitSet<256> bs{};

    setRange(bs, 10, 100);
    FM_REQUIRE(testAll(bs, 10, 100));
    FM_REQUIRE(testNone(bs, 0, 10));
    FM_REQUIRE(testNone(bs, 100, 256));
    FM_REQUIRE_EQ_U64(popcountRange(bs, 10, 100), 90);

    resetRange(bs, 20, 40);
    FM_REQUIRE(testNone(bs, 20, 40));
    FM_REQUIRE(testAny(bs, 10, 20));

    flipRange(bs, 30, 50);
    FM_REQUIRE(testAll(bs, 30, 40));

    FM_REQUIRE_EQ_U64(reverseBits64(0x0123456789ABCDEFULL), 0xF7B3D591E6A2C480ULL);
    FM_REQUIRE_EQ_U64(binaryToGray(10), 15);
    FM_REQUIRE_EQ_U64(grayToBinary(15), 10);
    FM_REQUIRE_EQ_U64(interleaveBits32(0b1010, 0b0101), 0b01100110);

    uint32_t x = 0, y = 0;
    deinterleaveBits(0b01100110, x, y);
    FM_REQUIRE_EQ_U64(x, 0b1010);
    FM_REQUIRE_EQ_U64(y, 0b0101);
}

FM_TEST(BitOps, DynamicStaticCompatibility) {
    BitSet<512> st = make_pattern<512>(0x11112222u);
    DynamicBitSet dy(512);

    copy(dy, st);
    FM_REQUIRE(equal(st, dy));

    DynamicBitSet dy2(512);
    copy(dy2, st);
    bitwiseAnd(dy, dy2);
    FM_REQUIRE(equal(dy, st));

    // andNot: a &= ~b
    flipAll(dy2);
    bitwiseAndNot(dy, dy2);
    FM_REQUIRE(equal(dy, st));

    DynamicBitSet dy3(512);
    copy(dy3, st);
    bitwiseAndNot(dy3, st);
    FM_REQUIRE_EQ_U64(popcount(dy3), 0);

    setAll(dy);
    FM_REQUIRE(all(dy));

    resetAll(dy);
    FM_REQUIRE(none(dy));
}

FM_TEST(BitOps, OptimizedApisMatchScalar) {
    BitSet<4096> a = make_pattern<4096>(0x77778888u);
    BitSet<4096> b = make_pattern<4096>(0x33334444u);

#if defined(__AVX2__) || defined(__SSE4_1__)
    BitSet<4096> scalar_and = a;
    BitSet<4096> opt_and = a;
    bitwiseAnd(scalar_and, b);
    bitwiseAndOptimized(opt_and, b);
    FM_REQUIRE(equal(scalar_and, opt_and));

    FM_REQUIRE_EQ_U64(popcountOptimized(a), popcount(a));
    FM_REQUIRE_EQ_U64(hammingDistanceOptimized(a, b), hammingDistance(a, b));
    FM_REQUIRE_EQ_U64(unionCountOptimized(a, b), unionCount(a, b));
    FM_REQUIRE_EQ_U64(intersectionCountOptimized(a, b), intersectionCount(a, b));

    FM_REQUIRE_EQ_U64(selectOptimized(a, 5), select(a, 5));
    FM_REQUIRE_EQ_U64(rankOptimized(a, 3000), rank(a, 3000));

    BitSet<4096> rs = a;
    BitSet<4096> ro = a;
    setRange(rs, 100, 3000);
    setRangeOptimized(ro, 100, 3000);
    FM_REQUIRE(equal(rs, ro));

    FM_REQUIRE_EQ_U64(popcountRangeOptimized(a, 50, 3800), popcountRange(a, 50, 3800));

    BitSet<4096> rev_s = a;
    BitSet<4096> rev_o = a;
    reverse(rev_s);
    reverseOptimized(rev_o);
    FM_REQUIRE(equal(rev_s, rev_o));
#else
    FM_REQUIRE(true);
#endif
}
