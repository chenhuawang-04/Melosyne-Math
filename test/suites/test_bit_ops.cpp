#include "test/framework/test_framework.h"

#include "fast_math/bit_ops.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <vector>

namespace {

using namespace MMath;
using namespace MMath::BitOps;

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

FM_TEST(BitOps, BitwiseAndCountMatchesSeparatePasses) {
    BitSet<257> a = make_pattern<257>(0x10293847u);
    BitSet<257> b = make_pattern<257>(0x55667788u);

    BitSet<257> separate = a;
    bitwiseAnd(separate, b);
    const std::size_t separate_count = popcount(separate);

    BitSet<257> fused = a;
    const std::size_t fused_count = bitwiseAndCount(fused, b);
    FM_REQUIRE_EQ_U64(fused_count, separate_count);
    FM_REQUIRE(equal(fused, separate));

#if defined(__AVX2__) || defined(__SSE4_1__)
    BitSet<257> fused_opt = a;
    const std::size_t fused_opt_count = bitwiseAndCountOptimized(fused_opt, b);
    FM_REQUIRE_EQ_U64(fused_opt_count, separate_count);
    FM_REQUIRE(equal(fused_opt, separate));
#endif

    BitSet<191> c = make_pattern<191>(0xCAFEBABEu);
    DynamicBitSet dyn_a(257);
    DynamicBitSet dyn_b(191);
    copy(dyn_a, a);
    copy(dyn_b, c);

    DynamicBitSet dyn_separate = dyn_a;
    bitwiseAnd(dyn_separate, dyn_b);
    const std::size_t dyn_separate_count = popcount(dyn_separate);

    DynamicBitSet dyn_fused = dyn_a;
    const std::size_t dyn_fused_count = bitwiseAndCount(dyn_fused, dyn_b);
    FM_REQUIRE_EQ_U64(dyn_fused_count, dyn_separate_count);
    FM_REQUIRE(equal(dyn_fused, dyn_separate));

#if defined(__AVX2__) || defined(__SSE4_1__)
    DynamicBitSet dyn_fused_opt = dyn_a;
    const std::size_t dyn_fused_opt_count = bitwiseAndCountOptimized(dyn_fused_opt, dyn_b);
    FM_REQUIRE_EQ_U64(dyn_fused_opt_count, dyn_separate_count);
    FM_REQUIRE(equal(dyn_fused_opt, dyn_separate));
#endif
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

FM_TEST(BitOps, ScanIgnoresPaddingAndBoundaryMasks) {
    BitSet<65> padded_only{};
    padded_only.words[1] = ~uint64_t{0} << 1;
    FM_REQUIRE_EQ_U64(findLast(padded_only), padded_only.kBits);

    BitSet<130> boundary{};
    set(boundary, 63);
    set(boundary, 64);
    FM_REQUIRE_EQ_U64(findPrev(boundary, 64), 63);

    BitSet<70> last_zero{};
    setRange(last_zero, 0, 70);
    reset(last_zero, 5);
    last_zero.words[1] |= (~uint64_t{0} << 6);
    FM_REQUIRE_EQ_U64(findLastZero(last_zero), 5);
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

FM_TEST(BitOps, AdvancedOpsHandlePartialWordsAndSentinels) {
    BitSet<70> rotated{};
    set(rotated, 0);
    set(rotated, 69);

    rotateLeft(rotated, 1);
    FM_REQUIRE(test(rotated, 0));
    FM_REQUIRE(test(rotated, 1));
    FM_REQUIRE_EQ_U64(popcount(rotated), 2);

    rotateRight(rotated, 1);
    FM_REQUIRE(test(rotated, 0));
    FM_REQUIRE(test(rotated, 69));

    BitSet<65> parity_padding{};
    parity_padding.words[1] = ~uint64_t{0} << 1;
    FM_REQUIRE(!parity(parity_padding));

    FM_REQUIRE_EQ_U64(nextPermutation(0), 0);
    FM_REQUIRE_EQ_U64(nextPermutation(0xE000000000000000ULL), 0);

    BitSet<80> scattered{};
    std::size_t indices[65]{};
    for (std::size_t i = 0; i < 65; ++i) {
        indices[i] = i;
    }
    scatterBits(scattered, ~uint64_t{0}, indices, 65);
    FM_REQUIRE(test(scattered, 0));
    FM_REQUIRE(test(scattered, 63));
    FM_REQUIRE(!test(scattered, 64));
    FM_REQUIRE_EQ_U64(gatherBits(scattered, indices, 65), ~uint64_t{0});
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

FM_TEST(BitOps, DynamicBitSetStorageManagement) {
    DynamicBitSet bits(32);
    FM_REQUIRE(bits.isSso());
    FM_REQUIRE_EQ_U64(bits.capacityWords(), DynamicBitSet::kLocalWords);

    bits.reserveBits(2048);
    FM_REQUIRE(bits.capacity() >= 2048);
    FM_REQUIRE(!bits.isSso());
    FM_REQUIRE(bits.isAligned32());

    bits.resize(257, true);
    FM_REQUIRE(!bits.test(31));
    FM_REQUIRE(bits.test(32));
    FM_REQUIRE(bits.test(256));
    FM_REQUIRE_EQ_U64(bits.wordCount(), 5);
    FM_REQUIRE_EQ_U64(bits.data()[4], 0x1ULL);

    bits.shrinkToFit();
    FM_REQUIRE_EQ_U64(bits.capacityWords(), bits.wordCount());
    FM_REQUIRE(bits.test(256));
}

FM_TEST(BitOps, OptimizedApisMatchScalar) {
    BitSet<8192> a = make_pattern<8192>(0x77778888u);
    BitSet<8192> b = make_pattern<8192>(0x33334444u);

#if defined(__AVX2__) || defined(__SSE4_1__)
    BitSet<8192> scalar_and = a;
    BitSet<8192> opt_and = a;
    bitwiseAnd(scalar_and, b);
    bitwiseAndOptimized(opt_and, b);
    FM_REQUIRE(equal(scalar_and, opt_and));

    BitSet<8192> fused_scalar = a;
    BitSet<8192> fused_opt = a;
    FM_REQUIRE_EQ_U64(bitwiseAndCount(fused_scalar, b), popcount(scalar_and));
    FM_REQUIRE_EQ_U64(bitwiseAndCountOptimized(fused_opt, b), popcount(scalar_and));
    FM_REQUIRE(equal(fused_scalar, scalar_and));
    FM_REQUIRE(equal(fused_opt, scalar_and));

    BitSet<8192> scalar_or = a;
    BitSet<8192> opt_or = a;
    bitwiseOr(scalar_or, b);
    bitwiseOrOptimized(opt_or, b);
    FM_REQUIRE(equal(scalar_or, opt_or));

    BitSet<8192> scalar_xor = a;
    BitSet<8192> opt_xor = a;
    bitwiseXor(scalar_xor, b);
    bitwiseXorOptimized(opt_xor, b);
    FM_REQUIRE(equal(scalar_xor, opt_xor));

    BitSet<8192> scalar_not = a;
    BitSet<8192> opt_not = a;
    bitwiseNot(scalar_not);
    bitwiseNotOptimized(opt_not);
    FM_REQUIRE(equal(scalar_not, opt_not));

    BitSet<8192> scalar_and_not = a;
    BitSet<8192> opt_and_not = a;
    bitwiseAndNot(scalar_and_not, b);
    bitwiseAndNotOptimized(opt_and_not, b);
    FM_REQUIRE(equal(scalar_and_not, opt_and_not));

    BitSet<8192> scalar_copy{};
    BitSet<8192> opt_copy{};
    copy(scalar_copy, a);
    copyOptimized(opt_copy, a);
    FM_REQUIRE(equal(scalar_copy, opt_copy));
    FM_REQUIRE(equalOptimized(opt_copy, a));

    FM_REQUIRE_EQ_U64(popcountOptimized(a), popcount(a));
    FM_REQUIRE_EQ_U64(hammingDistanceOptimized(a, b), hammingDistance(a, b));
    FM_REQUIRE_EQ_U64(unionCountOptimized(a, b), unionCount(a, b));
    FM_REQUIRE_EQ_U64(intersectionCountOptimized(a, b), intersectionCount(a, b));

    FM_REQUIRE_EQ_U64(selectOptimized(a, 5), select(a, 5));
    FM_REQUIRE_EQ_U64(rankOptimized(a, 3000), rank(a, 3000));

    BitSet<8192> rs = a;
    BitSet<8192> ro = a;
    setRange(rs, 100, 3000);
    setRangeOptimized(ro, 100, 3000);
    FM_REQUIRE(equal(rs, ro));

    FM_REQUIRE_EQ_U64(popcountRangeOptimized(a, 50, 3800), popcountRange(a, 50, 3800));

    BitSet<8192> rev_s = a;
    BitSet<8192> rev_o = a;
    reverse(rev_s);
    reverseOptimized(rev_o);
    FM_REQUIRE(equal(rev_s, rev_o));

    BitSet<8192> swap_s = a;
    BitSet<8192> swap_o = a;
    byteSwap(swap_s);
    byteSwapOptimized(swap_o);
    FM_REQUIRE(equal(swap_s, swap_o));

    BitSet<8192> rotl_s = a;
    BitSet<8192> rotl_o = a;
    rotateLeft(rotl_s, 137);
    rotateLeftOptimized(rotl_o, 137);
    FM_REQUIRE(equal(rotl_s, rotl_o));

    BitSet<8192> rotr_s = a;
    BitSet<8192> rotr_o = a;
    rotateRight(rotr_s, 211);
    rotateRightOptimized(rotr_o, 211);
    FM_REQUIRE(equal(rotr_s, rotr_o));
#else
    FM_REQUIRE(true);
#endif
}

FM_TEST(BitOps, CanonicalNamespaceAndLegacyCompatibility) {
    BitSet<128> bits{};
    BitOps::set(bits, 7);
    BitOps::set(bits, 64);

    FM_REQUIRE(BitOps::test(bits, 7));
    FM_REQUIRE(BitOps::test(bits, 64));
    FM_REQUIRE_EQ_U64(BitOps::popcount(bits), 2);

    DynamicBitSet dyn(128);
    BitOps::copy(dyn, bits);
    FM_REQUIRE(BitOps::equal(bits, dyn));

    Melosyne::BitSet<128> legacy{};
    Melosyne::BitOps::set(legacy, 17);
    FM_REQUIRE(Melosyne::BitOps::test(legacy, 17));

    BitSet<128> bridged{};
    BitOps::copy(bridged, legacy);
    FM_REQUIRE(BitOps::equal(bridged, legacy));
    FM_REQUIRE(BitOps::test(bridged, 17));
}
