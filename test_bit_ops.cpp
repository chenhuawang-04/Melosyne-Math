// ============================================================================
// BitOps Comprehensive Unit Tests
// ============================================================================

#include <fast_math/bit_ops.h>
#include <fast_math/bitset_static.h>
#include <fast_math/bitset_dynamic.h>
#include <fast_math/bitset_view.h>
#include <iostream>
#include <cassert>

using namespace Melosyne;

// Helper macro for zero-initialized BitSet
#define BITSET_ZERO(size, name) \
    BitSet<size> name{}; \
    for (auto& w : name.words) w = 0

// Test counter
int tests_passed = 0;
int tests_total = 0;

#define TEST(name) \
    ++tests_total; \
    std::cout << "Testing " << #name << "... "; \
    if (test_##name()) { \
        ++tests_passed; \
        std::cout << "✅ PASSED\n"; \
    } else { \
        std::cout << "❌ FAILED\n"; \
    }

// ============================================================================
// Core Operations Tests
// ============================================================================

bool test_bitwiseAnd() {
    BITSET_ZERO(256, a); BITSET_ZERO(256, b);
    BitOps::set(a, 10); BitOps::set(a, 50); BitOps::set(a, 100);
    BitOps::set(b, 50); BitOps::set(b, 100); BitOps::set(b, 150);

    BitOps::bitwiseAnd(a, b);

    return BitOps::test(a, 50) && BitOps::test(a, 100) && !BitOps::test(a, 10) && !BitOps::test(a, 150);
}

bool test_bitwiseOr() {
    BITSET_ZERO(256, a); BITSET_ZERO(256, b);
    BitOps::set(a, 10); BitOps::set(a, 50);
    BitOps::set(b, 100); BitOps::set(b, 150);

    BitOps::bitwiseOr(a, b);

    return BitOps::test(a, 10) && BitOps::test(a, 50) && BitOps::test(a, 100) && BitOps::test(a, 150);
}

bool test_bitwiseXor() {
    BITSET_ZERO(256, a); BITSET_ZERO(256, b);
    BitOps::set(a, 10); BitOps::set(a, 50); BitOps::set(a, 100);
    BitOps::set(b, 50); BitOps::set(b, 100); BitOps::set(b, 150);

    BitOps::bitwiseXor(a, b);

    return BitOps::test(a, 10) && !BitOps::test(a, 50) && !BitOps::test(a, 100) && BitOps::test(a, 150);
}

bool test_bitwiseNot() {
    BitSet<64> a{}; for(auto& w : a.words) w=0;
    BitOps::set(a, 10); BitOps::set(a, 50);

    BitOps::bitwiseNot(a);

    return !BitOps::test(a, 10) && !BitOps::test(a, 50) && BitOps::test(a, 0) && BitOps::test(a, 63);
}

bool test_bitwiseAndNot() {
    BITSET_ZERO(256, a); BITSET_ZERO(256, b);
    BitOps::set(a, 10); BitOps::set(a, 50); BitOps::set(a, 100);
    BitOps::set(b, 50); BitOps::set(b, 150);

    BitOps::bitwiseAndNot(a, b);  // a &= ~b

    return BitOps::test(a, 10) && !BitOps::test(a, 50) && BitOps::test(a, 100) && !BitOps::test(a, 150);
}

// ============================================================================
// Count Operations Tests
// ============================================================================

bool test_popcount() {
    BitSet<256> a{}; for(auto& w : a.words) w=0;
    BitOps::set(a, 10); BitOps::set(a, 50); BitOps::set(a, 100);

    std::size_t count = BitOps::popcount(a);
    return count == 3;
}

bool test_hammingDistance() {
    BITSET_ZERO(256, a); BITSET_ZERO(256, b);
    BitOps::set(a, 10); BitOps::set(a, 50); BitOps::set(a, 100);
    BitOps::set(b, 50); BitOps::set(b, 100); BitOps::set(b, 150);

    std::size_t dist = BitOps::hammingDistance(a, b);
    return dist == 2;  // Bits 10 and 150 differ
}

bool test_any_none_all() {
    BitSet<64> a{}; for(auto& w : a.words) w=0;

    if (!BitOps::none(a)) return false;

    BitOps::set(a, 10);
    if (!BitOps::any(a) || BitOps::none(a)) return false;

    BitOps::setAll(a);
    if (!BitOps::all(a)) return false;

    return true;
}

bool test_jaccardSimilarity() {
    BITSET_ZERO(256, a); BITSET_ZERO(256, b);
    BitOps::set(a, 10); BitOps::set(a, 50); BitOps::set(a, 100);
    BitOps::set(b, 50); BitOps::set(b, 100); BitOps::set(b, 150);

    float similarity = BitOps::jaccardSimilarity(a, b);
    // Intersection: 2 bits (50, 100)
    // Union: 4 bits (10, 50, 100, 150)
    // Jaccard = 2/4 = 0.5
    return std::abs(similarity - 0.5f) < 0.001f;
}

bool test_isSubsetOf() {
    BITSET_ZERO(256, a); BITSET_ZERO(256, b);
    BitOps::set(a, 10); BitOps::set(a, 50);
    BitOps::set(b, 10); BitOps::set(b, 50); BitOps::set(b, 100);

    return BitOps::isSubsetOf(a, b) && !BitOps::isSubsetOf(b, a);
}

// ============================================================================
// Scan Operations Tests
// ============================================================================

bool test_findFirst() {
    BitSet<256> a{}; for(auto& w : a.words) w=0;
    BitOps::set(a, 10); BitOps::set(a, 50); BitOps::set(a, 100);

    std::size_t first = BitOps::findFirst(a);
    return first == 10;
}

bool test_findLast() {
    BitSet<256> a{}; for(auto& w : a.words) w=0;
    BitOps::set(a, 10); BitOps::set(a, 50); BitOps::set(a, 100);

    std::size_t last = BitOps::findLast(a);
    return last == 100;
}

bool test_findNext() {
    BitSet<256> a{}; for(auto& w : a.words) w=0;
    BitOps::set(a, 10); BitOps::set(a, 50); BitOps::set(a, 100);

    std::size_t pos = BitOps::findFirst(a);
    pos = BitOps::findNext(a, pos);
    return pos == 50;
}

bool test_findPrev() {
    BitSet<256> a{}; for(auto& w : a.words) w=0;
    BitOps::set(a, 10); BitOps::set(a, 50); BitOps::set(a, 100);

    std::size_t pos = BitOps::findLast(a);
    pos = BitOps::findPrev(a, pos);
    return pos == 50;
}

bool test_BitPositionIterator() {
    BitSet<256> a{}; for(auto& w : a.words) w=0;
    BitOps::set(a, 10); BitOps::set(a, 50); BitOps::set(a, 100);

    std::size_t expected[] = {10, 50, 100};
    std::size_t i = 0;

    for (std::size_t pos : BitOps::BitPositionIterator(a)) {
        if (i >= 3 || pos != expected[i]) return false;
        ++i;
    }

    return i == 3;
}

bool test_select() {
    BitSet<256> a{}; for(auto& w : a.words) w=0;
    BitOps::set(a, 10); BitOps::set(a, 50); BitOps::set(a, 100);

    return BitOps::select(a, 0) == 10 &&
           BitOps::select(a, 1) == 50 &&
           BitOps::select(a, 2) == 100;
}

bool test_rank() {
    BitSet<256> a{}; for(auto& w : a.words) w=0;
    BitOps::set(a, 10); BitOps::set(a, 50); BitOps::set(a, 100);

    return BitOps::rank(a, 0) == 0 &&
           BitOps::rank(a, 11) == 1 &&
           BitOps::rank(a, 51) == 2 &&
           BitOps::rank(a, 101) == 3;
}

// ============================================================================
// Range Operations Tests
// ============================================================================

bool test_setRange() {
    BitSet<256> a{}; for(auto& w : a.words) w=0;

    BitOps::setRange(a, 10, 50);

    return BitOps::test(a, 10) && BitOps::test(a, 25) && BitOps::test(a, 49) && !BitOps::test(a, 9) && !BitOps::test(a, 50);
}

bool test_resetRange() {
    BitSet<256> a{}; for(auto& w : a.words) w=0;
    BitOps::setAll(a);

    BitOps::resetRange(a, 10, 50);

    return !BitOps::test(a, 10) && !BitOps::test(a, 25) && !BitOps::test(a, 49) && BitOps::test(a, 9) && BitOps::test(a, 50);
}

bool test_flipRange() {
    BitSet<256> a{}; for(auto& w : a.words) w=0;
    BitOps::set(a, 10); BitOps::set(a, 30); BitOps::set(a, 100);

    BitOps::flipRange(a, 20, 40);

    return BitOps::test(a, 10) && !BitOps::test(a, 30) && BitOps::test(a, 25) && BitOps::test(a, 100);
}

bool test_popcountRange() {
    BitSet<256> a{}; for(auto& w : a.words) w=0;
    BitOps::set(a, 10); BitOps::set(a, 30); BitOps::set(a, 50); BitOps::set(a, 100);

    std::size_t count = BitOps::popcountRange(a, 0, 60);
    return count == 3;  // Bits 10, 30, 50
}

// ============================================================================
// Advanced Operations Tests
// ============================================================================

bool test_reverseBits() {
    uint8_t v = 0b10110100;
    uint8_t r = BitOps::reverseBits8(v);
    return r == 0b00101101;
}

bool test_reverseBitset() {
    BitSet<64> a{}; for(auto& w : a.words) w=0;
    BitOps::set(a, 0); BitOps::set(a, 10); BitOps::set(a, 63);

    BitOps::reverse(a);

    return BitOps::test(a, 63) && BitOps::test(a, 53) && BitOps::test(a, 0) && !BitOps::test(a, 10);
}

bool test_parity() {
    BitSet<64> a{}; for(auto& w : a.words) w=0;
    BitOps::set(a, 10); BitOps::set(a, 20); BitOps::set(a, 30);  // 3 bits (odd)

    bool odd = BitOps::parity(a);
    BitOps::set(a, 40);  // 4 bits (even)
    bool even = !BitOps::parity(a);

    return odd && even;
}

bool test_binaryGrayConversion() {
    uint64_t binary = 0b10110;
    uint64_t gray = BitOps::binaryToGray(binary);
    uint64_t back = BitOps::grayToBinary(gray);

    return back == binary;
}

bool test_mortonCode() {
    uint32_t x = 0b1011;
    uint32_t y = 0b1101;

    uint64_t morton = BitOps::interleaveBits32(x, y);

    uint32_t x2, y2;
    BitOps::deinterleaveBits(morton, x2, y2);

    return x == x2 && y == y2;
}

// ============================================================================
// Cross-compatibility Test (BitSet<N> and DynamicBitSet)
// ============================================================================

bool test_staticDynamicCompatibility() {
    BitSet<256> static_bs{}; for(auto& w : static_bs.words) w=0;
    DynamicBitSet dynamic_bs(256);

    BitOps::set(static_bs, 10); BitOps::set(static_bs, 50);
    dynamic_bs.set(50); dynamic_bs.set(100);

    // Unified interface: works on both!
    BitOps::bitwiseOr(static_bs, dynamic_bs);

    return BitOps::test(static_bs, 10) && BitOps::test(static_bs, 50) && BitOps::test(static_bs, 100);
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "BitOps Comprehensive Unit Tests\n";
    std::cout << "========================================\n\n";

    // Core operations
    TEST(bitwiseAnd);
    TEST(bitwiseOr);
    TEST(bitwiseXor);
    TEST(bitwiseNot);
    TEST(bitwiseAndNot);

    // Count operations
    TEST(popcount);
    TEST(hammingDistance);
    TEST(any_none_all);
    TEST(jaccardSimilarity);
    TEST(isSubsetOf);

    // Scan operations
    TEST(findFirst);
    TEST(findLast);
    TEST(findNext);
    TEST(findPrev);
    TEST(BitPositionIterator);
    TEST(select);
    TEST(rank);

    // Range operations
    TEST(setRange);
    TEST(resetRange);
    TEST(flipRange);
    TEST(popcountRange);

    // Advanced operations
    TEST(reverseBits);
    TEST(reverseBitset);
    TEST(parity);
    TEST(binaryGrayConversion);
    TEST(mortonCode);

    // Compatibility
    TEST(staticDynamicCompatibility);

    std::cout << "\n========================================\n";
    std::cout << "Results: " << tests_passed << "/" << tests_total << " tests passed\n";
    std::cout << "========================================\n";

    if (tests_passed == tests_total) {
        std::cout << "✅ ALL TESTS PASSED!\n";
        return 0;
    } else {
        std::cout << "❌ " << (tests_total - tests_passed) << " tests failed\n";
        return 1;
    }
}
