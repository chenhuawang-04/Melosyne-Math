#include "test/framework/test_framework.h"

#include "fast_math/bitset_static.h"
#include "fast_math/bitset_dynamic.h"
#include "fast_math/bitset_view.h"
#include "fast_math/bit_ops_core.h"
#include "fast_math/bit_ops_scan.h"

FM_TEST(BitOps, DirectSubheadersExposeCanonicalMMathNamespace) {
    MMath::BitSet<128> fixed{};
    MMath::BitOps::set(fixed, 7);
    MMath::BitOps::set(fixed, 65);

    FM_REQUIRE(MMath::BitOps::test(fixed, 7));
    FM_REQUIRE(MMath::BitOps::test(fixed, 65));
    FM_REQUIRE_EQ_U64(MMath::BitOps::findFirst(fixed), 7);

    MMath::DynamicBitSet dynamic(128);
    MMath::BitOps::copy(dynamic, fixed);
    FM_REQUIRE(MMath::BitOps::test(dynamic, 65));

    MMath::BitSetView mutable_view(dynamic);
    MMath::ConstBitSetView const_view(dynamic);
    FM_REQUIRE_EQ_U64(mutable_view.bit_count, 128);
    FM_REQUIRE_EQ_U64(const_view.word_count, 2);
}
