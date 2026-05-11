#ifndef FAST_MATH_BITSET_TYPES_NAMESPACE_BRIDGE_DEFINED
namespace MMath {

template<std::size_t N>
using BitSet = ::Melosyne::BitSet<N>;

using DynamicBitSet = ::Melosyne::DynamicBitSet;
using BitSetView = ::Melosyne::BitSetView;
using ConstBitSetView = ::Melosyne::ConstBitSetView;

} // namespace MMath

#define FAST_MATH_BITSET_TYPES_NAMESPACE_BRIDGE_DEFINED
#endif

#ifndef FAST_MATH_BITOPS_NAMESPACE_ALIAS_DEFINED
namespace MMath {
namespace BitOps = ::Melosyne::BitOps;
} // namespace MMath

#define FAST_MATH_BITOPS_NAMESPACE_ALIAS_DEFINED
#endif
