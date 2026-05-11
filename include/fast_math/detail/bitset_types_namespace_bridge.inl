#ifndef FAST_MATH_BITSET_TYPES_NAMESPACE_BRIDGE_DEFINED
#define FAST_MATH_BITSET_TYPES_NAMESPACE_BRIDGE_DEFINED

namespace Melosyne {

template<std::size_t N>
struct BitSet;

class DynamicBitSet;
struct BitSetView;
struct ConstBitSetView;

} // namespace Melosyne

namespace MMath {

template<std::size_t N>
using BitSet = ::Melosyne::BitSet<N>;

using DynamicBitSet = ::Melosyne::DynamicBitSet;
using BitSetView = ::Melosyne::BitSetView;
using ConstBitSetView = ::Melosyne::ConstBitSetView;

} // namespace MMath

#endif // FAST_MATH_BITSET_TYPES_NAMESPACE_BRIDGE_DEFINED
