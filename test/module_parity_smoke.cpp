import fast_math.types;
import fast_math.common;
import fast_math.sqrt;
import fast_math.lerp;
import fast_math.power;
import fast_math.trig;
import fast_math.vec2;
import fast_math.vec3;
import fast_math.vec4;
import fast_math.mat3;
import fast_math.mat4;
import fast_math.bitset_view;
import fast_math.bitset_dynamic;
import fast_math.bit_ops;
import fast_math.bit_ops_core;
import fast_math.bit_ops_count;
import fast_math.bit_ops_scan;
import fast_math.bit_ops_scan_simd;
import fast_math.bit_ops_range;
import fast_math.bit_ops_advanced;
import fast_math.bit_ops_advanced_simd;

int moduleParitySmoke() {
    auto abs_diff = [](float a, float b) {
        float d = a - b;
        return d < 0.0f ? -d : d;
    };

    float clamped[4]{-1.0f, 0.25f, 2.0f, 0.75f};
    MMath::clampArray(clamped, 0.0f, 1.0f, 4);
    if (clamped[0] != 0.0f || clamped[1] != 0.25f || clamped[2] != 1.0f || clamped[3] != 0.75f) {
        return 1;
    }

    float roots[4]{0.0f, 1.0f, 4.0f, 9.0f};
    MMath::sqrtArray(roots, 4);
    if (roots[0] != 0.0f || roots[1] != 1.0f || roots[2] != 2.0f || roots[3] != 3.0f) {
        return 2;
    }

    float lerp_out[3]{};
    float lerp_a[3]{0.0f, 1.0f, 2.0f};
    float lerp_b[3]{2.0f, 3.0f, 4.0f};
    MMath::lerpArray(lerp_a, lerp_b, 0.5f, lerp_out, 3);
    if (lerp_out[0] != 1.0f || lerp_out[1] != 2.0f || lerp_out[2] != 3.0f) {
        return 3;
    }

    constexpr float half_pi = 1.570796326794897f;
    float trig_angles[2]{0.0f, half_pi};
    float out_sin[2]{};
    float out_cos[2]{};
    MMath::sincosArray(trig_angles, out_sin, out_cos, 2);
    if (abs_diff(out_sin[0], 0.0f) >= 1e-6f || abs_diff(out_cos[0], 1.0f) >= 1e-6f) {
        return 4;
    }
    if (abs_diff(out_sin[1], 1.0f) >= 1e-3f || abs_diff(out_cos[1], 0.0f) >= 1e-3f) {
        return 5;
    }

    MMath::Vec3 v{0.0f, 3.0f, 4.0f};
    if (abs_diff(MMath::vec3Length(v), 5.0f) >= 1e-6f) {
        return 6;
    }

    MMath::Mat3 translation = MMath::mat3FromTranslation(MMath::Vec2{2.0f, -1.0f});
    MMath::Vec2 moved = MMath::mat3TransformPoint(translation, MMath::Vec2{1.0f, 2.0f});
    if (abs_diff(moved.x, 3.0f) >= 1e-6f || abs_diff(moved.y, 1.0f) >= 1e-6f) {
        return 7;
    }

    MMath::Vec2 sum = MMath::vec2Add(MMath::Vec2{1.0f, 2.0f}, MMath::Vec2{3.0f, 4.0f});
    if (abs_diff(sum.x, 4.0f) >= 1e-6f || abs_diff(sum.y, 6.0f) >= 1e-6f) {
        return 8;
    }

    MMath::Vec4 normalized = MMath::vec4NormalizeSafe(MMath::Vec4{0.0f, 0.0f, 2.0f, 0.0f});
    if (abs_diff(normalized.z, 1.0f) >= 1e-6f || abs_diff(MMath::vec4Length(normalized), 1.0f) >= 1e-6f) {
        return 9;
    }

    if (abs_diff(MMath::pow(4.0f, 0.5f), 2.0f) >= 5e-2f) {
        return 10;
    }
    if (abs_diff(MMath::powi(-2.0f, 5), -32.0f) >= 1e-5f) {
        return 11;
    }

    auto ndc_depth = [&](const MMath::Vec4& clip) {
        return clip.z / clip.w;
    };

    constexpr float kNear = 0.5f;
    constexpr float kFar = 10.0f;
    const MMath::Mat4 proj = MMath::mat4Perspective(1.0f, 1.0f, kNear, kFar);
    const float near_depth = ndc_depth(MMath::mat4MultiplyVec4(proj, MMath::Vec4{0.0f, 0.0f, -kNear, 1.0f}));
    const float far_depth = ndc_depth(MMath::mat4MultiplyVec4(proj, MMath::Vec4{0.0f, 0.0f, -kFar, 1.0f}));
    if (abs_diff(near_depth, 0.0f) >= 1e-4f) {
        return 12;
    }
    if (abs_diff(far_depth, 1.0f) >= 1e-4f) {
        return 13;
    }

    Melosyne::BitSet<128> fixed{};
    Melosyne::BitSetView fixed_view(fixed);
    Melosyne::BitOps::set(fixed_view, 9);
    if (!Melosyne::BitOps::test(Melosyne::ConstBitSetView(fixed), 9)) {
        return 14;
    }

    Melosyne::DynamicBitSet dynamic(128);
    Melosyne::BitSetView dynamic_view(dynamic);
    Melosyne::BitOps::copy(dynamic_view, Melosyne::ConstBitSetView(fixed));
    if (Melosyne::BitOps::findFirst(Melosyne::ConstBitSetView(dynamic)) != 9) {
        return 15;
    }
    if (Melosyne::ConstBitSetView(dynamic).word_count != 2) {
        return 16;
    }

    Melosyne::DynamicBitSet managed(32);
    if (!managed.isSso()) {
        return 17;
    }
    managed.reserveBits(16384);
    if (managed.capacity() < 16384 || managed.isSso() || !managed.isAligned32()) {
        return 18;
    }
    managed.resize(257, true);
    if (managed.test(31) || !managed.test(32) || !managed.test(256)) {
        return 19;
    }
    if (managed.wordCount() != 5 || managed.data()[4] != 0x1ULL) {
        return 20;
    }
    managed.shrinkToFit();
    if (managed.capacityWords() != managed.wordCount()) {
        return 21;
    }

    Melosyne::BitOps::setRange(dynamic_view, 64, 68);
    if (!Melosyne::BitOps::testAll(Melosyne::ConstBitSetView(dynamic), 64, 68)) {
        return 22;
    }
    if (Melosyne::BitOps::popcountRange(Melosyne::ConstBitSetView(dynamic), 64, 68) != 4) {
        return 23;
    }
    if (Melosyne::BitOps::unionCount(Melosyne::ConstBitSetView(fixed), Melosyne::ConstBitSetView(dynamic)) != 5) {
        return 24;
    }
    if (Melosyne::BitOps::reverseBits64(0x0123456789ABCDEFULL) != 0xF7B3D591E6A2C480ULL) {
        return 25;
    }

    MMath::BitSet<64> canonical_bits{};
    MMath::BitOps::setRange(canonical_bits, 1, 5);
    if (!MMath::BitOps::testAll(canonical_bits, 1, 5)) {
        return 26;
    }
    if (MMath::BitOps::popcount(canonical_bits) != 4) {
        return 27;
    }
    if (MMath::BitOps::binaryToGray(10) != 15 || MMath::BitOps::grayToBinary(15) != 10) {
        return 28;
    }

    if (MMath::BitOps::selectOptimized(canonical_bits, 2) != 3) {
        return 29;
    }
    if (MMath::BitOps::rankOptimized(canonical_bits, 5) != 4) {
        return 30;
    }

    MMath::BitSet<70> rotated{};
    MMath::BitOps::set(rotated, 0);
    MMath::BitOps::set(rotated, 69);
    MMath::BitOps::rotateLeftOptimized(rotated, 1);
    if (!MMath::BitOps::test(rotated, 0) || !MMath::BitOps::test(rotated, 1)) {
        return 31;
    }

    MMath::BitSet<65> padded_only{};
    padded_only.words[1] = ~0ULL << 1;
    if (MMath::BitOps::findLast(padded_only) != padded_only.kBits) {
        return 32;
    }
    if (MMath::BitOps::parity(padded_only)) {
        return 33;
    }
    if (MMath::BitOps::nextPermutation(0) != 0) {
        return 34;
    }

    return 0;
}
