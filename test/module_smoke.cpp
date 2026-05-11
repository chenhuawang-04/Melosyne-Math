import fast_math.all;

int moduleParitySmoke();

int main() {
    using namespace MMath;
    auto abs_diff = [](float a, float b) {
        float d = a - b;
        return d < 0.0f ? -d : d;
    };

    Vec2 a{1.0f, 2.0f};
    Vec2 b{3.0f, 5.0f};
    Vec2 c = vec2Add(a, b);
    if (!(c.x == 4.0f && c.y == 7.0f)) return 1;

    Vec3 v{3.0f, 0.0f, 4.0f};
    float len = vec3Length(v);
    if (abs_diff(len, 5.0f) >= 1e-5f) return 2;

    Mat4 id = mat4Identity();
    Vec4 p{1.0f, 2.0f, 3.0f, 1.0f};
    Vec4 q = mat4MultiplyVec4(id, p);
    if (abs_diff(q.x, p.x) >= 1e-6f) return 3;
    if (abs_diff(q.y, p.y) >= 1e-6f) return 4;
    if (abs_diff(q.z, p.z) >= 1e-6f) return 5;
    if (abs_diff(q.w, p.w) >= 1e-6f) return 6;

    auto inv4 = mat4InverseChecked(id);
    if (!inv4.invertible) return 7;
    Vec4 qi = mat4MultiplyVec4(inv4.value, p);
    if (abs_diff(qi.x, p.x) >= 1e-6f) return 8;
    if (abs_diff(qi.y, p.y) >= 1e-6f) return 9;

    Mat3 affine = mat3FromTranslation(Vec2{2.0f, -1.0f});
    auto inv3 = mat3InverseAffineChecked(affine);
    if (!inv3.invertible) return 10;
    Vec2 restored = mat3TransformPoint(inv3.value, mat3TransformPoint(affine, a));
    if (abs_diff(restored.x, a.x) >= 1e-6f) return 11;
    if (abs_diff(restored.y, a.y) >= 1e-6f) return 12;

    float e = MMath::exp(1.0f);
    if (!(e > 2.70f && e < 2.73f)) return 13;

    float exp_values[2]{0.0f, 1.0f};
    expArray(exp_values, 2);
    if (abs_diff(exp_values[0], 1.0f) >= 1e-5f) return 14;
    if (abs_diff(exp_values[1], e) >= 5e-2f) return 15;

    float pow_base[3]{-2.0f, 0.0f, 4.0f};
    float pow_exp[3]{3.0f, -1.0f, 0.5f};
    float pow_out[3]{};
    powArray(pow_base, pow_exp, pow_out, 3);
    if (abs_diff(pow_out[0], -8.0f) >= 1e-5f) return 16;
    if (!(pow_out[1] > 1.0e30f)) return 17;
    if (abs_diff(pow_out[2], 2.0f) >= 5e-2f) return 18;

    SinCos sc{};
    sincos(Angle{0.0f}, &sc);
    if (abs_diff(sc.sin, 0.0f) >= 1e-6f) return 19;
    if (abs_diff(sc.cos, 1.0f) >= 1e-6f) return 20;

    BitSet<128> bits{};
    BitOps::set(bits, 9);
    if (!BitOps::test(bits, 9)) return 21;
    if (BitOps::popcount(bits) != 1) return 22;

    if (int parity = moduleParitySmoke(); parity != 0) return 100 + parity;

    return 0;
}
