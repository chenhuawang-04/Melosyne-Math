import fast_math.all;

#include <cassert>
#include <cmath>

int main() {
    using namespace MMath;

    Vec2 a{1.0f, 2.0f};
    Vec2 b{3.0f, 5.0f};
    Vec2 c = vec2Add(a, b);
    assert(c.x == 4.0f && c.y == 7.0f);

    Vec3 v{3.0f, 0.0f, 4.0f};
    float len = vec3Length(v);
    assert(std::abs(len - 5.0f) < 1e-5f);

    Mat4 id = mat4Identity();
    Vec4 p{1.0f, 2.0f, 3.0f, 1.0f};
    Vec4 q = mat4MulVec4(id, p);
    assert(std::abs(q.x - p.x) < 1e-6f);
    assert(std::abs(q.y - p.y) < 1e-6f);
    assert(std::abs(q.z - p.z) < 1e-6f);
    assert(std::abs(q.w - p.w) < 1e-6f);

    float e = MMath::exp(1.0f);
    assert(e > 2.70f && e < 2.73f);

    SinCos sc{};
    sincos(Angle{0.0f}, &sc);
    assert(std::abs(sc.sin) < 1e-6f);
    assert(std::abs(sc.cos - 1.0f) < 1e-6f);

    using namespace Melosyne;

    BitSet<128> bits{};
    BitOps::set(bits, 9);
    assert(BitOps::test(bits, 9));
    assert(BitOps::popcount(bits) == 1);

    return 0;
}
