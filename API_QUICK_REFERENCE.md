# MMath API 快速参考

**版本**: 1.1.0 | **命名空间**: `MMath` | **语言**: C++17

---

## Common - 通用数学 (`#include "fast_math/common.h"`)

### 标量函数

| 函数 | 说明 |
|------|------|
| `float min(a, b)` | 两数最小值 |
| `float max(a, b)` | 两数最大值 |
| `float clamp(x, min, max)` | 限制在[min, max]范围 |
| `float abs(x)` | 绝对值 |
| `float sign(x)` | 符号 (-1, 0, +1) |
| `float saturate(x)` | 限制在[0, 1]范围 |
| `float copySign(x, y)` | x的大小，y的符号 |
| `bool nearEqual(a, b, eps)` | 近似相等 |
| `bool isZero(x, eps)` | 近似为零 |

### 数组函数 (SIMD)

| 函数 | 说明 |
|------|------|
| `void minArray(a, b, count)` | 逐元素最小值 |
| `void maxArray(a, b, count)` | 逐元素最大值 |
| `void clampArray(values, min, max, count)` | 数组限幅 |
| `void absArray(values, count)` | 数组绝对值 |
| `void saturateArray(values, count)` | 数组饱和到[0,1] |

---

## Sqrt - 平方根与倒数 (`#include "fast_math/sqrt.h"`)

### 标量函数

| 函数 | 说明 |
|------|------|
| `float sqrt(x)` | 平方根 (精确) |
| `float inverseSqrt(x)` | 1/sqrt(x) - 快速 (2x) |
| `float reciprocal(x)` | 1/x - 快速 |
| `float normalizeFactor(len_sq)` | 归一化因子，安全处理零长度 |
| `bool isSafeForDivision(x, eps)` | 检查是否可安全除法 |
| `float safeReciprocal(x, fallback, eps)` | 安全倒数 |

### 数组函数 (SIMD)

| 函数 | 说明 |
|------|------|
| `void sqrtArray(values, count)` | 批量平方根 |
| `void inverseSqrtArray(values, count)` | 批量倒数平方根 (7.5x) |
| `void reciprocalArray(values, count)` | 批量倒数 |

---

## Lerp - 插值与平滑曲线 (`#include "fast_math/lerp.h"`)

### 标量函数

| 函数 | 说明 |
|------|------|
| `float lerp(a, b, t)` | 线性插值 a + t*(b-a) |
| `float inverseLerp(a, b, value)` | 逆插值，返回参数t |
| `float remap(val, in_min, in_max, out_min, out_max)` | 重映射范围 |
| `float smoothstep(edge0, edge1, t)` | 三次Hermite平滑 (S曲线) |
| `float smootherstep(edge0, edge1, t)` | 五次Hermite平滑 (更柔和) |
| `float lerpClamped(a, b, t)` | 限制t到[0,1]的插值 |
| `float lerpStepped(a, b, t, steps)` | 阶梯式插值 |
| `bool isInRange(value, a, b, eps)` | 检查是否在范围内 |

### 数组函数 (SIMD)

| 函数 | 说明 |
|------|------|
| `void lerpArray(a, b, t, result, count)` | 批量线性插值 |
| `void smoothstepArray(edge0, edge1, values, count)` | 批量smoothstep |
| `void smootherstepArray(edge0, edge1, values, count)` | 批量smootherstep |
| `void remapArray(input, in_min, in_max, out_min, out_max, output, count)` | 批量重映射 |

---

## Vec2 - 2D向量 (`#include "fast_math/vec2.h"`)

### 算术运算

| 函数 | 说明 |
|------|------|
| `Vec2 vec2Add(a, b)` | 向量加法 |
| `Vec2 vec2Sub(a, b)` | 向量减法 |
| `Vec2 vec2Mul(a, b)` | 分量乘法 |
| `Vec2 vec2Div(a, b)` | 分量除法 |
| `Vec2 vec2Scale(v, s)` | 标量缩放 |
| `Vec2 vec2Negate(v)` | 向量取反 |

### 几何运算

| 函数 | 说明 |
|------|------|
| `float vec2Dot(a, b)` | 点积 |
| `float vec2Cross(a, b)` | 2D叉积 (标量z) |
| `float vec2LengthSquared(v)` | 平方长度 |
| `float vec2Length(v)` | 长度 |
| `Vec2 vec2Normalize(v)` | 单位化 (精确) |
| `Vec2 vec2NormalizeFast(v)` | 单位化 (快速) |

### 高级操作

| 函数 | 说明 |
|------|------|
| `Vec2 vec2Perpendicular(v)` | 垂直向量 (逆时针90°) |
| `Vec2 vec2PerpendicularCw(v)` | 垂直向量 (顺时针90°) |
| `Vec2 vec2Lerp(a, b, t)` | 向量插值 |
| `Vec2 vec2Min(a, b)` | 分量最小值 |
| `Vec2 vec2Max(a, b)` | 分量最大值 |
| `Vec2 vec2Reflect(v, n)` | 向量反射 |
| `Vec2 vec2Project(a, b)` | 向量投影 |

### 距离与比较

| 函数 | 说明 |
|------|------|
| `float vec2DistanceSquared(a, b)` | 平方距离 |
| `float vec2Distance(a, b)` | 欧几里得距离 |
| `bool vec2Equal(a, b)` | 精确相等 |
| `bool vec2NearEqual(a, b, eps)` | 近似相等 |

### 批处理 (SIMD)

| 函数 | 说明 |
|------|------|
| `void vec2AddArray(a, b, result, count)` | 批量加法 |
| `void vec2DotArray(a, b, result, count)` | 批量点积 |
| `void vec2NormalizeArray(input, result, count)` | 批量单位化 (精确) |
| `void vec2NormalizeFastArray(input, result, count)` | 批量单位化 (快速) |

---

## Mat3 - 3x3矩阵 (`#include "fast_math/mat3.h"`)

### 构造

| 函数 | 说明 |
|------|------|
| `Mat3 mat3Identity()` | 单位矩阵 |
| `Mat3 mat3Zero()` | 零矩阵 |
| `Mat3 mat3FromTranslation(t)` | 平移矩阵 |
| `Mat3 mat3FromRotation(angle)` | 旋转矩阵 (弧度) |
| `Mat3 mat3FromScale(s)` | 缩放矩阵 |
| `Mat3 mat3FromTrs(t, r, s)` | TRS变换矩阵 |

### 运算

| 函数 | 说明 |
|------|------|
| `Mat3 mat3Multiply(a, b)` | 完整矩阵乘法 (27次乘法) |
| `Mat3 mat3MultiplyAffine(a, b)` | 仿射乘法 (18次乘法) |
| `Vec2 mat3TransformPoint(m, p)` | 变换点 (含平移) |
| `Vec2 mat3TransformVector(m, v)` | 变换向量 (不含平移) |

### 分解与变换

| 函数 | 说明 |
|------|------|
| `float mat3Determinant(m)` | 行列式 |
| `Mat3 mat3Inverse(m)` | 矩阵逆 (完整) |
| `Mat3 mat3InverseAffine(m)` | 仿射逆 (优化) |
| `Mat3 mat3Transpose(m)` | 转置 |

### 提取

| 函数 | 说明 |
|------|------|
| `Vec2 mat3ExtractTranslation(m)` | 提取平移 |
| `float mat3ExtractRotation(m)` | 提取旋转角 |
| `Vec2 mat3ExtractScale(m)` | 提取缩放 |

### 比较

| 函数 | 说明 |
|------|------|
| `bool mat3Equal(a, b)` | 精确相等 |
| `bool mat3NearEqual(a, b, eps)` | 近似相等 |

### 批处理 (SIMD)

| 函数 | 说明 |
|------|------|
| `void mat3FromTrsArray(t, r, s, result, count)` | 批量TRS构造 |
| `void mat3MultiplyAffineArray(a, b, result, count)` | 批量仿射乘法 |
| `void mat3TransformPointsArray(matrices, points, result, count)` | 批量变换 (一一对应) |
| `void mat3TransformPointsSingle(matrix, points, result, count)` | 单矩阵变换多点 |

---

## Aabb2 - 2D包围盒 (`#include "fast_math/aabb2.h"`)

### 构造

| 函数 | 说明 |
|------|------|
| `Aabb2 aabb2FromMinMax(min, max)` | 从最小/最大角创建 |
| `Aabb2 aabb2FromCenterExtents(center, extents)` | 从中心和半宽创建 |
| `Aabb2 aabb2FromPoint(point)` | 从单点创建 |
| `Aabb2 aabb2FromPoints(points, count)` | 从点数组创建 |
| `Aabb2 aabb2Empty()` | 空包围盒 |

### 访问

| 函数 | 说明 |
|------|------|
| `Vec2 aabb2Min(aabb)` | 获取最小角 |
| `Vec2 aabb2Max(aabb)` | 获取最大角 |
| `Vec2 aabb2Center(aabb)` | 获取中心点 |
| `Vec2 aabb2Extents(aabb)` | 获取半宽 |
| `Vec2 aabb2Size(aabb)` | 获取完整尺寸 |
| `float aabb2Width(aabb)` | 获取宽度 |
| `float aabb2Height(aabb)` | 获取高度 |

### 属性

| 函数 | 说明 |
|------|------|
| `float aabb2Area(aabb)` | 面积 |
| `float aabb2Perimeter(aabb)` | 周长 |
| `bool aabb2IsValid(aabb)` | 验证有效性 |

### 包含与交集

| 函数 | 说明 |
|------|------|
| `bool aabb2ContainsPoint(aabb, point)` | 测试点包含 |
| `bool aabb2Contains(outer, inner)` | 测试AABB包含 |
| `bool aabb2Intersects(a, b)` | 测试相交 (3条SSE指令) |
| `float aabb2OverlapArea(a, b)` | 计算重叠面积 |

### 集合操作

| 函数 | 说明 |
|------|------|
| `Aabb2 aabb2Merge(a, b)` | 合并 (单条SSE指令) |
| `Aabb2 aabb2Intersect(a, b)` | 交集 |
| `Aabb2 aabb2Expand(aabb, amount)` | 均匀扩展 |
| `Aabb2 aabb2ExpandToPoint(aabb, point)` | 扩展以包含点 |
| `Aabb2 aabb2ExpandToAabb(a, b)` | 扩展以包含AABB |

### 变换

| 函数 | 说明 |
|------|------|
| `Aabb2 aabb2Transform(aabb, matrix)` | 矩阵变换 |

### 距离查询

| 函数 | 说明 |
|------|------|
| `float aabb2DistanceSquared(aabb, point)` | 平方距离到点 |
| `float aabb2Distance(aabb, point)` | 距离到点 |
| `Vec2 aabb2ClosestPoint(aabb, point)` | 边界上最近点 |

### 射线投射

| 函数 | 说明 |
|------|------|
| `bool aabb2RayIntersects(aabb, origin, dir)` | 测试射线相交 |
| `bool aabb2RayIntersection(aabb, origin, dir, t_min, t_max)` | 获取交集参数 |

### 比较

| 函数 | 说明 |
|------|------|
| `bool aabb2Equal(a, b)` | 精确相等 |
| `bool aabb2NearEqual(a, b, eps)` | 近似相等 |

### 批处理 (SIMD)

| 函数 | 说明 |
|------|------|
| `void aabb2IntersectsArray(ref, candidates, results, count)` | 批量交集测试 |
| `void aabb2MergeArray(a, b, result, count)` | 批量合并 |
| `void aabb2ContainsPointsArray(aabb, points, results, count)` | 批量点包含测试 |

---

## Trig - 三角函数 (`#include "fast_math/trig.h"`)

### 单值API

| 函数 | 说明 |
|------|------|
| `void sincos(Angle angle, SinCos* result)` | 计算sin和cos (POD) |
| `float sin(float angle)` | 正弦 (弧度) |
| `float cos(float angle)` | 余弦 (弧度) |

### 批处理 (SIMD)

| 函数 | 说明 |
|------|------|
| `void sincosBatch(AngleBatch* angles, SinCosBatch* result)` | 批量sin/cos |
| `void sinBatch(AngleBatch* angles, float* out_sin)` | 批量sin |
| `void cosBatch(AngleBatch* angles, float* out_cos)` | 批量cos |

### C数组包装

| 函数 | 说明 |
|------|------|
| `void sincosArray(angles, out_sin, out_cos, count)` | 数组sin/cos |
| `void sinArray(angles, out_sin, count)` | 数组sin |
| `void cosArray(angles, out_cos, count)` | 数组cos |

---

## 类型定义

```cpp
struct Vec2 { float x, y; };
struct Mat3 { float m00, m01, m02, m10, m11, m12, m20, m21, m22; };
struct Aabb2 { float minx, miny, neg_maxx, neg_maxy; };
struct Angle { float value; };
struct SinCos { float sin, cos; };
struct AngleBatch { const float* data; int32_t count; };
struct SinCosBatch { float* sin_data; float* cos_data; int32_t count; };
```

---

## 编译要求

**标准**: C++17
**SIMD**: AVX2 + FMA (推荐) 或 SSE4.1 (兼容)
**编译标志**:
- GCC/Clang: `-std=c++17 -O3 -march=native -mfma -ffast-math`
- MSVC: `/std:c++17 /O2 /arch:AVX2 /fp:fast`

---

## 性能特性

| 操作 | 标量加速 | SIMD加速 |
|------|---------|---------|
| inverseSqrt | 2.08x | 7.53x |
| sqrtArray | - | 4.00x |
| lerpArray | - | 1.76x |
| clampArray | - | 2.18x |
| smoothstep | 1.07x (vs GLM) | - |

**注**: 加速倍数 vs std::或手写循环

---

**文档**: [API_REFERENCE.md](API_REFERENCE.md) (完整文档)
**版本**: 1.1.0 | **更新**: 2025-12-01
