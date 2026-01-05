# MMath API 参考文档

**版本**: 1.1.0
**语言**: C++17
**命名空间**: `MMath`

## ⚠️ Windows平台注意事项

MMath已解决与Windows `<windows.h>` 的宏冲突问题（`near` / `far` 宏）。即使在包含`<windows.h>`或使用DirectX、GLFW等图形库时，MMath也能正常工作。

如果你在使用中仍遇到命名冲突，建议：
```cpp
#define WIN32_LEAN_AND_MEAN  // 减少Windows.h包含的内容
#include <windows.h>
#undef near  // 可选：显式取消宏定义
#undef far
```

---

## 目录

1. [类型定义](#类型定义)
2. [Common - 通用数学函数](#common---通用数学函数)
3. [Sqrt - 平方根与倒数](#sqrt---平方根与倒数)
4. [Lerp - 线性插值与平滑曲线](#lerp---线性插值与平滑曲线)
5. [Vec2 - 2D向量](#vec2---2d向量)
6. [Mat3 - 3x3矩阵](#mat3---3x3矩阵)
7. [Aabb2 - 2D包围盒](#aabb2---2d包围盒)
8. [Trig - 三角函数](#trig---三角函数)

---

## 类型定义

### 基础类型

```cpp
// 2D向量
struct Vec2 {
    float x;
    float y;
};

// 3x3矩阵 (行主序)
struct Mat3 {
    float m00, m01, m02;
    float m10, m11, m12;
    float m20, m21, m22;
};

// 2D轴对齐包围盒 (AABB)
struct Aabb2 {
    float minx, miny;
    float neg_maxx, neg_maxy;  // 内部优化：存储 -maxx, -maxy
};

// 单个角度
struct Angle {
    float value;
};

// 单个sin/cos对
struct SinCos {
    float sin;
    float cos;
};

// 批量角度 (POD, 用于SIMD批处理)
struct AngleBatch {
    const float* MMATH_RESTRICT data;
    int32_t count;
};

// 批量sin/cos (POD, 用于SIMD批处理)
struct SinCosBatch {
    float* MMATH_RESTRICT sin_data;
    float* MMATH_RESTRICT cos_data;
    int32_t count;
};
```

### SIMD专用类型

```cpp
// AVX2 (8个角度)
struct MMATH_ALIGN(32) AnglesAvx2 {
    float angles[8];
};

struct MMATH_ALIGN(32) SinCosAvx2 {
    float sins[8];
    float coss[8];
};

// SSE4.1 (4个角度)
struct MMATH_ALIGN(16) AnglesSse {
    float angles[4];
};

struct MMATH_ALIGN(16) SinCosSse {
    float sins[4];
    float coss[4];
};
```

---

## Common - 通用数学函数

**头文件**: `#include "fast_math/common.h"`

### 比较函数 (Branch-Free)

```cpp
// 两个浮点数的最小值
MMATH_FORCE_INLINE float min(float a_, float b_) noexcept;

// 两个浮点数的最大值
MMATH_FORCE_INLINE float max(float a_, float b_) noexcept;

// 将值限制在[min_val, max_val]范围内
MMATH_FORCE_INLINE float clamp(float x_, float min_val_, float max_val_) noexcept;
```

**性能**: 编译器自动生成minss/maxss指令，无分支开销

**用例** (来自音频引擎分析):
- `clamp()` - 27次使用：音量限制 [0,1]、声像限制 [-1,1]、软削波
- `min/max()` - 22次使用：混音帧数计算、距离参数限制

### 绝对值与符号

```cpp
// 绝对值
MMATH_FORCE_INLINE float abs(float x_) noexcept;

// 符号函数 (-1.0f, 0.0f, +1.0f)
MMATH_FORCE_INLINE float sign(float x_) noexcept;

// 将值限制在[0, 1]范围 (等价于clamp(x, 0, 1))
MMATH_FORCE_INLINE float saturate(float x_) noexcept;

// 复制符号位 (返回x的大小，y的符号)
MMATH_FORCE_INLINE float copySign(float x_, float y_) noexcept;
```

**性能**:
- `abs()` - 使用std::fabs，编译为单条andps指令 (清除符号位)
- `sign()` - 编译器优化为无分支代码 (条件移动)

### 浮点比较工具

```cpp
// 近似相等 (|a - b| <= epsilon)
MMATH_FORCE_INLINE bool nearEqual(
    float a_,
    float b_,
    float epsilon_ = 1e-5f
) noexcept;

// 近似为零 (|x| <= epsilon)
MMATH_FORCE_INLINE bool isZero(
    float x_,
    float epsilon_ = 1e-6f
) noexcept;
```

**用例**: 物理计算容差检查、浮点测试断言

### 批处理 (SIMD优化)

```cpp
// 数组最小值 (a[i] = min(a[i], b[i]))
inline void minArray(
    float* MMATH_RESTRICT a_,
    const float* MMATH_RESTRICT b_,
    int32_t count_
) noexcept;

// 数组最大值 (a[i] = max(a[i], b[i]))
inline void maxArray(
    float* MMATH_RESTRICT a_,
    const float* MMATH_RESTRICT b_,
    int32_t count_
) noexcept;

// 数组限幅 (values[i] = clamp(values[i], min, max))
inline void clampArray(
    float* MMATH_RESTRICT values_,
    float min_,
    float max_,
    int32_t count_
) noexcept;

// 数组绝对值 (values[i] = abs(values[i]))
inline void absArray(
    float* MMATH_RESTRICT values_,
    int32_t count_
) noexcept;

// 数组饱和 (values[i] = saturate(values[i]))
inline void saturateArray(
    float* MMATH_RESTRICT values_,
    int32_t count_
) noexcept;
```

**性能** (1M元素大数组):
- `clampArray` - 2.18x vs std::循环
- `absArray` - 接近std::性能
- **SIMD路径**: AVX2处理8个/次，SSE4.1处理4个/次

**实际场景**: 音频音量限制 (480000采样，10s @ 48kHz) - **1.43x加速**

---

## Sqrt - 平方根与倒数

**头文件**: `#include "fast_math/sqrt.h"`

### 平方根函数

```cpp
// 精确平方根 (硬件sqrtss指令)
MMATH_FORCE_INLINE float sqrt(float x_) noexcept;
```

**性能**: ≥ std::sqrt (14-15周期延迟，与std::sqrt相同)
**精度**: 精确 (硬件精度)

### 快速倒数平方根

```cpp
// 快速倒数平方根 1/sqrt(x)
// rsqrt + Newton-Raphson迭代
MMATH_FORCE_INLINE float inverseSqrt(float x_) noexcept;
```

**性能**: **2.08x vs 1/std::sqrt** (标量)，**7.53x** (SIMD数组)
**精度**: 相对误差 < 0.01% (最大误差 2.3e-07)

**用例**: 向量归一化 (vec2Normalize内部使用)

```cpp
Vec2 vec2Normalize(Vec2 v) {
    float len_sq = vec2LengthSquared(v);
    float inv_len = inverseSqrt(len_sq);  // 比 1/sqrt快2x!
    return vec2Scale(v, inv_len);
}
```

### 快速倒数

```cpp
// 快速倒数 1/x
// rcp + Newton-Raphson迭代
MMATH_FORCE_INLINE float reciprocal(float x_) noexcept;
```

**性能**: **1.52x vs 1/x** (在旧CPU上更明显)
**精度**: 相对误差 < 0.01%

**注意**: 现代CPU的除法单元很快，优势在大批量SIMD操作

### 辅助函数

```cpp
// 归一化因子 (用于向量归一化)
// 返回 1/sqrt(length_squared)，若length_squared接近0返回0
MMATH_FORCE_INLINE float normalizeFactor(float length_squared_) noexcept;

// 检查浮点数是否可安全用于除法
MMATH_FORCE_INLINE bool isSafeForDivision(
    float x_,
    float epsilon_ = 1e-8f
) noexcept;

// 安全倒数 (若x接近0返回fallback)
MMATH_FORCE_INLINE float safeReciprocal(
    float x_,
    float fallback_ = 0.0f,
    float epsilon_ = 1e-8f
) noexcept;
```

### 批处理 (SIMD优化)

```cpp
// 批量平方根
inline void sqrtArray(
    float* MMATH_RESTRICT values_,
    int32_t count_
) noexcept;

// 批量倒数平方根
inline void inverseSqrtArray(
    float* MMATH_RESTRICT values_,
    int32_t count_
) noexcept;

// 批量倒数
inline void reciprocalArray(
    float* MMATH_RESTRICT values_,
    int32_t count_
) noexcept;
```

**性能** (1M元素):
- `sqrtArray` - **4.00x vs std::循环**
- `inverseSqrtArray` - **7.53x vs std::循环** 🏆
- `reciprocalArray` - **1.52x vs std::循环**

**比较**: MMath在所有测试中均快于或持平DirectXMath和GLM

---

## Lerp - 线性插值与平滑曲线

**头文件**: `#include "fast_math/lerp.h"`

### 线性插值

```cpp
// 线性插值: a + t*(b - a)
// 编译器自动生成FMA指令 (with -mfma)
MMATH_FORCE_INLINE float lerp(float a_, float b_, float t_) noexcept;
```

**数学**:
- `lerp(a, b, 0)` = a (精确)
- `lerp(a, b, 1)` = b (精确)
- `lerp(a, b, 0.5)` = (a+b)/2 (中点)

**性能**: 纯标量算术，编译器自动生成FMA (fmadd: t*(b-a)+a)

**用例** (音频引擎):
- 分数延迟插值 (读取位置在两采样之间)
- 音量交叉淡化: `lerp(old_vol, new_vol, fade_t)`
- 参数平滑

```cpp
// 逆线性插值 (返回参数t)
// 满足: lerp(a, b, inverseLerp(a, b, value)) ≈ value
MMATH_FORCE_INLINE float inverseLerp(
    float a_,
    float b_,
    float value_
) noexcept;
```

**用例**:
- 传感器读数映射到归一化范围
- UI滑块：值 → 位置转换

```cpp
// 重映射值从一个范围到另一个范围
// 等价于: lerp(out_min, out_max, inverseLerp(in_min, in_max, value))
MMATH_FORCE_INLINE float remap(
    float value_,
    float in_min_, float in_max_,
    float out_min_, float out_max_
) noexcept;
```

**示例**:
```cpp
float midi_note = 60;  // Middle C
float freq = remap(midi_note, 0, 127, 20.0f, 20000.0f);
```

### Hermite平滑曲线

```cpp
// 三次Hermite插值 (C1连续)
// 公式: x²(3 - 2x) where x = clamp((t - edge0)/(edge1 - edge0), 0, 1)
MMATH_FORCE_INLINE float smoothstep(
    float edge0_,
    float edge1_,
    float t_
) noexcept;
```

**特性**:
- 平滑的ease-in/ease-out (S曲线)
- 边界处一阶导数为零 (速度平滑过渡)
- `t <= edge0` 返回 0
- `t >= edge1` 返回 1

**性能**: **比GLM快7.3%** (通过内联除法优化)

**用例**:
- 音频淡入淡出效果
- UI动画 (按钮按下、菜单滑动)
- 相机过渡
- 程序化噪声平滑 (Perlin noise)

```cpp
// 五次Hermite插值 (C2连续)
// 公式: x³(x(x*6 - 15) + 10)
MMATH_FORCE_INLINE float smootherstep(
    float edge0_,
    float edge1_,
    float t_
) noexcept;
```

**特性**:
- 比smoothstep更平滑 (变化更柔和)
- 一阶和二阶导数在边界处为零 (加速度也平滑)

**用例**:
- 高质量动画曲线
- 程序化地形生成 (平滑混合)
- 高级音频交叉淡化
- 电影级相机运动

### 变体函数

```cpp
// 限制t到[0,1]的线性插值
MMATH_FORCE_INLINE float lerpClamped(
    float a_,
    float b_,
    float t_
) noexcept;

// 阶梯式插值 (像素艺术风格)
MMATH_FORCE_INLINE float lerpStepped(
    float a_,
    float b_,
    float t_,
    int steps_
) noexcept;
```

**用例**:
- `lerpClamped` - 保证结果在[min(a,b), max(a,b)]范围内
- `lerpStepped` - 复古游戏动画、量化参数变化

### 工具函数

```cpp
// 检查值是否在范围内 (带容差)
MMATH_FORCE_INLINE bool isInRange(
    float value_,
    float a_,
    float b_,
    float epsilon_ = 1e-6f
) noexcept;
```

### 批处理 (SIMD优化)

```cpp
// 批量线性插值
// result[i] = lerp(a[i], b[i], t)
inline void lerpArray(
    const float* MMATH_RESTRICT a_,
    const float* MMATH_RESTRICT b_,
    float t_,
    float* MMATH_RESTRICT result_,
    int32_t count_
) noexcept;

// 批量smoothstep (就地操作)
// values[i] = smoothstep(edge0, edge1, values[i])
inline void smoothstepArray(
    float edge0_,
    float edge1_,
    float* MMATH_RESTRICT values_,
    int32_t count_
) noexcept;

// 批量smootherstep (就地操作)
inline void smootherstepArray(
    float edge0_,
    float edge1_,
    float* MMATH_RESTRICT values_,
    int32_t count_
) noexcept;

// 批量重映射
inline void remapArray(
    const float* MMATH_RESTRICT input_,
    float in_min_, float in_max_,
    float out_min_, float out_max_,
    float* MMATH_RESTRICT output_,
    int32_t count_
) noexcept;
```

**性能** (1M元素):
- `lerpArray` - **1.76x vs std::循环** (通过循环展开+软件预取+流式存储优化)
- `smoothstepArray` - **1.08x vs std::循环**

**优化技术**:
1. 循环展开 - 一次处理2个SIMD寄存器 (AVX2: 16 floats, SSE: 8 floats)
2. 软件预取 - 提前64字节预取数据
3. 流式存储 - 大数组(>64KB)使用`_mm_stream_ps`避免cache污染

**比较**: lerpArray性能提升85%，从落后DirectXMath 36% → **领先26%**

---

## Vec2 - 2D向量

### 基本运算

```cpp
// 向量加法
MMATH_FORCE_INLINE Vec2 vec2Add(Vec2 a_, Vec2 b_) noexcept;

// 向量减法
MMATH_FORCE_INLINE Vec2 vec2Sub(Vec2 a_, Vec2 b_) noexcept;

// 分量乘法
MMATH_FORCE_INLINE Vec2 vec2Mul(Vec2 a_, Vec2 b_) noexcept;

// 分量除法
MMATH_FORCE_INLINE Vec2 vec2Div(Vec2 a_, Vec2 b_) noexcept;

// 标量缩放
MMATH_FORCE_INLINE Vec2 vec2Scale(Vec2 v_, float s_) noexcept;

// 向量取反
MMATH_FORCE_INLINE Vec2 vec2Negate(Vec2 v_) noexcept;
```

### 点积和长度

```cpp
// 点积
MMATH_FORCE_INLINE float vec2Dot(Vec2 a_, Vec2 b_) noexcept;

// 平方长度 (避免sqrt)
MMATH_FORCE_INLINE float vec2LengthSquared(Vec2 v_) noexcept;

// 长度
MMATH_FORCE_INLINE float vec2Length(Vec2 v_) noexcept;

// 单位化向量 (精确除法)
MMATH_FORCE_INLINE Vec2 vec2Normalize(Vec2 v_) noexcept;

// 快速单位化 (近似倒数平方根, ~0.1%误差)
MMATH_FORCE_INLINE Vec2 vec2NormalizeFast(Vec2 v_) noexcept;
```

### 叉积和垂直

```cpp
// 2D叉积 (返回标量z分量)
MMATH_FORCE_INLINE float vec2Cross(Vec2 a_, Vec2 b_) noexcept;

// 垂直向量 (逆时针90°)
MMATH_FORCE_INLINE Vec2 vec2Perpendicular(Vec2 v_) noexcept;

// 垂直向量 (顺时针90°)
MMATH_FORCE_INLINE Vec2 vec2PerpendicularCw(Vec2 v_) noexcept;
```

### 插值和极值

```cpp
// 线性插值
MMATH_FORCE_INLINE Vec2 vec2Lerp(Vec2 a_, Vec2 b_, float t_) noexcept;

// 分量最小值
MMATH_FORCE_INLINE Vec2 vec2Min(Vec2 a_, Vec2 b_) noexcept;

// 分量最大值
MMATH_FORCE_INLINE Vec2 vec2Max(Vec2 a_, Vec2 b_) noexcept;
```

### 反射和投影

```cpp
// 向量反射 (v - 2*dot(v,n)*n)
MMATH_FORCE_INLINE Vec2 vec2Reflect(Vec2 v_, Vec2 n_) noexcept;

// 向量投影
MMATH_FORCE_INLINE Vec2 vec2Project(Vec2 a_, Vec2 b_) noexcept;
```

### 距离

```cpp
// 平方距离 (避免sqrt)
MMATH_FORCE_INLINE float vec2DistanceSquared(Vec2 a_, Vec2 b_) noexcept;

// 欧几里得距离
MMATH_FORCE_INLINE float vec2Distance(Vec2 a_, Vec2 b_) noexcept;
```

### 比较

```cpp
// 精确相等
MMATH_FORCE_INLINE bool vec2Equal(Vec2 a_, Vec2 b_) noexcept;

// 近似相等
MMATH_FORCE_INLINE bool vec2NearEqual(Vec2 a_, Vec2 b_, float epsilon_) noexcept;
```

### 批处理 (SIMD优化)

```cpp
// 批量加法 (AVX2: 8个向量, SSE: 4个向量)
inline void vec2AddArray(
    const Vec2* MMATH_RESTRICT a_,
    const Vec2* MMATH_RESTRICT b_,
    Vec2* MMATH_RESTRICT result_,
    int32_t count_
) noexcept;

// 批量点积
inline void vec2DotArray(
    const Vec2* MMATH_RESTRICT a_,
    const Vec2* MMATH_RESTRICT b_,
    float* MMATH_RESTRICT result_,
    int32_t count_
) noexcept;

// 批量单位化 (精确)
inline void vec2NormalizeArray(
    const Vec2* MMATH_RESTRICT input_,
    Vec2* MMATH_RESTRICT result_,
    int32_t count_
) noexcept;

// 批量单位化 (快速)
inline void vec2NormalizeFastArray(
    const Vec2* MMATH_RESTRICT input_,
    Vec2* MMATH_RESTRICT result_,
    int32_t count_
) noexcept;
```

---

## Mat3 - 3x3矩阵

### 矩阵构造

```cpp
// 单位矩阵
MMATH_FORCE_INLINE Mat3 mat3Identity() noexcept;

// 零矩阵
MMATH_FORCE_INLINE Mat3 mat3Zero() noexcept;

// 平移矩阵
MMATH_FORCE_INLINE Mat3 mat3FromTranslation(Vec2 translation_) noexcept;

// 旋转矩阵 (弧度)
MMATH_FORCE_INLINE Mat3 mat3FromRotation(float angle_radians_) noexcept;

// 缩放矩阵
MMATH_FORCE_INLINE Mat3 mat3FromScale(Vec2 scale_) noexcept;

// TRS变换矩阵 (平移-旋转-缩放)
MMATH_FORCE_INLINE Mat3 mat3FromTrs(
    Vec2 translation_,
    float rotation_radians_,
    Vec2 scale_
) noexcept;
```

### 矩阵乘法

```cpp
// 完整3x3矩阵乘法 (27次乘法)
MMATH_FORCE_INLINE Mat3 mat3Multiply(Mat3 a_, Mat3 b_) noexcept;

// 仿射变换乘法优化 (18次乘法, 假设最后一行为[0,0,1])
MMATH_FORCE_INLINE Mat3 mat3MultiplyAffine(Mat3 a_, Mat3 b_) noexcept;
```

### 矩阵变换

```cpp
// 变换点 (应用平移)
MMATH_FORCE_INLINE Vec2 mat3TransformPoint(Mat3 m_, Vec2 point_) noexcept;

// 变换向量 (不应用平移)
MMATH_FORCE_INLINE Vec2 mat3TransformVector(Mat3 m_, Vec2 vector_) noexcept;
```

### 矩阵分解

```cpp
// 计算行列式
MMATH_FORCE_INLINE float mat3Determinant(Mat3 m_) noexcept;

// 矩阵逆 (完整3x3)
MMATH_FORCE_INLINE Mat3 mat3Inverse(Mat3 m_) noexcept;

// 仿射矩阵逆 (优化版本)
MMATH_FORCE_INLINE Mat3 mat3InverseAffine(Mat3 m_) noexcept;

// 矩阵转置
MMATH_FORCE_INLINE Mat3 mat3Transpose(Mat3 m_) noexcept;
```

### 矩阵分解

```cpp
// 提取平移分量
MMATH_FORCE_INLINE Vec2 mat3ExtractTranslation(Mat3 m_) noexcept;

// 提取旋转角度 (弧度)
MMATH_FORCE_INLINE float mat3ExtractRotation(Mat3 m_) noexcept;

// 提取缩放分量
MMATH_FORCE_INLINE Vec2 mat3ExtractScale(Mat3 m_) noexcept;
```

### 比较

```cpp
// 精确相等
MMATH_FORCE_INLINE bool mat3Equal(Mat3 a_, Mat3 b_) noexcept;

// 近似相等
MMATH_FORCE_INLINE bool mat3NearEqual(Mat3 a_, Mat3 b_, float epsilon_) noexcept;
```

### 批处理 (SIMD优化)

```cpp
// 批量TRS构造
inline void mat3FromTrsArray(
    const Vec2* MMATH_RESTRICT translations_,
    const float* MMATH_RESTRICT rotations_,
    const Vec2* MMATH_RESTRICT scales_,
    Mat3* MMATH_RESTRICT result_,
    int32_t count_
) noexcept;

// 批量仿射乘法
inline void mat3MultiplyAffineArray(
    const Mat3* MMATH_RESTRICT a_,
    const Mat3* MMATH_RESTRICT b_,
    Mat3* MMATH_RESTRICT result_,
    int32_t count_
) noexcept;

// 批量变换点 (每个矩阵对应每个点)
inline void mat3TransformPointsArray(
    const Mat3* MMATH_RESTRICT matrices_,
    const Vec2* MMATH_RESTRICT points_,
    Vec2* MMATH_RESTRICT result_,
    int32_t count_
) noexcept;

// 单个矩阵变换多个点
inline void mat3TransformPointsSingle(
    Mat3 matrix_,
    const Vec2* MMATH_RESTRICT points_,
    Vec2* MMATH_RESTRICT result_,
    int32_t count_
) noexcept;
```

---

## Aabb2 - 2D包围盒

### 构造

```cpp
// 从最小和最大角创建
MMATH_FORCE_INLINE Aabb2 aabb2FromMinMax(Vec2 min_, Vec2 max_) noexcept;

// 从中心和半宽度创建
MMATH_FORCE_INLINE Aabb2 aabb2FromCenterExtents(Vec2 center_, Vec2 extents_) noexcept;

// 从单点创建 (退化的包围盒)
MMATH_FORCE_INLINE Aabb2 aabb2FromPoint(Vec2 point_) noexcept;

// 从点阵列创建 (紧包围盒)
inline Aabb2 aabb2FromPoints(
    const Vec2* MMATH_RESTRICT points_,
    int32_t count_
) noexcept;

// 创建空包围盒 (用于扩展操作)
MMATH_FORCE_INLINE Aabb2 aabb2Empty() noexcept;
```

### 访问

```cpp
// 获取最小角
MMATH_FORCE_INLINE Vec2 aabb2Min(Aabb2 aabb_) noexcept;

// 获取最大角
MMATH_FORCE_INLINE Vec2 aabb2Max(Aabb2 aabb_) noexcept;

// 获取中心点
MMATH_FORCE_INLINE Vec2 aabb2Center(Aabb2 aabb_) noexcept;

// 获取半宽度 (从中心到边)
MMATH_FORCE_INLINE Vec2 aabb2Extents(Aabb2 aabb_) noexcept;

// 获取完整宽高
MMATH_FORCE_INLINE Vec2 aabb2Size(Aabb2 aabb_) noexcept;

// 获取宽度
MMATH_FORCE_INLINE float aabb2Width(Aabb2 aabb_) noexcept;

// 获取高度
MMATH_FORCE_INLINE float aabb2Height(Aabb2 aabb_) noexcept;
```

### 几何属性

```cpp
// 面积
MMATH_FORCE_INLINE float aabb2Area(Aabb2 aabb_) noexcept;

// 周长
MMATH_FORCE_INLINE float aabb2Perimeter(Aabb2 aabb_) noexcept;

// 验证有效性 (min <= max)
MMATH_FORCE_INLINE bool aabb2IsValid(Aabb2 aabb_) noexcept;
```

### 包含和交集

```cpp
// 测试点是否在包围盒内
MMATH_FORCE_INLINE bool aabb2ContainsPoint(Aabb2 aabb_, Vec2 point_) noexcept;

// 测试外部AABB是否包含内部AABB
MMATH_FORCE_INLINE bool aabb2Contains(Aabb2 outer_, Aabb2 inner_) noexcept;

// 测试两个AABB是否相交 (3条SSE指令优化)
MMATH_FORCE_INLINE bool aabb2Intersects(Aabb2 a_, Aabb2 b_) noexcept;

// 计算重叠面积 (不重叠返回0)
MMATH_FORCE_INLINE float aabb2OverlapArea(Aabb2 a_, Aabb2 b_) noexcept;
```

### 集合操作

```cpp
// 合并两个AABB (单条SSE指令!)
MMATH_FORCE_INLINE Aabb2 aabb2Merge(Aabb2 a_, Aabb2 b_) noexcept;

// 计算两个AABB的交集 (可能返回无效AABB)
MMATH_FORCE_INLINE Aabb2 aabb2Intersect(Aabb2 a_, Aabb2 b_) noexcept;

// 均匀扩展所有边
MMATH_FORCE_INLINE Aabb2 aabb2Expand(Aabb2 aabb_, float amount_) noexcept;

// 扩展以包含点
MMATH_FORCE_INLINE Aabb2 aabb2ExpandToPoint(Aabb2 aabb_, Vec2 point_) noexcept;

// 扩展以包含另一个AABB
MMATH_FORCE_INLINE Aabb2 aabb2ExpandToAabb(Aabb2 a_, Aabb2 b_) noexcept;
```

### 变换

```cpp
// 通过矩阵变换并重新计算包围盒
inline Aabb2 aabb2Transform(Aabb2 aabb_, Mat3 matrix_) noexcept;
```

### 距离查询

```cpp
// 平方距离到最近点
MMATH_FORCE_INLINE float aabb2DistanceSquared(Aabb2 aabb_, Vec2 point_) noexcept;

// 距离到最近点
MMATH_FORCE_INLINE float aabb2Distance(Aabb2 aabb_, Vec2 point_) noexcept;

// 在AABB边界上最近的点
MMATH_FORCE_INLINE Vec2 aabb2ClosestPoint(Aabb2 aabb_, Vec2 point_) noexcept;
```

### 射线投射

```cpp
// 测试射线是否与AABB相交
MMATH_FORCE_INLINE bool aabb2RayIntersects(
    Aabb2 aabb_,
    Vec2 origin_,
    Vec2 direction_
) noexcept;

// 获取射线-AABB交集参数
inline bool aabb2RayIntersection(
    Aabb2 aabb_,
    Vec2 origin_,
    Vec2 direction_,
    float* t_min_,
    float* t_max_
) noexcept;
```

### 比较

```cpp
// 精确相等
MMATH_FORCE_INLINE bool aabb2Equal(Aabb2 a_, Aabb2 b_) noexcept;

// 近似相等
MMATH_FORCE_INLINE bool aabb2NearEqual(Aabb2 a_, Aabb2 b_, float epsilon_) noexcept;
```

### 批处理 (SIMD优化)

```cpp
// 批量交集测试 (一个参考AABB vs 多个候选)
inline void aabb2IntersectsArray(
    Aabb2 reference_,
    const Aabb2* MMATH_RESTRICT candidates_,
    bool* MMATH_RESTRICT results_,
    int32_t count_
) noexcept;

// 批量合并 (配对)
inline void aabb2MergeArray(
    const Aabb2* MMATH_RESTRICT a_,
    const Aabb2* MMATH_RESTRICT b_,
    Aabb2* MMATH_RESTRICT result_,
    int32_t count_
) noexcept;

// 批量点包含测试 (一个AABB vs 多个点)
inline void aabb2ContainsPointsArray(
    Aabb2 aabb_,
    const Vec2* MMATH_RESTRICT points_,
    bool* MMATH_RESTRICT results_,
    int32_t count_
) noexcept;
```

---

## Trig - 三角函数

### 批处理 (SIMD优化)

```cpp
// 批量sin/cos计算 (AVX2: 8个角, SSE: 4个角)
inline void sincosBatch(
    const AngleBatch* MMATH_RESTRICT angles_,
    SinCosBatch* MMATH_RESTRICT result_
) noexcept;

// 批量sin计算
inline void sinBatch(
    const AngleBatch* MMATH_RESTRICT angles_,
    float* MMATH_RESTRICT out_sin_
) noexcept;

// 批量cos计算
inline void cosBatch(
    const AngleBatch* MMATH_RESTRICT angles_,
    float* MMATH_RESTRICT out_cos_
) noexcept;
```

### C数组包装

```cpp
// sin/cos for plain C arrays
inline void sincosArray(
    const float* MMATH_RESTRICT angles_,
    float* MMATH_RESTRICT out_sin_,
    float* MMATH_RESTRICT out_cos_,
    int32_t count_
) noexcept;

// sin for plain C arrays
inline void sinArray(
    const float* MMATH_RESTRICT angles_,
    float* MMATH_RESTRICT out_sin_,
    int32_t count_
) noexcept;

// cos for plain C arrays
inline void cosArray(
    const float* MMATH_RESTRICT angles_,
    float* MMATH_RESTRICT out_cos_,
    int32_t count_
) noexcept;
```

### 单值API

```cpp
// 单个角度的sin/cos (POD结构)
inline void sincos(Angle angle_, SinCos* MMATH_RESTRICT result_) noexcept;

// 单个角度的sin (返回float)
inline float sin(float angle_) noexcept;

// 单个角度的cos (返回float)
inline float cos(float angle_) noexcept;
```

---

## 宏定义

```cpp
// 内存对齐
#define MMATH_ALIGN(x) __attribute__((aligned(x)))  // GCC/Clang
#define MMATH_ALIGN(x) __declspec(align(x))         // MSVC

// 内联强制
#define MMATH_FORCE_INLINE __forceinline            // MSVC
#define MMATH_FORCE_INLINE __attribute__((always_inline)) inline  // GCC/Clang

// 受限指针
#define MMATH_RESTRICT __restrict                   // MSVC
#define MMATH_RESTRICT __restrict__                 // GCC/Clang
```

---

## 编译器要求

- **C++标准**: C++17 或更高
- **SIMD支持**:
  - AVX2 + FMA: 用于最优性能 (推荐)
  - SSE4.1: 用于兼容性回退
  - 标量: 无SIMD要求 (但性能降低)

### 推荐编译标志

**GCC/Clang**:
```bash
-std=c++17 -O3 -march=native -ffast-math -funroll-loops -finline-functions
```

**MSVC**:
```bash
/std:c++17 /O2 /arch:AVX2 /fp:fast
```

---

## 命名空间

所有函数都在 `MMath` 命名空间中:

```cpp
using namespace MMath;

// 直接使用
Vec2 v = vec2Add({1, 2}, {3, 4});

// 或使用完全限定名
MMath::Vec2 v = MMath::vec2Add({1, 2}, {3, 4});
```

---

## 头文件包含

```cpp
// 基础数学函数
#include "fast_math/common.h"      // min, max, clamp, abs, sign
#include "fast_math/sqrt.h"        // sqrt, inverseSqrt, reciprocal
#include "fast_math/lerp.h"        // lerp, smoothstep, smootherstep

// 几何类型
#include "fast_math/vec2.h"        // 2D向量运算
#include "fast_math/mat3.h"        // 3x3矩阵运算
#include "fast_math/aabb2.h"       // 2D包围盒
#include "fast_math/trig.h"        // 三角函数

// 类型定义 (被所有模块自动包含)
#include "fast_math/types.h"
```

**注意**: 每个头文件自动包含对应的`_simd.h`实现，无需手动包含

---

## 性能提示

### 通用建议

1. **使用批处理函数** 处理大型数组 (>100元素)
   - `lerpArray` vs 循环lerp: **1.76x加速**
   - `inverseSqrtArray` vs 循环1/sqrt: **7.53x加速**
   - `clampArray` vs 循环clamp: **2.18x加速**

2. **使用Fast变体** 当精度要求不严格时
   - `vec2NormalizeFast` - 近似倒数平方根 (~0.1%误差)
   - `inverseSqrt` vs `1/sqrt` - **2.08x加速** (误差<0.01%)

3. **避免不必要的sqrt**
   - 使用`vec2LengthSquared`, `vec2DistanceSquared`进行距离比较
   - 归一化用`inverseSqrt`代替`1/sqrt`

4. **预计算常量变换**
   - 将重复使用的矩阵预先计算，而不是每次构造
   - 使用`mat3FromTrsArray`批量构造多个矩阵

5. **使用仿射优化**
   - `mat3MultiplyAffine` vs `mat3Multiply`: 18次乘法 vs 27次
   - 2D游戏变换矩阵最后一行总是[0,0,1]，使用仿射版本

6. **合并AABB操作**
   - 先累积多个包围盒，最后一次合并
   - `aabb2Merge`使用单条SSE指令 (极快)

### 新模块性能建议 (v1.1.0)

7. **音频处理场景**
   - 音量限制: 使用`clampArray` (**1.43x实测加速**, 480k采样)
   - 淡入淡出: 使用`smoothstepArray`代替手写循环
   - 参数插值: 使用`lerpArray`代替逐样本lerp

8. **选择正确的平滑曲线**
   - `lerp` - 线性插值，最快但无缓动
   - `smoothstep` - 三次曲线，ease-in/out，比GLM快7.3%
   - `smootherstep` - 五次曲线，更平滑但稍慢

9. **利用编译器FMA优化**
   - 使用`-mfma`编译标志
   - `lerp(a, b, t)`自动优化为单条fmadd指令
   - 比手写`a + t*(b-a)`快5-30%

10. **大数组SIMD优化**
    - 数组>64KB时，`lerpArray`自动使用流式存储 (避免cache污染)
    - 软件预取提前64字节加载数据 (隐藏内存延迟)
    - 循环展开一次处理16个floats (AVX2) 或 8个floats (SSE)

---

## 版本历史

### 1.1.0 (2025-12-01)
- ✅ **新增**: Common模块 (min, max, clamp, abs, sign, saturate)
  - 标量函数: 编译器自动优化为minss/maxss指令
  - SIMD数组: clampArray 2.18x vs std::循环
  - 实际场景: 音频音量限制 1.43x加速
- ✅ **新增**: Sqrt模块 (sqrt, inverseSqrt, reciprocal)
  - inverseSqrt: 2.08x vs 1/std::sqrt (标量)，7.53x (SIMD) 🏆
  - sqrtArray: 4.00x vs std::循环
  - 精度验证: 相对误差 < 0.01% (最大2.3e-07)
- ✅ **新增**: Lerp模块 (lerp, smoothstep, smootherstep, remap)
  - smoothstep: 比GLM快7.3% (标量)
  - lerpArray: 1.76x vs std::循环 (通过循环展开+软件预取+流式存储优化85%)
  - 功能测试: 19/19全部通过
- **优化**: 移除SSE标量intrinsics开销，采用纯标量代码让编译器自动优化
- **测试**: 所有模块包含严格性能基准 (warmup + median of 5 runs)

### 1.0.0 (2025-11-28)
- 初始发布
- 完整的Vec2, Mat3, Aabb2, Trig实现
- AVX2/SSE4.1 SIMD优化
- 批处理函数支持

---

**作者**: Melosyne Project
**许可证**: MIT
**最后更新**: 2025-12-01
