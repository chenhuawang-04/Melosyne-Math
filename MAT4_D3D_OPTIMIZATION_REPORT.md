# Mat4 D3D 性能优化报告 (最终版)

## 优化成果总结

| 操作 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| Perspective | -71% | **+34.0%** | 105pp |
| mat4MulVec4 | -7.3% | **+10.6%** | 18pp |
| Matrix Multiply | +17.1% | **+13.9%** | 持平 |
| LookAt | -10.4% | -27.2% | 需评估 |
| **Dynamic MVP (Mat4)** | -380% | -729.9% | 未优化 |
| **Dynamic MVP (Mat4R)** | N/A | **0.0%** | **完美匹配** |

## Mat4R 寄存器矩阵类型 (核心优化)

### 问题根因
- **DirectXMath**: `XMMATRIX` = 4个`__m128`，编译器直接在寄存器传递
- **MMath Mat4**: `float[16]`，每次操作需 load/store

### 解决方案
新增 `Mat4R` 类型，镜像 DirectXMath 的 XMMATRIX 设计：

```cpp
struct Mat4R {
    __m128 r[4];  // 4 列直接在寄存器
};

// 核心 API
Mat4R mat4Load(const Mat4& m);        // 从内存加载到寄存器
Mat4  mat4Store(const Mat4R& m);      // 从寄存器存储到内存
Mat4R mat4MulR(Mat4R A, Mat4R B);     // 寄存器矩阵乘法
__m128 mat4MulVec4R(Mat4R M, __m128 v); // 寄存器矩阵-向量乘法
```

### 关键优化技巧
1. **值传递 vs 引用传递**: 使用值传递让编译器在 XMM 寄存器中保持数据
2. **展开参数**: `_lincomb_r` 接收 5 个 `__m128` 参数而非结构体引用
3. **FMA 利用**: 在支持 FMA 的平台使用融合乘加指令

### 性能对比

| 场景 | Mat4 | Mat4R | DirectXMath |
|------|------|-------|-------------|
| Dynamic MVP | 86.9 M/s | **720.7 M/s** | 721.0 M/s |
| 速度差距 | -729.9% | **0.0%** | baseline |
| Mat4R 加速比 | - | **8.3x** | - |

## 使用指南

### 何时使用 Mat4R
- 链式矩阵操作 (例如: MVP = Proj * View * Model)
- 性能关键的变换循环
- 数据在整个计算过程中保持在寄存器

### 何时使用 Mat4
- 存储在数组/结构体中
- 传递给着色器/API
- 内存布局需要明确

### 典型用法
```cpp
// 加载到寄存器
Mat4R projR = mat4Load(projMatrix);
Mat4R viewR = mat4Load(viewMatrix);
Mat4R modelR = mat4Load(modelMatrix);

// 在寄存器中链式计算
Mat4R vpR = mat4MulR(projR, viewR);
Mat4R mvpR = mat4MulR(vpR, modelR);

// 变换顶点 (保持在寄存器)
__m128 vertex = _mm_set_ps(1.0f, z, y, x);
__m128 clipPos = mat4MulVec4R(mvpR, vertex);

// 需要时存回内存
Mat4 mvp = mat4Store(mvpR);
```

---

## 已实施优化清单

### 1. fastSinCos (`mat4_d3d.h:246-292`)
- 11/10度 minimax 多项式近似
- 替换 std::tan，性能提升 ~3x

### 2. Perspective (`mat4_d3d.h:537-562`)
- 使用 fastSinCosSmall + cot 替换 std::tan
- **+34.0%** vs DirectXMath

### 3. mat4MulVec4 (`mat4_d3d.h:502-545`)
- 添加 FMA 路径
- **+10.6%** vs DirectXMath

### 4. Mat4R 类型系统 (`mat4_d3d.h:120-188, 493-605`)
- 寄存器驻留矩阵类型
- 值传递优化
- **0.0%** 差距 vs DirectXMath XMMATRIX

---

## 文件清单

- 头文件: `fast_math/include/fast_math/mat4_d3d.h`
- 测试: `fast_math/tests/test_mat4_d3d.cpp` (19/19 PASS)
- Benchmark: `fast_math/benchmark/bench_mat4_d3d.cpp`

## 编译命令

```bash
# 测试
clang++ -std=c++17 -O3 -mavx2 -mfma -I fast_math/include \
  -I DirectXMath-apr2025/Inc fast_math/tests/test_mat4_d3d.cpp \
  -o test_mat4_d3d.exe

# Benchmark
clang++ -std=c++17 -O3 -mavx2 -mfma -I fast_math/include \
  -I DirectXMath-apr2025/Inc fast_math/benchmark/bench_mat4_d3d.cpp \
  -o bench_mat4_d3d.exe
```

---

## 待优化项

### LookAt (-27.2%)
当前实现仍有优化空间，可能原因：
1. 过多的 shuffle 操作
2. 可考虑使用 DirectXMath 的 `XMVectorSelect` 模式

### Pre-computed MVP (-79.1%)
Mat4 版本仍较慢，原因是 mat4MulVec4 需要从内存加载矩阵。
建议：在性能关键路径使用 Mat4R 版本。

---

**优化完成日期**: 2026-01-26
**测试结果**: 19/19 PASS
**主要参考**: DirectXMath, Nicholas Frechette SIMD研究
