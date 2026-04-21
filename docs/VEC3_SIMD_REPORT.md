# Vec3 SIMD 优化报告

## 概述

本文档记录了 MMath `vec3_simd.h` 的设计与实现，包括 16 字节对齐的 `Vec3A` 类型和批量处理 API。

## 研究发现

### 1. 参考库分析

| 库 | Vec3 大小 | SIMD 策略 |
|---|----------|----------|
| **DirectXMath** | 16 字节 (`XMVECTOR`) | 完全 SIMD，`_mm_dp_ps` / shuffle+add |
| **GLM** | 12/16 字节可选 | 对齐模式使用 `__m128` 存储 |
| **glam-rs** | 16 字节 (`Vec3A`) | 25% 空间换 SIMD 性能 |

### 2. 关键技术发现

来源：[Optimising path tracing with SIMD](https://bitshifter.github.io/2018/06/04/simd-path-tracing/)

1. **Vec3 使用 SIMD 的挑战**：
   - 只使用 3/4 SIMD 通道（75% 占用率）
   - 跨通道操作（dot product）效率不高

2. **数据加载方式对比**：
   - `_mm_load_ps` (对齐) > `_mm_loadu_ps` (非对齐) > `_mm_set_ps` (最慢)

3. **SoA vs AoS**：
   - SoA (Structure of Arrays) 更适合批量 SIMD 处理
   - AoS-to-SoA 转换有开销，适合大批量操作

## 实现内容

### 文件：`vec3_simd.h`

#### 1. Vec3A 类型（16 字节对齐）

```cpp
struct MMATH_ALIGN(16) Vec3A {
    union {
        struct { float x, y, z, _pad; };
        __m128 data;  // SSE 直接操作
    };
};
```

#### 2. 单值 SIMD 操作

| 函数 | 实现策略 |
|-----|---------|
| `vec3AAdd/Sub/Mul/Div` | `_mm_add_ps` 等直接操作 |
| `vec3ADot` | SSE4.1: `_mm_dp_ps`，SSE: shuffle+add |
| `vec3ACross` | shuffle + mul + sub |
| `vec3ANormalize` | dot + sqrt + div |
| `vec3ANormalizeFast` | dot + rsqrt（~0.1% 误差） |
| `vec3AMin/Max` | `_mm_min_ps` / `_mm_max_ps` |

#### 3. 批量处理 API（SSE/AVX2）

| 函数 | 描述 |
|-----|------|
| `vec3AddBatchSimd` | 批量加法，4/8 个 Vec3/次 |
| `vec3DotBatchSimd` | 批量点积 |
| `vec3CrossBatchSimd` | 批量叉积 |
| `vec3NormalizeBatchSimd` | 批量归一化 |
| `vec3LengthBatchSimd` | 批量长度计算 |
| `vec3ScaleBatchSimd` | 批量缩放 |

#### 4. 优化的 Deinterleave/Interleave

使用纯 SSE shuffle 进行 AoS-to-SoA 转换：
- **deinterleave4Vec3**: 3 loads + 9 shuffles
- **interleave4Vec3**: 9 shuffles + 3 stores
- AVX2 版本调用 2x SSE 版本，避免复杂的跨 lane 操作

## 性能测试结果

### 测试环境
- CPU: 支持 AVX2/FMA
- 编译器: MSVC with `/O2 /arch:AVX2 /fp:fast`
- 数据量: 1M 向量，100 次迭代

### 单值操作（100M ops）

| 操作 | Vec3 (标量) | Vec3A (SIMD) | GLM | 结论 |
|-----|------------|--------------|-----|------|
| Dot | 947.9 M/s | 421.8 M/s | 989.7 M/s | **标量更快 2.2x** |
| Cross | 962.0 M/s | 522.3 M/s | 924.8 M/s | **标量更快 1.8x** |
| Normalize | 2265.5 M/s | 425.4 M/s | 2063.8 M/s | **标量更快 5.3x** |
| Length | 2362.4 M/s | 458.1 M/s | 2290.9 M/s | **标量更快 5.2x** |
| Add | 937.7 M/s | 687.6 M/s | 938.1 M/s | **标量更快 1.4x** |

**结论**: 单值 Vec3A 操作全面比标量 Vec3 慢！原因：
1. 编译器已将标量 Vec3 代码自动向量化优化
2. Vec3A 每次访问成员需要从 SIMD 寄存器提取标量，开销大
3. union 存储导致额外的加载/存储指令

### 批量操作（100M ops）

| 操作 | Vec3 标量循环 | Vec3 SIMD 批量 | GLM | 结论 |
|-----|-------------|---------------|-----|------|
| Add | 482.7 M/s | **555.5 M/s** | 542.5 M/s | **SIMD 快 15%** ✓ |
| Normalize | 780.7 M/s | 648.9 M/s | 844.8 M/s | 标量快 17% |
| Dot | 873.1 M/s | 731.8 M/s | 838.5 M/s | 标量快 16% |
| Cross | 532.0 M/s | 528.8 M/s | 589.2 M/s | 几乎相等 |

**结论**: 批量操作中只有 Add 显示 SIMD 优势。原因：
1. deinterleave/interleave 开销（18 个 shuffle）抵消了 SIMD 计算优势
2. 编译器自动向量化标量循环已经很高效

## 使用建议

### ❌ 不建议使用的场景

1. **单值操作后立即访问结果**
   ```cpp
   // 不推荐 - Vec3A 比 Vec3 慢 2-5x
   Vec3A a = vec3ASet(1, 2, 3);
   Vec3A b = vec3ASet(4, 5, 6);
   Vec3A c = vec3AAdd(a, b);
   float x = c.x;  // 从 SIMD 提取标量开销大
   ```

2. **AoS 格式的小批量操作**（< 1000 个向量）
   - deinterleave/interleave 开销太大

### ✓ 建议使用的场景

1. **Transform 管道（不提取中间结果）**
   ```cpp
   // 推荐 - 连续 SIMD 操作
   Vec3A v = vec3ASet(1, 2, 3);
   v = vec3ANormalize(v);
   v = vec3AScale(v, 2.0f);
   v = vec3ACross(v, other);
   // 只在最后提取结果
   float x = v.x;
   ```

2. **与矩阵 SIMD 操作集成**
   - 当 Vec3 是更大 SIMD 管道的一部分时

3. **SoA 数据结构**（未来优化方向）
   ```cpp
   struct Vec3SoA {
       float* x;
       float* y;
       float* z;
       int32_t count;
   };
   // 直接 SIMD 处理，无需 deinterleave
   ```

4. **大批量简单操作**（>10000 向量，Add/Scale）

## 后续优化方向

1. **添加 SoA 数据结构 API**
   - 避免 AoS-to-SoA 转换开销
   - 对粒子系统等场景最有效

2. **流式处理优化**
   - 处理完一批后立即输出，减少内存访问

3. **去掉 union，使用纯 SIMD 类型**
   - 避免成员访问的开销
   - 但牺牲了易用性

4. **考虑移除单值 Vec3A 操作**
   - 只保留批量 API
   - 或明确标注"仅用于 SIMD 管道"

## 总结

当前实现提供了两层 SIMD 优化：

1. **Vec3A（16 字节对齐）**: 单值操作比标量慢 2-5x，**仅适用于 SIMD 管道**
2. **批量 API**: 只有 Add 比标量快 15%，其他操作因 deinterleave 开销反而更慢

**核心发现**: 现代编译器（MSVC with AVX2）对标量 Vec3 代码的自动向量化已经非常高效。手写 SIMD 代码只有在避免数据转换开销时才有优势。

建议：对于一般用途，直接使用标量 `Vec3`，让编译器优化。只有在特定场景（SIMD 管道、SoA 数据）下才使用 SIMD 版本。

## 文件清单

- `include/fast_math/vec3_simd.h` - SIMD Vec3 实现
- `benchmark/bench_vec3_simd.cpp` - 性能测试
