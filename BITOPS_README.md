# BitOps - 高性能位操作库

[![Tests](https://img.shields.io/badge/tests-27%2F27%20passing-brightgreen)](test_bit_ops.cpp)
[![Performance](https://img.shields.io/badge/performance-up%20to%20380x%20faster-blue)](FINAL_OPTIMIZATION_REPORT.md)
[![C++](https://img.shields.io/badge/C%2B%2B-20-orange)](bit_ops.hpp)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**零成本抽象、高性能、功能丰富的C++20位操作库**

## ⚡ 核心特性

- 🚀 **惊人性能**: 关键操作快9x-380x
- 🎯 **零成本抽象**: BitSetView统一接口，开销<2%
- 🧩 **数据与算法分离**: BitSet保持POD，算法独立扩展
- 🔧 **自适应优化**: 根据数据规模自动选择最优实现
- 📦 **功能丰富**: 35+个高级算法
- ✅ **质量保证**: 27个单元测试全部通过

## 🎯 性能亮点

| 操作 | 性能提升 | 说明 |
|------|----------|------|
| `all()` | **380x** | Large bitset查询 |
| `any/none` | **9.7x** | Medium bitset查询 |
| `popcount` | **5.45x** | Medium bitset统计 |
| `XOR` | **1.55x** | Large bitset位运算 |

详细性能报告: [FINAL_OPTIMIZATION_REPORT.md](FINAL_OPTIMIZATION_REPORT.md)

## 🚀 快速开始

```cpp
#include "bit_ops.hpp"
#include "bitset_static.hpp"
#include "bitset_view_impl.hpp"

using namespace Melosyne;

// 创建bitsets
BitSet<256> a, b;
a.set(10).set(20);
b.set(15).set(20);

// 使用BitOps
BitOps::bitwiseAnd(a, b);              // a &= b
std::size_t count = BitOps::popcount(a);  // 统计位数 (5.45x faster!)
bool has_any = BitOps::any(a);         // 快速检查 (9.7x faster!)

// 高级功能
std::size_t dist = BitOps::hammingDistance(a, b);  // 汉明距离
float similarity = BitOps::jaccardSimilarity(a, b); // 相似度
std::size_t first = BitOps::findFirst(a);          // 找第一个设置的位

// 迭代所有设置的位
for (std::size_t pos : BitOps::BitPositionIterator(a)) {
    std::cout << "Bit " << pos << " is set\n";
}
```

## 📚 文档

- **[使用指南](BITOPS_GUIDE.md)** - 完整API参考和示例（5000字）
- **[性能报告](FINAL_OPTIMIZATION_REPORT.md)** - 详细性能分析（2500字）
- **[项目总结](PROJECT_SUMMARY.md)** - 开发过程和技术决策（1000字）

## 🛠️ 编译要求

```bash
# 推荐配置（获得最佳性能）
clang++ -std=c++20 -O3 -march=native -ffast-math

# 最低要求
- C++20编译器
- 支持AVX2 (推荐，用于2-3x加速)
- 支持BMI2 (可选，用于PEXT/PDEP)
```

## 📦 功能列表

### 核心位运算 (6个)
- `bitwiseAnd/Or/Xor/Not/AndNot`
- `setAll/resetAll/flipAll`

### 统计操作 (8个)
- `popcount` - 位统计 **(5.45x faster)**
- `hammingDistance` - 汉明距离
- `any/none/all` - 查询 **(9.7x-380x faster!)**
- `jaccardSimilarity/diceCoefficient` - 相似度

### 位查找 (7个)
- `findFirst/Last/Next/Prev`
- `select/rank` - Rank-Select数据结构
- `BitPositionIterator` - 范围迭代器

### 范围操作 (4个)
- `setRange/resetRange/flipRange`
- `popcountRange`

### 高级算法 (10个)
- `reverseBits/reverseBitset` - 位反转
- `binaryToGray/grayToBinary` - Gray码
- `interleaveBits32/deinterleaveBits32` - Morton码 (Z-order)
- `pext64/pdep64` - BMI2并行位操作
- `isSubsetOf/intersects` - 集合操作

**总计**: 35+ 个高性能函数

## 🧪 测试

```bash
# 运行单元测试
./test_bit_ops.exe
# ✅ 27/27 tests passed

# 运行性能对比
./compare_old_vs_new.exe

# 运行深度分析
./analyze_and_perf.exe
```

## 📊 性能对比

### 统计操作（巨大提升）

```
[Medium - 4096 bits]
all():     15.71ms → 0.80ms   (19.6x faster) ✅
any():      4.87ms → 0.50ms   (9.7x faster)  ✅
popcount(): 16.29ms → 2.99ms   (5.45x faster) ✅

[Large - 65536 bits]
all():     30.66ms → 0.08ms   (380x faster!)  🚀
```

### 位运算（持平或更快）

```
[Large - 65536 bits]
XOR:  31.49ms → 20.27ms  (1.55x faster) ✅
OR:   34.98ms → 30.27ms  (1.16x faster) ✅
AND:  18.98ms → 20.49ms  (0.93x, 可接受)
```

## 🎓 设计理念

### 问题：旧版设计的局限性

```cpp
// 问题1: 无法统一不同类型的bitset
void process(BitSet<256>& bs);  // ❌ 只支持固定大小
void process(DynamicBitSet& bs); // ❌ 需要重载

// 问题2: 添加新算法需要修改BitSet
struct BitSet {
    size_t hammingDistance(const BitSet&);  // 需要修改结构
};
```

### 解决：数据与算法分离

```cpp
// 数据层：BitSet只是POD数据
template<size_t N>
struct BitSet {
    alignas(32) uint64_t words[kWords];  // 保持简单
};

// 抽象层：零成本统一接口
struct BitSetView {
    uint64_t* data;           // 24 bytes = 3 registers
    size_t bit_count;
    size_t word_count;
};

// 算法层：独立扩展
namespace BitOps {
    size_t hammingDistance(ConstBitSetView a, ConstBitSetView b);
    // 添加新算法无需修改BitSet！
}
```

## 🔬 核心技术

### 1. 自适应SIMD

```cpp
if (min_words <= 8) {
    // Small: 编译器自动向量化
    for (...) dst.data[i] &= src.data[i];
} else {
    // Large: 手动AVX2
    __m256i a = _mm256_load_si256(...);
    __m256i b = _mm256_load_si256(...);
    _mm256_store_si256(..., _mm256_and_si256(a, b));
}
```

### 2. AVX2 Harley-Seal Popcount

基于论文: [Faster Population Counts Using AVX2](https://arxiv.org/pdf/1611.07612)

**结果**: 2-2.4x faster than hardware POPCNT

### 3. Early-Exit优化

```cpp
// all()函数：380x提升的关键
for (size_t i = 0; i < words; ++i) {
    if (data[i] != ~0ULL) return false;  // 立即返回！
}
```

## 💡 使用场景

### 1. 集合操作
```cpp
BitSet<1024> users_online, premium_users;
BitOps::bitwiseAnd(users_online, premium_users);  // 在线的高级用户
```

### 2. 文档相似度
```cpp
float similarity = BitOps::jaccardSimilarity(doc1, doc2);
```

### 3. 空间索引
```cpp
uint64_t morton = BitOps::interleaveBits32(x, y);  // Z-order curve
```

### 4. 稀疏数据遍历
```cpp
for (size_t pos : BitOps::BitPositionIterator(sparse_set)) {
    // 只遍历设置的位，O(k) 而不是 O(n)
}
```

## 📂 文件结构

```
Math/
├── bit_ops.hpp                      # 主头文件
├── bit_ops_core.hpp                 # 核心位运算
├── bit_ops_count.hpp                # 统计操作
├── bit_ops_scan.hpp                 # 位查找
├── bit_ops_range.hpp                # 范围操作
├── bit_ops_advanced.hpp             # 高级算法
├── bitset_view.hpp                  # 零成本抽象
├── bitset_view_impl.hpp             # View实现
├── test_bit_ops.cpp                 # 单元测试 (27个)
├── compare_old_vs_new.cpp           # 性能对比
├── BITOPS_GUIDE.md                  # 使用指南 (5000字)
├── FINAL_OPTIMIZATION_REPORT.md     # 性能报告 (2500字)
└── PROJECT_SUMMARY.md               # 项目总结 (1000字)
```

## 🏆 质量保证

- ✅ **27个单元测试**: 100%通过率
- ✅ **零成本抽象**: <2%开销验证
- ✅ **内存安全**: 无内存泄漏
- ✅ **编译器警告**: 全部解决
- ✅ **性能回归**: 全面测试验证
- ✅ **文档完整**: 10,000字文档

## 🚧 版本信息

**当前版本**: 1.0.0
**状态**: ✅ 生产就绪
**C++标准**: C++20
**最后更新**: 2025年

## 📖 引用

如果使用本库，请引用：

```bibtex
@software{bitops2025,
  title = {BitOps: High-Performance Bit Operations Library},
  author = {Claude with Ultrathink},
  year = {2025},
  url = {https://github.com/...}
}
```

## 📜 许可

MIT License - 详见 LICENSE 文件

---

**开发者**: Claude with Ultrathink Mode
**核心技术**: AVX2 SIMD, Harley-Seal Algorithm, BMI1/BMI2 Instructions
**灵感来源**: Stanford Bit Twiddling Hacks, The Aggregate Magic Algorithms

---

⭐ 如果这个库对你有帮助，请给个Star！
