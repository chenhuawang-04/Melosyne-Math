# 位操作库文件结构速查

## 📁 E:\Project\MelosyneTest\Math\fast_math\include\fast_math\

### 数据结构 (3个文件)
```
✅ bitset_static.h        - BitSet<N> 静态位集 (POD)
✅ bitset_dynamic.h       - DynamicBitSet 动态位集 (PMR)
✅ bitset_view.h          - BitSetView 统一视图
```

### 核心位操作 (2个文件)
```
✅ bit_ops_core.h         - AND/OR/XOR/NOT/ANDNOT (标量)
✅ bit_ops_core_simd.h    - 核心操作 AVX2 优化
```

### 统计操作 (2个文件)
```
✅ bit_ops_count.h        - popcount, hamming, any/none/all (标量)
✅ bit_ops_count_simd.h   - Harley-Seal + AVX2
```

### 扫描操作 (2个文件)
```
✅ bit_ops_scan.h         - find, select, rank, iterator (标量)
✅ bit_ops_scan_simd.h    - select/rank AVX2 优化
```

### 范围操作 (2个文件)
```
✅ bit_ops_range.h        - setRange, resetRange, flipRange (标量)
✅ bit_ops_range_simd.h   - 范围操作 AVX2 批量处理
```

### 高级操作 (2个文件)
```
✅ bit_ops_advanced.h     - reverse, rotate, PEXT/PDEP, Gray code (标量)
✅ bit_ops_advanced_simd.h - 高级操作 AVX2 bulk ops
```

### 统一头文件 (1个文件)
```
✅ bit_ops.h              - 完整库统一入口 + 文档
```

---

## 📊 总计

- **总文件数**: 14个新头文件
- **代码行数**: ~5000行（含详细文档）
- **覆盖算法**: 27种位操作算法

---

## 🚀 快速开始

### 最简单的使用方式
```cpp
#include <fast_math/bit_ops.h>

// 就这一个头文件，包含所有功能！
```

### 按需包含（减少编译时间）
```cpp
// 只需要核心操作
#include <fast_math/bitset_static.h>
#include <fast_math/bit_ops_core.h>

// 只需要统计功能
#include <fast_math/bitset_view.h>
#include <fast_math/bit_ops_count.h>
```

### SIMD优化（自动启用）
```cpp
// 编译时定义 __AVX2__ 即可
// GCC/Clang: -mavx2
// MSVC: /arch:AVX2

#include <fast_math/bit_ops.h>

// 所有 "*Optimized" 函数自动使用AVX2
BitOps::popcountOptimized(bitset);
```

---

## 📚 函数速查表

### 核心操作 (bit_ops_core.h)
```cpp
bitwiseAnd(dst, src)         // dst &= src
bitwiseOr(dst, src)          // dst |= src
bitwiseXor(dst, src)         // dst ^= src
bitwiseNot(view)             // ~view
bitwiseAndNot(dst, src)      // dst &= ~src
copy(dst, src)               // dst = src
equal(a, b) -> bool          // a == b
```

### 统计操作 (bit_ops_count.h)
```cpp
popcount(view) -> size_t              // 计数设置的位
hammingDistance(a, b) -> size_t       // 汉明距离
any(view) -> bool                     // 是否有任何位设置
none(view) -> bool                    // 是否没有位设置
all(view) -> bool                     // 是否所有位设置
unionCount(a, b) -> size_t            // |A ∪ B|
intersectionCount(a, b) -> size_t     // |A ∩ B|
jaccardSimilarity(a, b) -> float      // Jaccard相似度
diceCoefficient(a, b) -> float        // Dice系数
isSubsetOf(a, b) -> bool              // A ⊆ B
intersects(a, b) -> bool              // A ∩ B ≠ ∅
```

### 扫描操作 (bit_ops_scan.h)
```cpp
findFirst(view) -> size_t             // 第一个设置的位
findLast(view) -> size_t              // 最后一个设置的位
findNext(view, pos) -> size_t         // pos之后下一个设置的位
findPrev(view, pos) -> size_t         // pos之前上一个设置的位
findFirstZero(view) -> size_t         // 第一个为0的位
findLastZero(view) -> size_t          // 最后一个为0的位
select(view, k) -> size_t             // 第k个设置的位
rank(view, pos) -> size_t             // [0, pos)中设置的位数

// 迭代器
for (size_t pos : BitPositionIterator(view)) {
    // 遍历所有设置的位
}
```

### 范围操作 (bit_ops_range.h)
```cpp
setRange(view, start, end)            // 设置[start, end)
resetRange(view, start, end)          // 清除[start, end)
flipRange(view, start, end)           // 翻转[start, end)
testAll(view, start, end) -> bool     // [start, end)全为1?
testAny(view, start, end) -> bool     // [start, end)有1?
testNone(view, start, end) -> bool    // [start, end)全为0?
popcountRange(view, start, end)       // [start, end)中1的个数
```

### 高级操作 (bit_ops_advanced.h)
```cpp
// 位反转
reverseBits64(v) -> uint64_t          // 反转64位
reverseBits32/16/8(v)                 // 反转32/16/8位
reverse(view)                         // 反转整个位集

// 旋转
rotateLeft(view, shift)               // 循环左移
rotateRight(view, shift)              // 循环右移

// 字节交换
byteSwap(view)                        // 字节序转换

// 奇偶校验
parity(view) -> bool                  // XOR所有位

// BMI2指令
pext64(src, mask) -> uint64_t         // 并行位提取
pdep64(src, mask) -> uint64_t         // 并行位存储

// Gray码
binaryToGray(binary) -> uint64_t      // 二进制 -> Gray码
grayToBinary(gray) -> uint64_t        // Gray码 -> 二进制

// Morton码 (Z-order curve)
interleaveBits32(x, y) -> uint64_t    // 交错位 (2D Morton)
deinterleaveBits(morton, x, y)        // 反交错

// 排列
nextPermutation(v) -> uint64_t        // Gosper's hack

// 散布/收集
scatterBits(dst, src, indices, count) // 按索引散布
gatherBits(src, indices, count)       // 按索引收集
```

---

## 🎯 性能提示

### 使用 "Optimized" 后缀获得最佳性能
```cpp
// ✅ 推荐 - 自动SIMD派发
popcountOptimized(bs);
bitwiseAndOptimized(bs1, bs2);
setRangeOptimized(bs, 0, 1000);

// ⚠️ 仅在需要强制标量时使用
popcount(bs);
bitwiseAnd(bs1, bs2);
```

### 以下操作始终使用标量（已经最优）
```cpp
// 1-周期指令
findFirst(bs);
findLast(bs);

// 早退优化
any(bs);
none(bs);
testAny(bs, 0, 100);

// 单字SWAR
reverseBits64(v);
pext64(src, mask);
binaryToGray(v);
```

---

## 🔧 编译选项

### GCC/Clang
```bash
g++ -O3 -march=native -mavx2 -mbmi2 main.cpp
```

### MSVC
```bash
cl /O2 /arch:AVX2 main.cpp
```

### CMake
```cmake
target_compile_options(your_target PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2 -mbmi2>
    $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>
)
```

---

## 📖 文档位置

- **REFACTORING_COMPLETE.md** - 完整重构总结 + 性能数据
- **REFACTORING_GUIDE.md** - 重构指南 + 模式参考
- **bit_ops.h** - 完整API文档 + 使用示例
- **各模块头文件** - 函数级详细文档（Doxygen）

---

## ✅ 验证清单

```bash
# 1. 检查所有文件存在
ls -la fast_math/include/fast_math/*.h

# 2. 编译测试
cd build && cmake --build . --config Release

# 3. 运行测试
./build/bin/Release/bit_ops_test.exe

# 4. 性能测试
./build/bin/Release/bit_ops_benchmark.exe
```

---

**状态**: ✅ 所有14个文件已完成
**准备就绪**: 可以开始编译和测试
**下一步**: 运行验证测试，确认所有功能正常
