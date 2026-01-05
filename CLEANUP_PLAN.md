# 位操作库清理计划

## 当前状态
- ✅ 已完成14个.h文件的重构（标量.h + SIMD _simd.h）
- ✅ 性能回归测试 6/6 通过
- ❌ 旧的10个.hpp文件仍然存在
- ❌ 测试文件依赖.hpp的成员函数API

## 问题分析
1. **旧.hpp文件**: `bitset_static.hpp` 等提供成员函数API（`a.set(10)`, `a.test(50)`）
2. **新.h文件**: `bitset_static.h` 是纯POD结构体，只有 `words` 数组，无成员函数
3. **测试代码**: `test_bit_ops.cpp` 等大量使用成员函数API（约100+处调用）

## 清理方案

### 步骤1：保留现有.hpp文件作为兼容层（推荐）

**原因**:
- 成员函数API更符合C++习惯（`a.set(10)` vs `BitOps::set(a, 10)`）
- 测试代码可读性更好
- 避免大量代码修改
- 兼容现有代码

**实施**:
1. 保留 `bitset_static.hpp` 作为便利层
2. 让.hpp内部使用.h的纯POD + 自由函数实现
3. 删除其他已废弃的.hpp文件（bit_ops*.hpp等）

### 步骤2：添加单比特操作自由函数（如果选择完全清理）

在 `bit_ops_core.h` 中添加:
```cpp
namespace BitOps {

// 单比特操作
inline void set(BitSetView view, std::size_t pos) noexcept {
    view.data[pos / 64] |= (uint64_t{1} << (pos % 64));
}

inline void reset(BitSetView view, std::size_t pos) noexcept {
    view.data[pos / 64] &= ~(uint64_t{1} << (pos % 64));
}

inline void flip(BitSetView view, std::size_t pos) noexcept {
    view.data[pos / 64] ^= (uint64_t{1} << (pos % 64));
}

[[nodiscard]] inline bool test(ConstBitSetView view, std::size_t pos) noexcept {
    return (view.data[pos / 64] >> (pos % 64)) & 1;
}

} // namespace BitOps
```

### 步骤3：批量替换测试代码（如果选择完全清理）

使用sed批量替换：
```bash
# a.set(X) → BitOps::set(a, X)
sed -i 's/\([a-z_][a-z0-9_]*\)\.set(\([0-9]*\))/BitOps::set(\1, \2)/g'

# a.test(X) → BitOps::test(a, X)
sed -i 's/\([a-z_][a-z0-9_]*\)\.test(\([0-9]*\))/BitOps::test(\1, \2)/g'
```

### 步骤4：删除废弃文件

**需要删除的.hpp文件**:
```
include/fast_math/bit_ops.hpp
include/fast_math/bit_ops_advanced.hpp
include/fast_math/bit_ops_core.hpp
include/fast_math/bit_ops_count.hpp
include/fast_math/bit_ops_range.hpp
include/fast_math/bit_ops_scan.hpp
include/fast_math/bitset_view_impl.hpp
# 可能保留：
include/fast_math/bitset_static.hpp  # 兼容层
include/fast_math/bitset_pmr.hpp     # 兼容层
include/fast_math/bitset_view.hpp    # 兼容层
```

## 推荐决策

**保留3个便利层.hpp文件**:
1. `bitset_static.hpp` - 提供成员函数API（继承或包装纯POD）
2. `bitset_pmr.hpp` → 重命名为 `bitset_dynamic.hpp`
3. `bitset_view.hpp` - BitSetView定义

**删除7个已废弃.hpp文件**:
- `bit_ops*.hpp` (6个文件) - 已完全迁移到.h
- `bitset_view_impl.hpp` - 已合并到bitset_view.h

## 决策点

请选择清理策略：

**A. 保守清理（推荐）**:
- 保留3个bitset相关.hpp作为便利API层
- 删除7个bit_ops相关.hpp
- 测试代码无需修改
- 兼容性最好

**B. 彻底清理**:
- 删除所有10个.hpp文件
- 添加单比特操作自由函数
- 批量修改所有测试代码
- 纯度最高但工作量大
