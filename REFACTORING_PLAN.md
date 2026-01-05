# 位操作库重构计划

## 目标
符合数学库的文件规范：
- 使用 `.h` 扩展名
- 拆分标量版本和SIMD版本（`操作.h` + `操作_simd.h`）
- Doxygen注释风格
- POD数据 + 自由函数设计

## 文件映射

### 数据结构（POD）
- bitset_static.hpp  -> bitset_static.h  ✓ (纯POD，无成员函数)
- bitset_pmr.hpp     -> bitset_dynamic.h  (保留构造函数)
- bitset_view.hpp    -> bitset_view.h  ✓ (已完成)

### 位运算模块（拆分标量/SIMD）
- bit_ops_core.hpp     -> bit_ops_core.h + bit_ops_core_simd.h
- bit_ops_count.hpp    -> bit_ops_count.h + bit_ops_count_simd.h
- bit_ops_scan.hpp     -> bit_ops_scan.h + bit_ops_scan_simd.h
- bit_ops_range.hpp    -> bit_ops_range.h + bit_ops_range_simd.h
- bit_ops_advanced.hpp -> bit_ops_advanced.h + bit_ops_advanced_simd.h

### 统一头文件
- bit_ops.hpp -> bit_ops.h (包含所有模块)

## 拆分策略

### 标量版本 (*.h)
- 单个元素操作
- 简单循环（无intrinsics）
- 使用 `inline` 函数
- 适用于小型bitset

### SIMD版本 (*_simd.h)
- 批量操作
- AVX2/SSE4.1 intrinsics
- 在 `detail` 命名空间
- 优化技术：循环展开、预取等

## 进度
- [x] bitset_static.h
- [x] bitset_view.h
- [ ] bitset_dynamic.h
- [ ] bit_ops_core.h / bit_ops_core_simd.h
- [ ] bit_ops_count.h / bit_ops_count_simd.h  
- [ ] bit_ops_scan.h / bit_ops_scan_simd.h
- [ ] bit_ops_range.h / bit_ops_range_simd.h
- [ ] bit_ops_advanced.h / bit_ops_advanced_simd.h
- [ ] bit_ops.h
- [ ] 验证编译
