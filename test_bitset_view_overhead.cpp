// ============================================================================
// BitSetView开销测试 - 对比View函数 vs DynamicBitSet成员函数
// ============================================================================

#include <fast_math/bitset_dynamic.h>
#include <fast_math/bitset_view.h>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

using namespace Melosyne;

// ============================================================================
// BitSetView定义（轻量级视图）
// ============================================================================

struct BitSetView {
    uint64_t* data;
    std::size_t bit_count;
    std::size_t word_count;

    // 从DynamicBitSet构造
    constexpr BitSetView(DynamicBitSet& bs) noexcept
        : data(bs.data()),
          bit_count(bs.size()),
          word_count((bs.size() + 63) / 64) {}

    constexpr BitSetView(uint64_t* d, std::size_t bits) noexcept
        : data(d), bit_count(bits), word_count((bits + 63) / 64) {}
};

struct ConstBitSetView {
    const uint64_t* data;
    std::size_t bit_count;
    std::size_t word_count;

    constexpr ConstBitSetView(const DynamicBitSet& bs) noexcept
        : data(bs.data()),
          bit_count(bs.size()),
          word_count((bs.size() + 63) / 64) {}

    constexpr ConstBitSetView(const uint64_t* d, std::size_t bits) noexcept
        : data(d), bit_count(bits), word_count((bits + 63) / 64) {}
};

// ============================================================================
// View版本的函数 - 完全模仿DynamicBitSet::count()的实现
// ============================================================================

[[nodiscard]] inline std::size_t count_view(ConstBitSetView view) noexcept {
    // 完全复制DynamicBitSet::count()的实现（第223-230行）
    const uint64_t* words = view.data;
    std::size_t total = 0;
    for (std::size_t i = 0; i < view.word_count; ++i) {
        total += std::popcount(words[i]);
    }
    return total;
}

[[nodiscard]] inline bool any_view(ConstBitSetView view) noexcept {
    // 完全复制DynamicBitSet::any()的实现（第232-238行）
    const uint64_t* words = view.data;
    for (std::size_t i = 0; i < view.word_count; ++i) {
        if (words[i] != 0) return true;
    }
    return false;
}

[[nodiscard]] inline std::size_t findFirst_view(ConstBitSetView view) noexcept {
    // 完全复制DynamicBitSet::findFirst()的实现（第252-260行）
    const uint64_t* words = view.data;
    for (std::size_t i = 0; i < view.word_count; ++i) {
        if (words[i] != 0) {
            return i * 64 + std::countr_zero(words[i]);
        }
    }
    return view.bit_count;
}

[[nodiscard]] inline std::size_t hammingDistance_view(
    ConstBitSetView a, ConstBitSetView b) noexcept {

    // 模仿两个bitset的XOR + count操作
    std::size_t min_words = std::min(a.word_count, b.word_count);
    std::size_t distance = 0;

    for (std::size_t i = 0; i < min_words; ++i) {
        distance += std::popcount(a.data[i] ^ b.data[i]);
    }

    // 处理长度不同的情况
    for (std::size_t i = min_words; i < a.word_count; ++i) {
        distance += std::popcount(a.data[i]);
    }
    for (std::size_t i = min_words; i < b.word_count; ++i) {
        distance += std::popcount(b.data[i]);
    }

    return distance;
}

// ============================================================================
// 基准测试工具
// ============================================================================

template<typename Func>
double benchmark(const char* name, Func&& func, int iterations = 1000000) {
    // Warmup
    for (int i = 0; i < 1000; ++i) {
        func();
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << name << ": " << ms << " ms ("
              << (ms * 1000000.0 / iterations) << " ns/iter)\n";

    return ms;
}

// ============================================================================
// 测试用例
// ============================================================================

void test_count(std::size_t bit_size) {
    std::cout << "\n========================================\n";
    std::cout << "Test: count() - BitSet size: " << bit_size << " bits\n";
    std::cout << "========================================\n";

    // 创建随机bitset
    DynamicBitSet bs(bit_size);
    std::mt19937_64 rng(12345);
    for (std::size_t i = 0; i < bit_size; ++i) {
        if (rng() & 1) bs.set(i);
    }

    std::size_t result_member = 0;
    std::size_t result_view = 0;

    // 测试成员函数
    double time_member = benchmark("  DynamicBitSet::count()", [&]() {
        result_member = bs.count();
    });

    // 测试View函数
    double time_view = benchmark("  count_view(BitSetView)", [&]() {
        result_view = count_view(bs);  // 隐式构造ConstBitSetView
    });

    // 验证结果
    if (result_member != result_view) {
        std::cout << "❌ ERROR: Results differ! member=" << result_member
                  << " view=" << result_view << "\n";
    } else {
        std::cout << "✅ Results match: " << result_member << " bits set\n";
    }

    // 性能对比
    double overhead = ((time_view - time_member) / time_member) * 100.0;
    std::cout << "Performance: ";
    if (std::abs(overhead) < 2.0) {
        std::cout << "✅ IDENTICAL (";
    } else if (overhead > 0) {
        std::cout << "⚠️ View slower by ";
    } else {
        std::cout << "✅ View faster by ";
    }
    std::cout << std::abs(overhead) << "%)\n";
}

void test_any(std::size_t bit_size) {
    std::cout << "\n========================================\n";
    std::cout << "Test: any() - BitSet size: " << bit_size << " bits\n";
    std::cout << "========================================\n";

    DynamicBitSet bs(bit_size);
    bs.set(bit_size / 2);  // 只设置中间的位

    bool result_member = false;
    bool result_view = false;

    double time_member = benchmark("  DynamicBitSet::any()", [&]() {
        result_member = bs.any();
    });

    double time_view = benchmark("  any_view(BitSetView)", [&]() {
        result_view = any_view(bs);
    });

    if (result_member != result_view) {
        std::cout << "❌ ERROR: Results differ!\n";
    } else {
        std::cout << "✅ Results match: " << (result_member ? "true" : "false") << "\n";
    }

    double overhead = ((time_view - time_member) / time_member) * 100.0;
    std::cout << "Performance: ";
    if (std::abs(overhead) < 2.0) {
        std::cout << "✅ IDENTICAL (";
    } else if (overhead > 0) {
        std::cout << "⚠️ View slower by ";
    } else {
        std::cout << "✅ View faster by ";
    }
    std::cout << std::abs(overhead) << "%)\n";
}

void test_findFirst(std::size_t bit_size) {
    std::cout << "\n========================================\n";
    std::cout << "Test: findFirst() - BitSet size: " << bit_size << " bits\n";
    std::cout << "========================================\n";

    DynamicBitSet bs(bit_size);
    bs.set(bit_size / 3);  // 设置前1/3处的位

    std::size_t result_member = 0;
    std::size_t result_view = 0;

    double time_member = benchmark("  DynamicBitSet::findFirst()", [&]() {
        result_member = bs.findFirst();
    });

    double time_view = benchmark("  findFirst_view(BitSetView)", [&]() {
        result_view = findFirst_view(bs);
    });

    if (result_member != result_view) {
        std::cout << "❌ ERROR: Results differ! member=" << result_member
                  << " view=" << result_view << "\n";
    } else {
        std::cout << "✅ Results match: first bit at " << result_member << "\n";
    }

    double overhead = ((time_view - time_member) / time_member) * 100.0;
    std::cout << "Performance: ";
    if (std::abs(overhead) < 2.0) {
        std::cout << "✅ IDENTICAL (";
    } else if (overhead > 0) {
        std::cout << "⚠️ View slower by ";
    } else {
        std::cout << "✅ View faster by ";
    }
    std::cout << std::abs(overhead) << "%)\n";
}

void test_hammingDistance(std::size_t bit_size) {
    std::cout << "\n========================================\n";
    std::cout << "Test: Hamming Distance - BitSet size: " << bit_size << " bits\n";
    std::cout << "========================================\n";

    DynamicBitSet bs1(bit_size);
    DynamicBitSet bs2(bit_size);

    std::mt19937_64 rng(54321);
    for (std::size_t i = 0; i < bit_size; ++i) {
        if (rng() & 1) bs1.set(i);
        if (rng() & 1) bs2.set(i);
    }

    std::size_t result_manual = 0;
    std::size_t result_view = 0;

    // 成员函数方式（手动XOR + count）
    double time_member = benchmark("  Manual (XOR + count)", [&]() {
        DynamicBitSet temp = bs1;
        temp ^= bs2;
        result_manual = temp.count();
    });

    // View函数方式
    double time_view = benchmark("  hammingDistance_view()", [&]() {
        result_view = hammingDistance_view(bs1, bs2);
    });

    if (result_manual != result_view) {
        std::cout << "❌ ERROR: Results differ! manual=" << result_manual
                  << " view=" << result_view << "\n";
    } else {
        std::cout << "✅ Results match: distance = " << result_manual << "\n";
    }

    double speedup = time_member / time_view;
    std::cout << "Performance: View is " << speedup << "x ";
    if (speedup > 1.02) {
        std::cout << "FASTER ✅\n";
    } else if (speedup < 0.98) {
        std::cout << "slower ⚠️\n";
    } else {
        std::cout << "(same) ✅\n";
    }
}

// ============================================================================
// 主测试
// ============================================================================

int main() {
    std::cout << "============================================\n";
    std::cout << "BitSetView Overhead Test\n";
    std::cout << "Comparing View functions vs Member functions\n";
    std::cout << "============================================\n";

    std::vector<std::size_t> sizes = {256, 1024, 4096, 16384, 65536};

    for (std::size_t size : sizes) {
        test_count(size);
    }

    for (std::size_t size : sizes) {
        test_any(size);
    }

    for (std::size_t size : sizes) {
        test_findFirst(size);
    }

    for (std::size_t size : sizes) {
        test_hammingDistance(size);
    }

    std::cout << "\n========================================\n";
    std::cout << "Summary\n";
    std::cout << "========================================\n";
    std::cout << "If overhead is <2%, BitSetView is a ZERO-COST abstraction ✅\n";
    std::cout << "Modern compilers (GCC/Clang/MSVC with -O2/-O3) will inline\n";
    std::cout << "the view construction and generate identical assembly code.\n";

    return 0;
}
