module;

#include "fast_math/config_macros.h"
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>
#include <bit>
#include <iterator>
#include <type_traits>
#include <memory>
#include <memory_resource>
#include <cfloat>

export module fast_math.bitset_dynamic;

export {


namespace Melosyne {

// ============================================================================
// DynamicBitSet - 支持std::pmr的运行时长度bitset
//
// 设计理念：数据与算法分离
// - DynamicBitSet只是数据容器，提供最基本的位访问操作
// - 所有数学运算（AND/OR/XOR、popcount、findFirst等）由BitOps命名空间提供
// - 通过BitSetView可统一处理BitSet<N>和DynamicBitSet
// ============================================================================

class DynamicBitSet {
public:
    // 小对象优化 - 256位以下使用栈内存
    static constexpr std::size_t kLocalWords = 4; // 256位
    static constexpr std::size_t kLocalBits = kLocalWords * 64;

    // ========================================================================
    // 构造/析构
    // ========================================================================

    // 默认构造 - 0位，使用默认memory_resource
    DynamicBitSet() noexcept
        : bit_count_(0), word_count_(0), is_local_(true),
          owns_memory_(false), memory_resource_(std::pmr::get_default_resource()) {
        std::memset(local_words_, 0, sizeof(local_words_));
    }

    // 指定位数构造 - 使用默认memory_resource
    explicit DynamicBitSet(std::size_t num_bits_)
        : bit_count_(num_bits_), word_count_((num_bits_ + 63) / 64),
          memory_resource_(std::pmr::get_default_resource()) {

        if (word_count_ <= kLocalWords) {
            is_local_ = true;
            owns_memory_ = false;
            std::memset(local_words_, 0, sizeof(local_words_));
        } else {
            is_local_ = false;
            owns_memory_ = true;
            remote_words_ = static_cast<uint64_t*>(
                memory_resource_->allocate(word_count_ * sizeof(uint64_t), alignof(uint64_t))
            );
            std::memset(remote_words_, 0, word_count_ * sizeof(uint64_t));
        }
    }

    // PMR构造 - 使用指定的memory_resource
    DynamicBitSet(std::size_t num_bits_, std::pmr::memory_resource* mr_)
        : bit_count_(num_bits_), word_count_((num_bits_ + 63) / 64),
          memory_resource_(mr_) {

        if (word_count_ <= kLocalWords) {
            is_local_ = true;
            owns_memory_ = false;
            std::memset(local_words_, 0, sizeof(local_words_));
        } else {
            is_local_ = false;
            owns_memory_ = true;
            remote_words_ = static_cast<uint64_t*>(
                memory_resource_->allocate(word_count_ * sizeof(uint64_t), alignof(uint64_t))
            );
            std::memset(remote_words_, 0, word_count_ * sizeof(uint64_t));
        }
    }

    // 拷贝构造
    DynamicBitSet(const DynamicBitSet& other_)
        : bit_count_(other_.bit_count_),
          word_count_(other_.word_count_),
          is_local_(other_.is_local_),
          memory_resource_(other_.memory_resource_) {

        if (is_local_) {
            owns_memory_ = false;
            std::memcpy(local_words_, other_.local_words_, sizeof(local_words_));
        } else {
            owns_memory_ = true;
            remote_words_ = static_cast<uint64_t*>(
                memory_resource_->allocate(word_count_ * sizeof(uint64_t), alignof(uint64_t))
            );
            std::memcpy(remote_words_, other_.remote_words_,
                       word_count_ * sizeof(uint64_t));
        }
    }

    // 移动构造
    DynamicBitSet(DynamicBitSet&& other_) noexcept
        : bit_count_(other_.bit_count_),
          word_count_(other_.word_count_),
          is_local_(other_.is_local_),
          owns_memory_(other_.owns_memory_),
          memory_resource_(other_.memory_resource_) {

        if (is_local_) {
            std::memcpy(local_words_, other_.local_words_, sizeof(local_words_));
        } else {
            remote_words_ = other_.remote_words_;
            other_.remote_words_ = nullptr;
        }
        other_.bit_count_ = 0;
        other_.word_count_ = 0;
        other_.owns_memory_ = false;
    }

    // 析构
    ~DynamicBitSet() {
        if (!is_local_ && owns_memory_ && remote_words_) {
            memory_resource_->deallocate(remote_words_,
                                        word_count_ * sizeof(uint64_t),
                                        alignof(uint64_t));
        }
    }

    // 赋值运算符
    DynamicBitSet& operator=(const DynamicBitSet& other_) {
        if (this != &other_) {
            // 释放旧内存
            if (!is_local_ && owns_memory_ && remote_words_) {
                memory_resource_->deallocate(remote_words_,
                                            word_count_ * sizeof(uint64_t),
                                            alignof(uint64_t));
            }

            bit_count_ = other_.bit_count_;
            word_count_ = other_.word_count_;
            is_local_ = other_.is_local_;
            memory_resource_ = other_.memory_resource_;

            if (is_local_) {
                owns_memory_ = false;
                std::memcpy(local_words_, other_.local_words_, sizeof(local_words_));
            } else {
                owns_memory_ = true;
                remote_words_ = static_cast<uint64_t*>(
                    memory_resource_->allocate(word_count_ * sizeof(uint64_t), alignof(uint64_t))
                );
                std::memcpy(remote_words_, other_.remote_words_,
                           word_count_ * sizeof(uint64_t));
            }
        }
        return *this;
    }

    DynamicBitSet& operator=(DynamicBitSet&& other_) noexcept {
        if (this != &other_) {
            if (!is_local_ && owns_memory_ && remote_words_) {
                memory_resource_->deallocate(remote_words_,
                                            word_count_ * sizeof(uint64_t),
                                            alignof(uint64_t));
            }

            bit_count_ = other_.bit_count_;
            word_count_ = other_.word_count_;
            is_local_ = other_.is_local_;
            owns_memory_ = other_.owns_memory_;
            memory_resource_ = other_.memory_resource_;

            if (is_local_) {
                std::memcpy(local_words_, other_.local_words_, sizeof(local_words_));
            } else {
                remote_words_ = other_.remote_words_;
                other_.remote_words_ = nullptr;
            }

            other_.bit_count_ = 0;
            other_.word_count_ = 0;
            other_.owns_memory_ = false;
        }
        return *this;
    }

    // ========================================================================
    // 基础位访问操作
    // ========================================================================

    [[nodiscard]] bool test(std::size_t pos_) const noexcept {
        const uint64_t* words = getWords();
        return (words[pos_ / 64] >> (pos_ % 64)) & 1;
    }

    DynamicBitSet& set(std::size_t pos_) noexcept {
        uint64_t* words = getWords();
        words[pos_ / 64] |= (uint64_t{1} << (pos_ % 64));
        return *this;
    }

    DynamicBitSet& reset(std::size_t pos_) noexcept {
        uint64_t* words = getWords();
        words[pos_ / 64] &= ~(uint64_t{1} << (pos_ % 64));
        return *this;
    }

    DynamicBitSet& flip(std::size_t pos_) noexcept {
        uint64_t* words = getWords();
        words[pos_ / 64] ^= (uint64_t{1} << (pos_ % 64));
        return *this;
    }

    DynamicBitSet& setAll() noexcept {
        uint64_t* words = getWords();
        std::memset(words, 0xFF, word_count_ * sizeof(uint64_t));
        clearUnusedBits();
        return *this;
    }

    DynamicBitSet& resetAll() noexcept {
        uint64_t* words = getWords();
        std::memset(words, 0, word_count_ * sizeof(uint64_t));
        return *this;
    }

    DynamicBitSet& flipAll() noexcept {
        uint64_t* words = getWords();
        for (std::size_t i = 0; i < word_count_; ++i) {
            words[i] = ~words[i];
        }
        clearUnusedBits();
        return *this;
    }

    // ========================================================================
    // 比较操作
    // ========================================================================

    [[nodiscard]] bool operator==(const DynamicBitSet& rhs_) const noexcept {
        if (bit_count_ != rhs_.bit_count_) return false;
        const uint64_t* words = getWords();
        const uint64_t* rhs_words = rhs_.getWords();
        for (std::size_t i = 0; i < word_count_; ++i) {
            if (words[i] != rhs_words[i]) return false;
        }
        return true;
    }

    [[nodiscard]] bool operator!=(const DynamicBitSet& rhs_) const noexcept {
        return !(*this == rhs_);
    }

    // ========================================================================
    // 信息查询
    // ========================================================================

    [[nodiscard]] std::size_t size() const noexcept {
        return bit_count_;
    }

    [[nodiscard]] std::size_t wordCount() const noexcept {
        return word_count_;
    }

    [[nodiscard]] std::size_t capacity() const noexcept {
        return word_count_ * 64;
    }

    [[nodiscard]] bool isSso() const noexcept {
        return is_local_;
    }

    [[nodiscard]] std::size_t memoryUsage() const noexcept {
        return is_local_ ? sizeof(local_words_) : word_count_ * sizeof(uint64_t);
    }

    [[nodiscard]] std::pmr::memory_resource* getMemoryResource() const noexcept {
        return memory_resource_;
    }

    // ========================================================================
    // 数据访问
    // ========================================================================

    [[nodiscard]] const uint64_t* data() const noexcept {
        return getWords();
    }

    [[nodiscard]] uint64_t* data() noexcept {
        return getWords();
    }

private:
    uint64_t* getWords() noexcept {
        return is_local_ ? local_words_ : remote_words_;
    }

    const uint64_t* getWords() const noexcept {
        return is_local_ ? local_words_ : remote_words_;
    }

    void clearUnusedBits() noexcept {
        if (bit_count_ == 0) return;
        std::size_t unused_bits = word_count_ * 64 - bit_count_;
        if (unused_bits > 0) {
            uint64_t* words = getWords();
            words[word_count_ - 1] &= ~(~uint64_t{0} << (64 - unused_bits));
        }
    }

    // ========================================================================
    // 数据成员
    // ========================================================================

    std::size_t bit_count_;
    std::size_t word_count_;
    bool is_local_;
    bool owns_memory_;
    std::pmr::memory_resource* memory_resource_;

    union {
        uint64_t local_words_[kLocalWords];  // SSO: 栈上256位
        uint64_t* remote_words_;              // PMR分配
    };
};

} // namespace Melosyne
}
