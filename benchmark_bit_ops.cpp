// ============================================================================
// BitOps Performance Benchmark
// ============================================================================

#include <fast_math/bit_ops.h>
#include <fast_math/bitset_static.h>
#include <fast_math/bitset_dynamic.h>
#include <fast_math/bitset_view.h>
#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>

using namespace Melosyne;

// Benchmark timer
class Timer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start_time;

public:
    Timer() : start_time(Clock::now()) {}

    double elapsedMs() const {
        auto end = Clock::now();
        return std::chrono::duration<double, std::milli>(end - start_time).count();
    }

    void reset() { start_time = Clock::now(); }
};

// Initialize bitset with random data
template<std::size_t N>
void randomize(BitSet<N>& bs, std::mt19937_64& rng) {
    for (auto& w : bs.words) {
        w = rng();
    }
}

// ============================================================================
// Core Operations Benchmark
// ============================================================================

template<std::size_t N>
void benchmarkCoreOps(const char* name, std::size_t iterations) {
    std::cout << "\n[" << name << " - " << N << " bits]\n";
    std::cout << std::string(60, '-') << "\n";

    std::mt19937_64 rng(12345);
    BitSet<N> a, b;
    randomize(a, rng);
    randomize(b, rng);

    Timer timer;
    double elapsed;

    // AND
    timer.reset();
    for (std::size_t i = 0; i < iterations; ++i) {
        BitOps::bitwiseAnd(a, b);
    }
    elapsed = timer.elapsedMs();
    std::cout << "  AND:    " << std::setw(8) << std::fixed << std::setprecision(2)
              << elapsed << " ms (" << (iterations / elapsed * 1000) << " ops/s)\n";

    randomize(a, rng);

    // OR
    timer.reset();
    for (std::size_t i = 0; i < iterations; ++i) {
        BitOps::bitwiseOr(a, b);
    }
    elapsed = timer.elapsedMs();
    std::cout << "  OR:     " << std::setw(8) << elapsed << " ms ("
              << (iterations / elapsed * 1000) << " ops/s)\n";

    randomize(a, rng);

    // XOR
    timer.reset();
    for (std::size_t i = 0; i < iterations; ++i) {
        BitOps::bitwiseXor(a, b);
    }
    elapsed = timer.elapsedMs();
    std::cout << "  XOR:    " << std::setw(8) << elapsed << " ms ("
              << (iterations / elapsed * 1000) << " ops/s)\n";

    randomize(a, rng);

    // NOT
    timer.reset();
    for (std::size_t i = 0; i < iterations; ++i) {
        BitOps::bitwiseNot(a);
    }
    elapsed = timer.elapsedMs();
    std::cout << "  NOT:    " << std::setw(8) << elapsed << " ms ("
              << (iterations / elapsed * 1000) << " ops/s)\n";
}

// ============================================================================
// Popcount Benchmark
// ============================================================================

template<std::size_t N>
void benchmarkPopcount(const char* name, std::size_t iterations) {
    std::cout << "\n[Popcount - " << name << " - " << N << " bits]\n";
    std::cout << std::string(60, '-') << "\n";

    std::mt19937_64 rng(12345);
    BitSet<N> a;
    randomize(a, rng);

    Timer timer;
    volatile std::size_t sum = 0;  // Prevent optimization

    timer.reset();
    for (std::size_t i = 0; i < iterations; ++i) {
        sum += BitOps::popcount(a);
    }
    double elapsed = timer.elapsedMs();

    std::cout << "  Popcount: " << std::setw(8) << std::fixed << std::setprecision(2)
              << elapsed << " ms (" << (iterations / elapsed * 1000) << " ops/s)\n";
    std::cout << "  Result: " << (sum / iterations) << " bits set (avg)\n";

#if defined(__AVX2__)
    std::cout << "  Implementation: AVX2 (Harley-Seal algorithm)\n";
#else
    std::cout << "  Implementation: Scalar (std::popcount)\n";
#endif
}

// ============================================================================
// Hamming Distance Benchmark
// ============================================================================

template<std::size_t N>
void benchmarkHamming(const char* name, std::size_t iterations) {
    std::cout << "\n[Hamming Distance - " << name << " - " << N << " bits]\n";
    std::cout << std::string(60, '-') << "\n";

    std::mt19937_64 rng(12345);
    BitSet<N> a, b;
    randomize(a, rng);
    randomize(b, rng);

    Timer timer;
    volatile std::size_t sum = 0;

    timer.reset();
    for (std::size_t i = 0; i < iterations; ++i) {
        sum += BitOps::hammingDistance(a, b);
    }
    double elapsed = timer.elapsedMs();

    std::cout << "  Hamming: " << std::setw(8) << std::fixed << std::setprecision(2)
              << elapsed << " ms (" << (iterations / elapsed * 1000) << " ops/s)\n";
    std::cout << "  Result: " << (sum / iterations) << " bits differ (avg)\n";
}

// ============================================================================
// Bit Scanning Benchmark
// ============================================================================

template<std::size_t N>
void benchmarkScanning(const char* name, std::size_t iterations) {
    std::cout << "\n[Bit Scanning - " << name << " - " << N << " bits]\n";
    std::cout << std::string(60, '-') << "\n";

    std::mt19937_64 rng(12345);
    BitSet<N> a;
    randomize(a, rng);

    Timer timer;
    volatile std::size_t sum = 0;

    // findFirst
    timer.reset();
    for (std::size_t i = 0; i < iterations; ++i) {
        sum += BitOps::findFirst(a);
    }
    double elapsed = timer.elapsedMs();
    std::cout << "  findFirst: " << std::setw(8) << std::fixed << std::setprecision(2)
              << elapsed << " ms (" << (iterations / elapsed * 1000) << " ops/s)\n";

    // findLast
    timer.reset();
    for (std::size_t i = 0; i < iterations; ++i) {
        sum += BitOps::findLast(a);
    }
    elapsed = timer.elapsedMs();
    std::cout << "  findLast:  " << std::setw(8) << elapsed << " ms ("
              << (iterations / elapsed * 1000) << " ops/s)\n";

#if defined(__BMI1__)
    std::cout << "  Implementation: BMI1 (TZCNT/LZCNT)\n";
#else
    std::cout << "  Implementation: std::countr_zero/countl_zero\n";
#endif
}

// ============================================================================
// Range Operations Benchmark
// ============================================================================

template<std::size_t N>
void benchmarkRangeOps(const char* name, std::size_t iterations) {
    std::cout << "\n[Range Operations - " << name << " - " << N << " bits]\n";
    std::cout << std::string(60, '-') << "\n";

    std::mt19937_64 rng(12345);
    BitSet<N> a;

    Timer timer;

    // setRange
    timer.reset();
    for (std::size_t i = 0; i < iterations; ++i) {
        for (auto& w : a.words) w = 0;
        BitOps::setRange(a, 10, N - 10);
    }
    double elapsed = timer.elapsedMs();
    std::cout << "  setRange:   " << std::setw(8) << std::fixed << std::setprecision(2)
              << elapsed << " ms (" << (iterations / elapsed * 1000) << " ops/s)\n";

    // resetRange
    timer.reset();
    for (std::size_t i = 0; i < iterations; ++i) {
        BitOps::setAll(a);
        BitOps::resetRange(a, 10, N - 10);
    }
    elapsed = timer.elapsedMs();
    std::cout << "  resetRange: " << std::setw(8) << elapsed << " ms ("
              << (iterations / elapsed * 1000) << " ops/s)\n";

    // flipRange
    timer.reset();
    for (std::size_t i = 0; i < iterations; ++i) {
        for (auto& w : a.words) w = 0;
        BitOps::flipRange(a, 10, N - 10);
    }
    elapsed = timer.elapsedMs();
    std::cout << "  flipRange:  " << std::setw(8) << elapsed << " ms ("
              << (iterations / elapsed * 1000) << " ops/s)\n";
}

// ============================================================================
// Advanced Operations Benchmark
// ============================================================================

void benchmarkAdvancedOps(std::size_t iterations) {
    std::cout << "\n[Advanced Operations]\n";
    std::cout << std::string(60, '-') << "\n";

    std::mt19937_64 rng(12345);
    Timer timer;
    volatile uint64_t sum = 0;

    // reverseBits64
    uint64_t val = rng();
    timer.reset();
    for (std::size_t i = 0; i < iterations * 100; ++i) {
        sum += BitOps::reverseBits64(val);
    }
    double elapsed = timer.elapsedMs();
    std::cout << "  reverseBits64: " << std::setw(8) << std::fixed << std::setprecision(2)
              << elapsed << " ms (" << (iterations * 100 / elapsed * 1000) << " ops/s)\n";

    // Gray code conversion
    timer.reset();
    for (std::size_t i = 0; i < iterations * 100; ++i) {
        uint64_t gray = BitOps::binaryToGray(val);
        sum += BitOps::grayToBinary(gray);
    }
    elapsed = timer.elapsedMs();
    std::cout << "  Gray code (roundtrip): " << std::setw(8) << elapsed << " ms ("
              << (iterations * 100 / elapsed * 1000) << " ops/s)\n";

    // Morton code
    uint32_t x = rng() & 0xFFFFFFFF;
    uint32_t y = rng() & 0xFFFFFFFF;
    timer.reset();
    for (std::size_t i = 0; i < iterations * 100; ++i) {
        sum += BitOps::interleaveBits32(x, y);
    }
    elapsed = timer.elapsedMs();
    std::cout << "  Morton interleave: " << std::setw(8) << elapsed << " ms ("
              << (iterations * 100 / elapsed * 1000) << " ops/s)\n";

#if defined(__BMI2__)
    // PEXT/PDEP
    uint64_t mask = rng();
    timer.reset();
    for (std::size_t i = 0; i < iterations * 100; ++i) {
        sum += BitOps::pext64(val, mask);
    }
    elapsed = timer.elapsedMs();
    std::cout << "  PEXT (BMI2): " << std::setw(8) << elapsed << " ms ("
              << (iterations * 100 / elapsed * 1000) << " ops/s)\n";

    timer.reset();
    for (std::size_t i = 0; i < iterations * 100; ++i) {
        sum += BitOps::pdep64(val, mask);
    }
    elapsed = timer.elapsedMs();
    std::cout << "  PDEP (BMI2): " << std::setw(8) << elapsed << " ms ("
              << (iterations * 100 / elapsed * 1000) << " ops/s)\n";
#else
    std::cout << "  PEXT/PDEP: BMI2 not available (using software fallback)\n";
#endif
}

// ============================================================================
// Main Benchmark Suite
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "BitOps Performance Benchmark\n";
    std::cout << "========================================\n";

    // Feature detection
    std::cout << "\nFeature Detection:\n";
    std::cout << std::string(60, '-') << "\n";
#if defined(__AVX2__)
    std::cout << "  AVX2: Enabled\n";
#else
    std::cout << "  AVX2: Not available\n";
#endif

#if defined(__BMI1__)
    std::cout << "  BMI1 (TZCNT/LZCNT): Enabled\n";
#else
    std::cout << "  BMI1: Not available\n";
#endif

#if defined(__BMI2__)
    std::cout << "  BMI2 (PEXT/PDEP): Enabled\n";
#else
    std::cout << "  BMI2: Not available\n";
#endif

    std::cout << "  Compiler: ";
#if defined(__clang__)
    std::cout << "Clang " << __clang_major__ << "." << __clang_minor__ << "\n";
#elif defined(__GNUC__)
    std::cout << "GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "\n";
#elif defined(_MSC_VER)
    std::cout << "MSVC " << _MSC_VER << "\n";
#else
    std::cout << "Unknown\n";
#endif

    // Run benchmarks
    const std::size_t fast_iters = 10000000;  // 10M iterations
    const std::size_t slow_iters = 1000000;   // 1M iterations

    // Small bitsets (cache-friendly)
    benchmarkCoreOps<256>("Core Ops", fast_iters);
    benchmarkPopcount<256>("Small", fast_iters);
    benchmarkHamming<256>("Small", fast_iters);
    benchmarkScanning<256>("Small", fast_iters);
    benchmarkRangeOps<256>("Small", slow_iters);

    // Medium bitsets
    benchmarkCoreOps<4096>("Core Ops", slow_iters);
    benchmarkPopcount<4096>("Medium", slow_iters);
    benchmarkHamming<4096>("Medium", slow_iters);
    benchmarkScanning<4096>("Medium", slow_iters);
    benchmarkRangeOps<4096>("Medium", slow_iters / 10);

    // Large bitsets (tests AVX2 effectiveness)
    benchmarkCoreOps<65536>("Core Ops", slow_iters / 10);
    benchmarkPopcount<65536>("Large", slow_iters / 10);
    benchmarkHamming<65536>("Large", slow_iters / 10);
    benchmarkScanning<65536>("Large", slow_iters / 10);
    benchmarkRangeOps<65536>("Large", slow_iters / 100);

    // Advanced operations
    benchmarkAdvancedOps(slow_iters / 10);

    std::cout << "\n========================================\n";
    std::cout << "Benchmark Complete\n";
    std::cout << "========================================\n";

    return 0;
}
