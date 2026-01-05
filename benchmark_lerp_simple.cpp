/**
 * @file benchmark_lerp_simple.cpp
 * @brief Simple lerp/smoothstep performance test
 */

#include "include/fast_math/lerp.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>

using namespace MMath;

class Timer {
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double elapsed_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff = end_time - start_time;
        return diff.count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;
};

std::vector<float> generateRandomFloats(size_t count, float min_val, float max_val) {
    std::vector<float> data(count);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(rng);
    }
    return data;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "MMath Lerp Simple Performance Test\n";
    std::cout << "========================================\n\n";

    const int ITERATIONS = 10000000;

    // Test lerp
    {
        auto data_a = generateRandomFloats(ITERATIONS, 0.0f, 100.0f);
        auto data_b = generateRandomFloats(ITERATIONS, 0.0f, 100.0f);
        float t = 0.3f;
        float result = 0.0f;

        Timer timer;
        timer.start();
        for (int i = 0; i < ITERATIONS; ++i) {
            result += lerp(data_a[i], data_b[i], t);
        }
        double time = timer.elapsed_ms();

        std::cout << "lerp (10M iterations): " << std::fixed << std::setprecision(2)
                  << time << " ms\n";
        if (result == 1e30f) std::cout << "";
    }

    // Test smoothstep
    {
        auto data = generateRandomFloats(ITERATIONS, 0.0f, 1.0f);
        float result = 0.0f;

        Timer timer;
        timer.start();
        for (int i = 0; i < ITERATIONS; ++i) {
            result += smoothstep(0.0f, 1.0f, data[i]);
        }
        double time = timer.elapsed_ms();

        std::cout << "smoothstep (10M iterations): " << std::fixed << std::setprecision(2)
                  << time << " ms\n";
        if (result == 1e30f) std::cout << "";
    }

    // Test lerpArray
    {
        const size_t SIZE = 1000000;
        auto data_a = generateRandomFloats(SIZE, 0.0f, 100.0f);
        auto data_b = generateRandomFloats(SIZE, 0.0f, 100.0f);
        std::vector<float> result(SIZE);
        float t = 0.3f;

        Timer timer;
        timer.start();
        lerpArray(data_a.data(), data_b.data(), t, result.data(), static_cast<int32_t>(SIZE));
        double time = timer.elapsed_ms();

        std::cout << "lerpArray (1M elements): " << std::fixed << std::setprecision(3)
                  << time << " ms\n";
    }

    // Test smoothstepArray
    {
        const size_t SIZE = 1000000;
        auto data = generateRandomFloats(SIZE, 0.0f, 1.0f);
        std::vector<float> result(SIZE);

        Timer timer;
        timer.start();
        smoothstepArray(0.0f, 1.0f, data.data(), static_cast<int32_t>(SIZE));
        double time = timer.elapsed_ms();

        std::cout << "smoothstepArray (1M elements): " << std::fixed << std::setprecision(3)
                  << time << " ms\n";
    }

    std::cout << "\n========================================\n";
    std::cout << "Test Complete\n";
    std::cout << "========================================\n";

    return 0;
}
