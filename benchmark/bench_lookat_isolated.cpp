/**
 * @brief Isolated mat4LookAt benchmark
 * Purpose: Test different LookAt implementations without affecting other functions
 */

#include "fast_math/mat4_simd.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

using namespace std;

class Timer {
    std::chrono::high_resolution_clock::time_point start_time;
public:
    void start() { start_time = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_time).count();
    }
};

template<typename Func>
double benchmark_median(Func&& func, int runs = 5) {
    std::vector<double> times;
    times.reserve(runs);

    func();  // Warm-up

    Timer timer;
    for (int r = 0; r < runs; ++r) {
        timer.start();
        func();
        times.push_back(timer.elapsed_ms());
    }

    std::sort(times.begin(), times.end());
    return times[runs / 2];
}

int main() {
    const int COUNT = 10000;
    const int ITERS = 100;
    const int64_t TOTAL_OPS = (int64_t)COUNT * ITERS;

    MMath::Vec4 eye = {0.0f, 0.0f, 5.0f, 1.0f};
    MMath::Vec4 target = {0.0f, 0.0f, 0.0f, 1.0f};
    MMath::Vec4 up = {0.0f, 1.0f, 0.0f, 0.0f};

    std::vector<MMath::Mat4> results(COUNT);
    volatile float sink = 0.0f;

    cout << "\n";
    cout << "================================================================\n";
    cout << "  mat4LookAt Isolated Benchmark\n";
    cout << "================================================================\n";
    cout << "  Operations: " << TOTAL_OPS << " (10K x 100 iterations)\n";
    cout << "================================================================\n\n";

    double time = benchmark_median([&]() {
        for (int iter = 0; iter < ITERS; ++iter)
            for (int i = 0; i < COUNT; ++i)
                results[i] = MMath::mat4LookAt(eye, target, up);
        sink = results[0].m[0];
    });

    cout << "  mat4LookAt: " << fixed << setprecision(2) << time << " ms\n";
    cout << "  Performance: " << setprecision(1) << (TOTAL_OPS / time / 1000.0) << " M ops/s\n";
    cout << "\n";
    cout << "  Target: < 1.25 ms (DirectXMath performance)\n";
    cout << "  Status: " << (time < 1.25 ? "PASS" : "FAIL") << "\n";
    cout << "\n================================================================\n";

    return 0;
}
