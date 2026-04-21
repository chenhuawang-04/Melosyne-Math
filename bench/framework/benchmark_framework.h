#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace fmbench {

struct BenchConfig {
    int warmup_rounds = 3;
    int measure_rounds = 8;
    int repeats = 7;
    double min_measure_ms = 20.0;
};

struct RunOptions {
    BenchConfig config;
    std::string filter;
};

struct BackendTask {
    std::string backend;
    bool enabled = true;
    std::function<void()> fn;
};

struct SampleStats {
    double median_ns_per_op = 0.0;
    double mean_ns_per_op = 0.0;
    double stdev_ns_per_op = 0.0;
    double ops_per_sec = 0.0;
    int samples = 0;
};

struct BenchmarkCase {
    std::string suite;
    std::string name;
    void (*fn)(const RunOptions& options);
};

class Registry {
public:
    static Registry& instance();
    bool add(std::string suite, std::string name, void (*fn)(const RunOptions&));
    const std::vector<BenchmarkCase>& cases() const noexcept;

private:
    std::vector<BenchmarkCase> cases_;
};

bool register_case(const char* suite, const char* name, void (*fn)(const RunOptions&));

SampleStats measure_backend(
    const BackendTask& task,
    std::size_t ops_per_call,
    const BenchConfig& config);

void run_comparison_case(
    const std::string& title,
    std::size_t ops_per_call,
    const std::vector<BackendTask>& tasks,
    const BenchConfig& config);

int run_all(const RunOptions& options = {});

void consume(double value);
double sink_value();
int invalid_sink_inputs();

} // namespace fmbench

#define FMBENCH_CONCAT_INNER(a, b) a##b
#define FMBENCH_CONCAT(a, b) FMBENCH_CONCAT_INNER(a, b)

#define FM_BENCH(SUITE, NAME)                                                                  \
    static void FMBENCH_CONCAT(fm_bench_fn_, __LINE__)(const ::fmbench::RunOptions&);         \
    namespace {                                                                                 \
    const bool FMBENCH_CONCAT(fm_bench_reg_, __LINE__) =                                       \
        ::fmbench::register_case(#SUITE, #NAME, &FMBENCH_CONCAT(fm_bench_fn_, __LINE__));      \
    }                                                                                           \
    static void FMBENCH_CONCAT(fm_bench_fn_, __LINE__)(const ::fmbench::RunOptions& options)
