#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace FmBench {

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
    void (*fn)(const RunOptions& options_);
};

class Registry {
public:
    static Registry& instance();
    bool add(std::string suite_, std::string name_, void (*fn_)(const RunOptions&));
    const std::vector<BenchmarkCase>& cases() const noexcept;

private:
    std::vector<BenchmarkCase> bench_cases;
};

bool registerCase(const char* suite_, const char* name_, void (*fn_)(const RunOptions& options_));

SampleStats measureBackend(
    const BackendTask& task_,
    std::size_t ops_per_call_,
    const BenchConfig& config_);

void runComparisonCase(
    const std::string& title_,
    std::size_t ops_per_call_,
    const std::vector<BackendTask>& tasks_,
    const BenchConfig& config_);

int runAll(const RunOptions& options_ = {});

void consume(double value_);
double sinkValue();
int invalidSinkInputs();

} // namespace FmBench

#undef FM_BENCH

#define FMBENCH_CONCAT_INNER(a, b) a##b
#define FMBENCH_CONCAT(a, b) FMBENCH_CONCAT_INNER(a, b)

#define FM_BENCH(SUITE, NAME)                                                                  \
    static void FMBENCH_CONCAT(fm_bench_fn_, __LINE__)(const ::FmBench::RunOptions&);         \
    namespace {                                                                                 \
    const bool FMBENCH_CONCAT(fm_bench_reg_, __LINE__) =                                       \
        ::FmBench::registerCase(#SUITE, #NAME, &FMBENCH_CONCAT(fm_bench_fn_, __LINE__));      \
    }                                                                                           \
    static void FMBENCH_CONCAT(fm_bench_fn_, __LINE__)(const ::FmBench::RunOptions& options)
