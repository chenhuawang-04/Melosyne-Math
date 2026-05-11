#include "bench/framework/benchmark_framework.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace FmBench {

namespace {
volatile double g_sink = 0.0;
int g_invalid_sink_inputs = 0;
}

Registry& Registry::instance() {
    static Registry r;
    return r;
}

bool Registry::add(std::string suite_, std::string name_, void (*fn_)(const RunOptions&)) {
    bench_cases.push_back(BenchmarkCase{std::move(suite_), std::move(name_), fn_});
    return true;
}

const std::vector<BenchmarkCase>& Registry::cases() const noexcept {
    return bench_cases;
}

bool registerCase(const char* suite_, const char* name_, void (*fn_)(const RunOptions& options_)) {
    return Registry::instance().add(suite_, name_, fn_);
}

void consume(double value_) {
    if (!std::isfinite(value_)) {
        ++g_invalid_sink_inputs;
        return;
    }
    g_sink += value_;
}

double sinkValue() {
    return g_sink;
}

int invalidSinkInputs() {
    return g_invalid_sink_inputs;
}

SampleStats measureBackend(
    const BackendTask& task_,
    std::size_t ops_per_call_,
    const BenchConfig& config_) {
    SampleStats stats{};
    if (!task_.enabled || !task_.fn || ops_per_call_ == 0) {
        return stats;
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(std::max(1, config_.repeats)));

    for (int repeat = 0; repeat < std::max(1, config_.repeats); ++repeat) {
        for (int w = 0; w < std::max(0, config_.warmup_rounds); ++w) {
            task_.fn();
        }

        const int min_calls = std::max(1, config_.measure_rounds);
        int measured_calls = 0;
        double elapsed_ns = 0.0;

        auto run_chunk = [&](int calls_) {
            const auto begin = std::chrono::steady_clock::now();
            for (int i = 0; i < calls_; ++i) {
                task_.fn();
            }
            const auto end = std::chrono::steady_clock::now();
            elapsed_ns += std::chrono::duration<double, std::nano>(end - begin).count();
            measured_calls += calls_;
        };

        run_chunk(min_calls);

        const double min_elapsed_ns = std::max(0.0, config_.min_measure_ms) * 1e6;
        while (elapsed_ns < min_elapsed_ns) {
            // Exponentially increase workload to amortize timer noise on tiny kernels.
            run_chunk(std::max(1, measured_calls));
        }

        const double ns_per_op =
            elapsed_ns / (static_cast<double>(measured_calls) * static_cast<double>(ops_per_call_));
        samples.push_back(ns_per_op);
    }

    std::sort(samples.begin(), samples.end());
    const std::size_t mid = samples.size() / 2;
    stats.median_ns_per_op = (samples.size() % 2 == 0)
                                 ? 0.5 * (samples[mid - 1] + samples[mid])
                                 : samples[mid];

    stats.mean_ns_per_op =
        std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());

    double var = 0.0;
    for (double x : samples) {
        const double d = x - stats.mean_ns_per_op;
        var += d * d;
    }
    var /= static_cast<double>(samples.size());
    stats.stdev_ns_per_op = std::sqrt(var);
    stats.ops_per_sec = stats.median_ns_per_op > 0.0 ? 1e9 / stats.median_ns_per_op : 0.0;
    stats.samples = static_cast<int>(samples.size());
    return stats;
}

void runComparisonCase(
    const std::string& title_,
    std::size_t ops_per_call_,
    const std::vector<BackendTask>& tasks_,
    const BenchConfig& config_) {
    std::cout << "\n[CASE] " << title_ << "\n";
    std::cout << "  ops/call=" << ops_per_call_
              << ", repeats=" << config_.repeats
              << ", warmup=" << config_.warmup_rounds
              << ", measure_calls=" << config_.measure_rounds
              << ", min_ms=" << std::fixed << std::setprecision(1) << config_.min_measure_ms << "\n";

    struct Row {
        std::string backend;
        SampleStats stats;
        bool enabled;
    };

    std::vector<Row> rows;
    rows.reserve(tasks_.size());

    for (const auto& task : tasks_) {
        Row row{task.backend, {}, task.enabled};
        if (task.enabled) {
            row.stats = measureBackend(task, ops_per_call_, config_);
        }
        rows.push_back(std::move(row));
    }

    int baseline_index = -1;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].enabled && rows[i].backend == "fast_math") {
            baseline_index = static_cast<int>(i);
            break;
        }
    }
    if (baseline_index < 0) {
        for (std::size_t i = 0; i < rows.size(); ++i) {
            if (rows[i].enabled) {
                baseline_index = static_cast<int>(i);
                break;
            }
        }
    }

    const double baseline_ns =
        (baseline_index >= 0) ? rows[baseline_index].stats.median_ns_per_op : 0.0;

    std::cout << "  " << std::left << std::setw(16) << "backend"
              << std::right << std::setw(14) << "median(ns/op)"
              << std::setw(14) << "stdev"
              << std::setw(10) << "cv(%)"
              << std::setw(14) << "Mops/s"
              << std::setw(12) << "vs fast"
              << "\n";

    for (const auto& row : rows) {
        if (!row.enabled) {
            std::cout << "  " << std::left << std::setw(16) << row.backend
                      << "(disabled)\n";
            continue;
        }

        const double mops = row.stats.ops_per_sec / 1e6;
        const double cv =
            (row.stats.mean_ns_per_op > 0.0)
                ? (row.stats.stdev_ns_per_op / row.stats.mean_ns_per_op) * 100.0
                : 0.0;
        const double speed = (row.stats.median_ns_per_op > 0.0 && baseline_ns > 0.0)
                                 ? baseline_ns / row.stats.median_ns_per_op
                                 : 0.0;

        std::cout << "  " << std::left << std::setw(16) << row.backend << std::right
                  << std::setw(14) << std::fixed << std::setprecision(3)
                  << row.stats.median_ns_per_op << std::setw(14) << row.stats.stdev_ns_per_op
                  << std::setw(10) << cv
                  << std::setw(14) << mops << std::setw(12) << speed << "x\n";
    }
}

int runAll(const RunOptions& options_) {
    std::vector<BenchmarkCase> cases = Registry::instance().cases();
    std::sort(cases.begin(), cases.end(), [](const BenchmarkCase& a_, const BenchmarkCase& b_) {
        if (a_.suite == b_.suite) return a_.name < b_.name;
        return a_.suite < b_.suite;
    });

    std::string current_suite;
    int executed = 0;

    for (const auto& c : cases) {
        const std::string id = c.suite + "." + c.name;
        if (!options_.filter.empty() && id.find(options_.filter) == std::string::npos) {
            continue;
        }

        if (c.suite != current_suite) {
            current_suite = c.suite;
            std::cout << "\n========================================\n";
            std::cout << "[SUITE] " << current_suite << "\n";
            std::cout << "========================================\n";
        }

        c.fn(options_);
        ++executed;
    }

    std::cout << "\n========================================\n";
    std::cout << "Benchmark summary\n";
    std::cout << "  Cases executed: " << executed << "\n";
    std::cout << "  Sink value    : " << sinkValue() << "\n";
    std::cout << "  Invalid sink inputs: " << invalidSinkInputs() << "\n";
    std::cout << "========================================\n";

    return 0;
}

} // namespace FmBench
