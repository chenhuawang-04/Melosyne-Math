#include "bench/framework/benchmark_framework.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace fmbench {

namespace {
volatile double g_sink = 0.0;
int g_invalid_sink_inputs = 0;
}

Registry& Registry::instance() {
    static Registry r;
    return r;
}

bool Registry::add(std::string suite, std::string name, void (*fn)(const RunOptions&)) {
    cases_.push_back(BenchmarkCase{std::move(suite), std::move(name), fn});
    return true;
}

const std::vector<BenchmarkCase>& Registry::cases() const noexcept {
    return cases_;
}

bool register_case(const char* suite, const char* name, void (*fn)(const RunOptions&)) {
    return Registry::instance().add(suite, name, fn);
}

void consume(double value) {
    if (!std::isfinite(value)) {
        ++g_invalid_sink_inputs;
        return;
    }
    g_sink += value;
}

double sink_value() {
    return g_sink;
}

int invalid_sink_inputs() {
    return g_invalid_sink_inputs;
}

SampleStats measure_backend(
    const BackendTask& task,
    std::size_t ops_per_call,
    const BenchConfig& config) {
    SampleStats stats{};
    if (!task.enabled || !task.fn || ops_per_call == 0) {
        return stats;
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(std::max(1, config.repeats)));

    for (int repeat = 0; repeat < std::max(1, config.repeats); ++repeat) {
        for (int w = 0; w < std::max(0, config.warmup_rounds); ++w) {
            task.fn();
        }

        const int min_calls = std::max(1, config.measure_rounds);
        int measured_calls = 0;
        double elapsed_ns = 0.0;

        auto run_chunk = [&](int calls) {
            const auto begin = std::chrono::steady_clock::now();
            for (int i = 0; i < calls; ++i) {
                task.fn();
            }
            const auto end = std::chrono::steady_clock::now();
            elapsed_ns += std::chrono::duration<double, std::nano>(end - begin).count();
            measured_calls += calls;
        };

        run_chunk(min_calls);

        const double min_elapsed_ns = std::max(0.0, config.min_measure_ms) * 1e6;
        while (elapsed_ns < min_elapsed_ns) {
            // Exponentially increase workload to amortize timer noise on tiny kernels.
            run_chunk(std::max(1, measured_calls));
        }

        const double ns_per_op =
            elapsed_ns / (static_cast<double>(measured_calls) * static_cast<double>(ops_per_call));
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

void run_comparison_case(
    const std::string& title,
    std::size_t ops_per_call,
    const std::vector<BackendTask>& tasks,
    const BenchConfig& config) {
    std::cout << "\n[CASE] " << title << "\n";
    std::cout << "  ops/call=" << ops_per_call
              << ", repeats=" << config.repeats
              << ", warmup=" << config.warmup_rounds
              << ", measure_calls=" << config.measure_rounds
              << ", min_ms=" << std::fixed << std::setprecision(1) << config.min_measure_ms << "\n";

    struct Row {
        std::string backend;
        SampleStats stats;
        bool enabled;
    };

    std::vector<Row> rows;
    rows.reserve(tasks.size());

    for (const auto& task : tasks) {
        Row row{task.backend, {}, task.enabled};
        if (task.enabled) {
            row.stats = measure_backend(task, ops_per_call, config);
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

int run_all(const RunOptions& options) {
    std::vector<BenchmarkCase> cases = Registry::instance().cases();
    std::sort(cases.begin(), cases.end(), [](const BenchmarkCase& a, const BenchmarkCase& b) {
        if (a.suite == b.suite) return a.name < b.name;
        return a.suite < b.suite;
    });

    std::string current_suite;
    int executed = 0;

    for (const auto& c : cases) {
        const std::string id = c.suite + "." + c.name;
        if (!options.filter.empty() && id.find(options.filter) == std::string::npos) {
            continue;
        }

        if (c.suite != current_suite) {
            current_suite = c.suite;
            std::cout << "\n========================================\n";
            std::cout << "[SUITE] " << current_suite << "\n";
            std::cout << "========================================\n";
        }

        c.fn(options);
        ++executed;
    }

    std::cout << "\n========================================\n";
    std::cout << "Benchmark summary\n";
    std::cout << "  Cases executed: " << executed << "\n";
    std::cout << "  Sink value    : " << sink_value() << "\n";
    std::cout << "  Invalid sink inputs: " << invalid_sink_inputs() << "\n";
    std::cout << "========================================\n";

    return 0;
}

} // namespace fmbench
