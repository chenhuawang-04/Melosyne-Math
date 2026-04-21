#include "bench/framework/benchmark_framework.h"

#include <algorithm>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    fmbench::RunOptions options;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--quick") {
            options.config.repeats = 3;
            options.config.warmup_rounds = 1;
            options.config.measure_rounds = 3;
            options.config.min_measure_ms = 2.0;
        } else if (arg.rfind("--filter=", 0) == 0) {
            options.filter = arg.substr(std::string("--filter=").size());
        } else if (arg.rfind("--repeats=", 0) == 0) {
            options.config.repeats = std::max(1, std::stoi(arg.substr(10)));
        } else if (arg.rfind("--warmup=", 0) == 0) {
            options.config.warmup_rounds = std::max(0, std::stoi(arg.substr(9)));
        } else if (arg.rfind("--measure=", 0) == 0) {
            options.config.measure_rounds = std::max(1, std::stoi(arg.substr(10)));
        } else if (arg.rfind("--min-ms=", 0) == 0) {
            options.config.min_measure_ms = std::max(0.0, std::stod(arg.substr(9)));
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "fast_math_bench usage:\n"
                      << "  --quick             short run for smoke checks\n"
                      << "  --filter=<token>    only run cases whose suite.name contains token\n"
                      << "  --repeats=<N>       sample count per backend\n"
                      << "  --warmup=<N>        warmup rounds per sample\n"
                      << "  --measure=<N>       minimum measured calls per sample\n"
                      << "  --min-ms=<N>        minimum measured time (ms) per sample\n";
            return 0;
        }
    }

    return fmbench::run_all(options);
}
