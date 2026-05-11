#include "test/framework/test_framework.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    fmtest::RunOptions options;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--fail-fast") {
            options.fail_fast = true;
        } else if (arg.rfind("--filter=", 0) == 0) {
            options.filter = arg.substr(std::string("--filter=").size());
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "fast_math_tests usage:\n"
                      << "  --fail-fast        stop on first failing test\n"
                      << "  --filter=<token>   run tests whose suite.name contains token\n";
            return 0;
        }
    }

    const fmtest::RunSummary summary = fmtest::runAllTests(options);
    return summary.failed == 0 ? 0 : 1;
}
