#include "test/framework/test_framework.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace fmtest {

Registry& Registry::instance() {
    static Registry reg;
    return reg;
}

bool Registry::add(std::string suite, std::string name, TestFn fn) {
    tests_.push_back(TestCase{std::move(suite), std::move(name), fn});
    return true;
}

const std::vector<TestCase>& Registry::tests() const noexcept {
    return tests_;
}

bool register_test(const char* suite, const char* name, TestFn fn) {
    return Registry::instance().add(suite, name, fn);
}

[[noreturn]] void fail(
    const char* expr,
    const std::string& detail,
    const char* file,
    int line) {
    std::ostringstream oss;
    oss << file << ':' << line << "\n"
        << "  expression: " << expr << "\n"
        << "  detail    : " << detail;
    throw TestFailure(oss.str());
}

void require_true(bool condition, const char* expr, const char* file, int line) {
    if (!condition) {
        fail(expr, "condition evaluated to false", file, line);
    }
}

void require_near(
    double actual,
    double expected,
    double epsilon,
    const char* actual_expr,
    const char* expected_expr,
    const char* file,
    int line) {
    const double diff = std::abs(actual - expected);
    if (diff > epsilon) {
        std::ostringstream oss;
        oss << std::setprecision(17)
            << actual_expr << " = " << actual << ", "
            << expected_expr << " = " << expected << ", "
            << "|diff| = " << diff << ", epsilon = " << epsilon;
        fail("near(actual, expected)", oss.str(), file, line);
    }
}

void require_eq(
    std::string_view actual,
    std::string_view expected,
    const char* actual_expr,
    const char* expected_expr,
    const char* file,
    int line) {
    if (actual != expected) {
        std::ostringstream oss;
        oss << actual_expr << " = '" << actual << "', "
            << expected_expr << " = '" << expected << "'";
        fail("string equality", oss.str(), file, line);
    }
}

void require_eq(
    long long actual,
    long long expected,
    const char* actual_expr,
    const char* expected_expr,
    const char* file,
    int line) {
    if (actual != expected) {
        std::ostringstream oss;
        oss << actual_expr << " = " << actual << ", "
            << expected_expr << " = " << expected;
        fail("integer equality", oss.str(), file, line);
    }
}

void require_eq(
    unsigned long long actual,
    unsigned long long expected,
    const char* actual_expr,
    const char* expected_expr,
    const char* file,
    int line) {
    if (actual != expected) {
        std::ostringstream oss;
        oss << actual_expr << " = " << actual << ", "
            << expected_expr << " = " << expected;
        fail("unsigned integer equality", oss.str(), file, line);
    }
}

RunSummary run_all_tests(const RunOptions& options) {
    std::vector<TestCase> tests = Registry::instance().tests();
    std::sort(tests.begin(), tests.end(), [](const TestCase& a, const TestCase& b) {
        if (a.suite == b.suite) {
            return a.name < b.name;
        }
        return a.suite < b.suite;
    });

    const auto wall_start = std::chrono::steady_clock::now();

    RunSummary summary;
    std::string current_suite;

    for (const auto& tc : tests) {
        const std::string id = tc.suite + "." + tc.name;
        if (!options.filter.empty() && id.find(options.filter) == std::string::npos) {
            continue;
        }

        if (tc.suite != current_suite) {
            current_suite = tc.suite;
            std::cout << "\n[SUITE] " << current_suite << '\n';
        }

        ++summary.total;
        const auto start = std::chrono::steady_clock::now();

        try {
            tc.fn();
            const auto end = std::chrono::steady_clock::now();
            const auto ms = std::chrono::duration<double, std::milli>(end - start).count();
            ++summary.passed;
            std::cout << "  [PASS] " << tc.name << " (" << std::fixed << std::setprecision(3) << ms
                      << " ms)\n";
        } catch (const TestFailure& e) {
            ++summary.failed;
            std::cout << "  [FAIL] " << tc.name << "\n" << e.what() << "\n";
            if (options.fail_fast) {
                break;
            }
        } catch (const std::exception& e) {
            ++summary.failed;
            std::cout << "  [FAIL] " << tc.name << "\n"
                      << "  unexpected exception: " << e.what() << "\n";
            if (options.fail_fast) {
                break;
            }
        } catch (...) {
            ++summary.failed;
            std::cout << "  [FAIL] " << tc.name << "\n"
                      << "  unexpected non-standard exception\n";
            if (options.fail_fast) {
                break;
            }
        }
    }

    const auto wall_end = std::chrono::steady_clock::now();
    summary.elapsed_seconds = std::chrono::duration<double>(wall_end - wall_start).count();

    std::cout << "\n========================================\n";
    std::cout << "Test summary\n";
    std::cout << "  Total : " << summary.total << '\n';
    std::cout << "  Passed: " << summary.passed << '\n';
    std::cout << "  Failed: " << summary.failed << '\n';
    std::cout << "  Time  : " << std::fixed << std::setprecision(3) << summary.elapsed_seconds << " s\n";
    std::cout << "========================================\n";

    return summary;
}

} // namespace fmtest
