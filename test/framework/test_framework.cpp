#include "test/framework/test_framework.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace FmTest {

Registry& Registry::instance() {
    static Registry reg;
    return reg;
}

bool Registry::add(std::string suite_, std::string name_, TestFn fn_) {
    test_cases.push_back(TestCase{std::move(suite_), std::move(name_), fn_});
    return true;
}

const std::vector<TestCase>& Registry::tests() const noexcept {
    return test_cases;
}

bool registerTest(const char* suite_, const char* name_, TestFn fn_) {
    return Registry::instance().add(suite_, name_, fn_);
}

[[noreturn]] void fail(
    const char* expr_,
    const std::string& detail_,
    const char* file_,
    int line_) {
    std::ostringstream oss;
    oss << file_ << ':' << line_ << "\n"
        << "  expression: " << expr_ << "\n"
        << "  detail    : " << detail_;
    throw TestFailure(oss.str());
}

void requireTrue(bool condition_, const char* expr_, const char* file_, int line_) {
    if (!condition_) {
        fail(expr_, "condition evaluated to false", file_, line_);
    }
}

void requireNear(
    double actual_,
    double expected_,
    double epsilon_,
    const char* actual_expr_,
    const char* expected_expr_,
    const char* file_,
    int line_) {
    const double diff = std::abs(actual_ - expected_);
    if (diff > epsilon_) {
        std::ostringstream oss;
        oss << std::setprecision(17)
            << actual_expr_ << " = " << actual_ << ", "
            << expected_expr_ << " = " << expected_ << ", "
            << "|diff| = " << diff << ", epsilon = " << epsilon_;
        fail("near(actual, expected)", oss.str(), file_, line_);
    }
}

void requireEq(
    std::string_view actual_,
    std::string_view expected_,
    const char* actual_expr_,
    const char* expected_expr_,
    const char* file_,
    int line_) {
    if (actual_ != expected_) {
        std::ostringstream oss;
        oss << actual_expr_ << " = '" << actual_ << "', "
            << expected_expr_ << " = '" << expected_ << "'";
        fail("string equality", oss.str(), file_, line_);
    }
}

void requireEq(
    long long actual_,
    long long expected_,
    const char* actual_expr_,
    const char* expected_expr_,
    const char* file_,
    int line_) {
    if (actual_ != expected_) {
        std::ostringstream oss;
        oss << actual_expr_ << " = " << actual_ << ", "
            << expected_expr_ << " = " << expected_;
        fail("integer equality", oss.str(), file_, line_);
    }
}

void requireEq(
    unsigned long long actual_,
    unsigned long long expected_,
    const char* actual_expr_,
    const char* expected_expr_,
    const char* file_,
    int line_) {
    if (actual_ != expected_) {
        std::ostringstream oss;
        oss << actual_expr_ << " = " << actual_ << ", "
            << expected_expr_ << " = " << expected_;
        fail("unsigned integer equality", oss.str(), file_, line_);
    }
}

RunSummary runAllTests(const RunOptions& options_) {
    std::vector<TestCase> tests = Registry::instance().tests();
    std::sort(tests.begin(), tests.end(), [](const TestCase& a_, const TestCase& b_) {
        if (a_.suite == b_.suite) {
            return a_.name < b_.name;
        }
        return a_.suite < b_.suite;
    });

    const auto wall_start = std::chrono::steady_clock::now();

    RunSummary summary;
    std::string current_suite;

    for (const auto& tc : tests) {
        const std::string id = tc.suite + "." + tc.name;
        if (!options_.filter.empty() && id.find(options_.filter) == std::string::npos) {
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
            if (options_.fail_fast) {
                break;
            }
        } catch (const std::exception& e) {
            ++summary.failed;
            std::cout << "  [FAIL] " << tc.name << "\n"
                      << "  unexpected exception: " << e.what() << "\n";
            if (options_.fail_fast) {
                break;
            }
        } catch (...) {
            ++summary.failed;
            std::cout << "  [FAIL] " << tc.name << "\n"
                      << "  unexpected non-standard exception\n";
            if (options_.fail_fast) {
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

} // namespace FmTest
