#pragma once

#include <chrono>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fmtest {

using TestFn = void (*)();

struct TestCase {
    std::string suite;
    std::string name;
    TestFn fn;
};

struct RunOptions {
    bool fail_fast = false;
    std::string filter;
};

struct RunSummary {
    std::size_t total = 0;
    std::size_t passed = 0;
    std::size_t failed = 0;
    double elapsed_seconds = 0.0;
};

class TestFailure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class Registry {
public:
    static Registry& instance();
    bool add(std::string suite, std::string name, TestFn fn);
    const std::vector<TestCase>& tests() const noexcept;

private:
    std::vector<TestCase> tests_;
};

bool register_test(const char* suite, const char* name, TestFn fn);
RunSummary run_all_tests(const RunOptions& options = {});

[[noreturn]] void fail(
    const char* expr,
    const std::string& detail,
    const char* file,
    int line);

void require_true(bool condition, const char* expr, const char* file, int line);
void require_near(
    double actual,
    double expected,
    double epsilon,
    const char* actual_expr,
    const char* expected_expr,
    const char* file,
    int line);
void require_eq(
    std::string_view actual,
    std::string_view expected,
    const char* actual_expr,
    const char* expected_expr,
    const char* file,
    int line);
void require_eq(
    long long actual,
    long long expected,
    const char* actual_expr,
    const char* expected_expr,
    const char* file,
    int line);
void require_eq(
    unsigned long long actual,
    unsigned long long expected,
    const char* actual_expr,
    const char* expected_expr,
    const char* file,
    int line);

} // namespace fmtest

#define FM_CONCAT_INNER(a, b) a##b
#define FM_CONCAT(a, b) FM_CONCAT_INNER(a, b)

#define FM_TEST(SUITE, NAME)                                                                  \
    static void FM_CONCAT(fm_test_fn_, __LINE__)();                                           \
    namespace {                                                                                \
    const bool FM_CONCAT(fm_test_reg_, __LINE__) =                                            \
        ::fmtest::register_test(#SUITE, #NAME, &FM_CONCAT(fm_test_fn_, __LINE__));            \
    }                                                                                          \
    static void FM_CONCAT(fm_test_fn_, __LINE__)()

#define FM_REQUIRE(EXPR)                                                                       \
    ::fmtest::require_true((EXPR), #EXPR, __FILE__, __LINE__)

#define FM_REQUIRE_NEAR(ACTUAL, EXPECTED, EPS)                                                 \
    ::fmtest::require_near(                                                                    \
        static_cast<double>(ACTUAL),                                                           \
        static_cast<double>(EXPECTED),                                                         \
        static_cast<double>(EPS),                                                              \
        #ACTUAL,                                                                               \
        #EXPECTED,                                                                             \
        __FILE__,                                                                              \
        __LINE__)

#define FM_REQUIRE_EQ_STR(ACTUAL, EXPECTED)                                                    \
    ::fmtest::require_eq((ACTUAL), (EXPECTED), #ACTUAL, #EXPECTED, __FILE__, __LINE__)

#define FM_REQUIRE_EQ_I64(ACTUAL, EXPECTED)                                                    \
    ::fmtest::require_eq(                                                                      \
        static_cast<long long>(ACTUAL),                                                        \
        static_cast<long long>(EXPECTED),                                                      \
        #ACTUAL,                                                                               \
        #EXPECTED,                                                                             \
        __FILE__,                                                                              \
        __LINE__)

#define FM_REQUIRE_EQ_U64(ACTUAL, EXPECTED)                                                    \
    ::fmtest::require_eq(                                                                      \
        static_cast<unsigned long long>(ACTUAL),                                               \
        static_cast<unsigned long long>(EXPECTED),                                             \
        #ACTUAL,                                                                               \
        #EXPECTED,                                                                             \
        __FILE__,                                                                              \
        __LINE__)
