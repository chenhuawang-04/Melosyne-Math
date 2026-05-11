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
    std::vector<TestCase> test_cases;
};

bool registerTest(const char* suite_, const char* name_, TestFn fn_);
RunSummary runAllTests(const RunOptions& options_ = {});

[[noreturn]] void fail(
    const char* expr,
    const std::string& detail,
    const char* file,
    int line);

void requireTrue(bool condition_, const char* expr_, const char* file_, int line_);
void requireNear(
    double actual_,
    double expected_,
    double epsilon_,
    const char* actual_expr_,
    const char* expected_expr_,
    const char* file_,
    int line_);
void requireEq(
    std::string_view actual_,
    std::string_view expected_,
    const char* actual_expr_,
    const char* expected_expr_,
    const char* file_,
    int line_);
void requireEq(
    long long actual_,
    long long expected_,
    const char* actual_expr_,
    const char* expected_expr_,
    const char* file_,
    int line_);
void requireEq(
    unsigned long long actual_,
    unsigned long long expected_,
    const char* actual_expr_,
    const char* expected_expr_,
    const char* file_,
    int line_);

} // namespace fmtest

#define FM_CONCAT_INNER(a, b) a##b
#define FM_CONCAT(a, b) FM_CONCAT_INNER(a, b)

#define FM_TEST(SUITE, NAME)                                                                  \
    static void FM_CONCAT(fm_test_fn_, __LINE__)();                                           \
    namespace {                                                                                \
    const bool FM_CONCAT(fm_test_reg_, __LINE__) =                                            \
        ::fmtest::registerTest(#SUITE, #NAME, &FM_CONCAT(fm_test_fn_, __LINE__));             \
    }                                                                                          \
    static void FM_CONCAT(fm_test_fn_, __LINE__)()

#define FM_REQUIRE(EXPR)                                                                       \
    ::fmtest::requireTrue((EXPR), #EXPR, __FILE__, __LINE__)

#define FM_REQUIRE_NEAR(ACTUAL, EXPECTED, EPS)                                                 \
    ::fmtest::requireNear(                                                                     \
        static_cast<double>(ACTUAL),                                                           \
        static_cast<double>(EXPECTED),                                                         \
        static_cast<double>(EPS),                                                              \
        #ACTUAL,                                                                               \
        #EXPECTED,                                                                             \
        __FILE__,                                                                              \
        __LINE__)

#define FM_REQUIRE_EQ_STR(ACTUAL, EXPECTED)                                                    \
    ::fmtest::requireEq((ACTUAL), (EXPECTED), #ACTUAL, #EXPECTED, __FILE__, __LINE__)

#define FM_REQUIRE_EQ_I64(ACTUAL, EXPECTED)                                                    \
    ::fmtest::requireEq(                                                                       \
        static_cast<long long>(ACTUAL),                                                        \
        static_cast<long long>(EXPECTED),                                                      \
        #ACTUAL,                                                                               \
        #EXPECTED,                                                                             \
        __FILE__,                                                                              \
        __LINE__)

#define FM_REQUIRE_EQ_U64(ACTUAL, EXPECTED)                                                    \
    ::fmtest::requireEq(                                                                       \
        static_cast<unsigned long long>(ACTUAL),                                               \
        static_cast<unsigned long long>(EXPECTED),                                             \
        #ACTUAL,                                                                               \
        #EXPECTED,                                                                             \
        __FILE__,                                                                              \
        __LINE__)
