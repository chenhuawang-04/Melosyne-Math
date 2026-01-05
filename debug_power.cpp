// Quick debug test to see actual errors
#include "include/fast_math/power.h"
#include <iostream>
#include <cmath>
#include <iomanip>

int main() {
    std::cout << std::setprecision(10);

    // Test exp
    std::cout << "exp() test:" << std::endl;
    float test_vals[] = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f};
    for (float x : test_vals) {
        float mmath = MMath::exp(x);
        float std_val = std::exp(x);
        float rel_error = std::fabs(mmath - std_val) / std::fabs(std_val);
        std::cout << "  exp(" << x << "): MMath=" << mmath
                  << " std=" << std_val
                  << " rel_err=" << rel_error << std::endl;
    }

    std::cout << "\nlog() test:" << std::endl;
    float log_vals[] = {0.1f, 1.0f, 2.0f, 10.0f, 100.0f};
    for (float x : log_vals) {
        float mmath = MMath::log(x);
        float std_val = std::log(x);
        float rel_error = std::fabs(mmath - std_val) / std::fabs(std_val);
        std::cout << "  log(" << x << "): MMath=" << mmath
                  << " std=" << std_val
                  << " rel_err=" << rel_error << std::endl;
    }

    return 0;
}
