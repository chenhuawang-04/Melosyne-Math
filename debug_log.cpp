// Debug log
#include "include/fast_math/power.h"
#include <iostream>
#include <cmath>
#include <iomanip>

int main() {
    std::cout << std::setprecision(10);

    // Test log boundaries
    std::cout << "log() boundaries test:" << std::endl;

    // log(1) = 0
    float log1_mmath = MMath::log(1.0f);
    float log1_std = std::log(1.0f);
    std::cout << "  log(1): MMath=" << log1_mmath << " std=" << log1_std << std::endl;

    // log(e) ≈ 1
    float loge_mmath = MMath::log(2.718281828459045f);
    float loge_std = std::log(2.718281828459045f);
    std::cout << "  log(e): MMath=" << loge_mmath << " std=" << loge_std
              << " diff=" << std::fabs(loge_mmath - loge_std) << std::endl;

    // log(10) ≈ ln(10)
    float log10_mmath = MMath::log(10.0f);
    float log10_std = std::log(10.0f);
    std::cout << "  log(10): MMath=" << log10_mmath << " std=" << log10_std
              << " diff=" << std::fabs(log10_mmath - log10_std) << std::endl;

    // log(0) = -inf
    float log0_mmath = MMath::log(0.0f);
    float log0_std = std::log(0.0f);
    std::cout << "  log(0): MMath=" << log0_mmath << " std=" << log0_std
              << " isinf(MMath)=" << std::isinf(log0_mmath)
              << " isinf(std)=" << std::isinf(log0_std) << std::endl;

    // log(negative) = NaN
    float logNeg_mmath = MMath::log(-1.0f);
    float logNeg_std = std::log(-1.0f);
    std::cout << "  log(-1): MMath=" << logNeg_mmath << " std=" << logNeg_std
              << " isnan(MMath)=" << std::isnan(logNeg_mmath)
              << " isnan(std)=" << std::isnan(logNeg_std) << std::endl;

    return 0;
}
