/**
 * Test specific angles to find error source
 */
#include <FastMath/Trig.h>
#include <iostream>
#include <iomanip>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main() {
    std::cout << std::scientific << std::setprecision(10);

    // Test critical angles
    float test_angles[] = {
        0.0f,
        (float)M_PI / 4,
        (float)M_PI / 2,
        (float)M_PI,
        3 * (float)M_PI / 2,
        2 * (float)M_PI,
        -(float)M_PI / 2,
        -(float)M_PI,
        10.0f,
        100.0f
    };

    std::cout << "Angle          | sin error      | cos error\n";
    std::cout << "---------------|----------------|----------------\n";

    for (float angle : test_angles) {
        float fast_s, fast_c;
        FastMath::sincos_scalar(angle, &fast_s, &fast_c);

        double ref_s = std::sin(angle);
        double ref_c = std::cos(angle);

        double sin_err = std::abs(fast_s - ref_s);
        double cos_err = std::abs(fast_c - ref_c);

        std::cout << std::setw(14) << angle << " | "
                  << std::setw(14) << sin_err << " | "
                  << std::setw(14) << cos_err << "\n";
    }

    return 0;
}
