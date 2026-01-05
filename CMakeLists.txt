cmake_minimum_required(VERSION 3.15)
project(fast_math VERSION 1.0.0 LANGUAGES CXX)

# ============================================================================
# Compiler Configuration
# ============================================================================

set(CMAKE_CXX_STANDARD 20)  # C++20 required for std::popcount, std::countr_zero
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Aggressive optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(FAST_COMPILE_FLAGS
        -O3
        -march=native        # Enable all CPU features
        -mtune=native
        -ffast-math         # Fast math optimizations
        -funroll-loops
        -finline-functions
        -ftree-vectorize    # Auto-vectorization
        -fno-trapping-math
        -fno-math-errno
        -fomit-frame-pointer
    )

    # Link-time optimization
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        list(APPEND FAST_COMPILE_FLAGS -flto)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")
    endif()

elseif(MSVC)
    set(FAST_COMPILE_FLAGS
        /O2                  # Maximum optimization
        /Ob3                 # Aggressive inlining
        /Oi                  # Intrinsic functions
        /Ot                  # Favor fast code
        /GL                  # Whole program optimization
        /arch:AVX2           # Enable AVX2
        /fp:fast             # Fast floating point
        /GS-                 # Disable security checks
        /Gy                  # Function-level linking
    )
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG")
endif()

message(STATUS "Compile Flags: ${FAST_COMPILE_FLAGS}")

# ============================================================================
# Library Target (Header-only)
# ============================================================================

add_library(fast_math INTERFACE)
target_include_directories(fast_math INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(fast_math INTERFACE cxx_std_20)  # C++20 required
target_compile_options(fast_math INTERFACE ${FAST_COMPILE_FLAGS})

# ============================================================================
# Benchmarks
# ============================================================================

# Find perf_math for comparison
set(PERF_MATH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../perf_math" CACHE PATH "Path to perf_math library")

if(EXISTS "${PERF_MATH_DIR}/include/perf_math")
    message(STATUS "Found perf_math at: ${PERF_MATH_DIR}")

    add_executable(bench_fast_vs_perf benchmark/bench_fast_vs_perf.cpp)
    target_link_libraries(bench_fast_vs_perf PRIVATE fast_math)
    target_include_directories(bench_fast_vs_perf PRIVATE
        ${PERF_MATH_DIR}/include
    )
    target_compile_options(bench_fast_vs_perf PRIVATE ${FAST_COMPILE_FLAGS})

    add_executable(bench_vec2_vs_perf benchmark/bench_vec2_vs_perf.cpp)
    target_link_libraries(bench_vec2_vs_perf PRIVATE fast_math)
    target_include_directories(bench_vec2_vs_perf PRIVATE
        ${PERF_MATH_DIR}/include
    )
    target_compile_options(bench_vec2_vs_perf PRIVATE ${FAST_COMPILE_FLAGS})

    add_executable(bench_mat3_vs_perf benchmark/bench_mat3_vs_perf.cpp)
    target_link_libraries(bench_mat3_vs_perf PRIVATE fast_math)
    target_include_directories(bench_mat3_vs_perf PRIVATE
        ${PERF_MATH_DIR}/include
    )
    target_compile_options(bench_mat3_vs_perf PRIVATE ${FAST_COMPILE_FLAGS})

    if(MSVC)
        set_target_properties(bench_fast_vs_perf bench_vec2_vs_perf bench_mat3_vs_perf PROPERTIES LINK_FLAGS "/LTCG")
    endif()

    # Triple-library comparison (fast_math vs GLM vs perf_math)
    set(GLM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../glm-1.0.2" CACHE PATH "Path to GLM library")
    if(EXISTS "${GLM_DIR}/glm")
        message(STATUS "Found GLM at: ${GLM_DIR}")

        add_executable(bench_mat3_triple benchmark/bench_mat3_triple.cpp)
        target_link_libraries(bench_mat3_triple PRIVATE fast_math)
        target_include_directories(bench_mat3_triple PRIVATE
            ${PERF_MATH_DIR}/include
            ${GLM_DIR}
        )
        target_compile_options(bench_mat3_triple PRIVATE ${FAST_COMPILE_FLAGS})

        if(MSVC)
            set_target_properties(bench_mat3_triple PROPERTIES LINK_FLAGS "/LTCG")
        endif()

        # AABB2 triple comparison (fast_math vs GLM vs DirectXMath)
        set(DXM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../DirectXMath-apr2025" CACHE PATH "Path to DirectXMath library")
        if(EXISTS "${DXM_DIR}/Inc")
            message(STATUS "Found DirectXMath at: ${DXM_DIR}")

            add_executable(bench_aabb2_triple benchmark/bench_aabb2_triple.cpp)
            target_link_libraries(bench_aabb2_triple PRIVATE fast_math)
            target_include_directories(bench_aabb2_triple PRIVATE
                ${GLM_DIR}
                ${DXM_DIR}/Inc
            )
            target_compile_options(bench_aabb2_triple PRIVATE ${FAST_COMPILE_FLAGS})

            if(MSVC)
                set_target_properties(bench_aabb2_triple PROPERTIES LINK_FLAGS "/LTCG")
            endif()
        else()
            message(WARNING "DirectXMath not found at ${DXM_DIR}, bench_aabb2_triple disabled")
        endif()
    else()
        message(WARNING "GLM not found at ${GLM_DIR}, bench_mat3_triple disabled")
    endif()
else()
    message(WARNING "perf_math not found at ${PERF_MATH_DIR}, benchmark disabled")
endif()

# Standalone benchmarks (no dependencies)
add_executable(bench_fast benchmark/bench_fast.cpp)
target_link_libraries(bench_fast PRIVATE fast_math)
target_compile_options(bench_fast PRIVATE ${FAST_COMPILE_FLAGS})

add_executable(bench_simd benchmark/bench_simd.cpp)
target_link_libraries(bench_simd PRIVATE fast_math)
target_compile_options(bench_simd PRIVATE ${FAST_COMPILE_FLAGS})

add_executable(bench_vec2 benchmark/bench_vec2.cpp)
target_link_libraries(bench_vec2 PRIVATE fast_math)
target_compile_options(bench_vec2 PRIVATE ${FAST_COMPILE_FLAGS})

add_executable(bench_mat3 benchmark/bench_mat3.cpp)
target_link_libraries(bench_mat3 PRIVATE fast_math)
target_compile_options(bench_mat3 PRIVATE ${FAST_COMPILE_FLAGS})

add_executable(bench_aabb2 benchmark/bench_aabb2.cpp)
target_link_libraries(bench_aabb2 PRIVATE fast_math)
target_compile_options(bench_aabb2 PRIVATE ${FAST_COMPILE_FLAGS})

# Lerp benchmarks and tests
add_executable(test_lerp test_lerp.cpp)
target_link_libraries(test_lerp PRIVATE fast_math)
target_compile_options(test_lerp PRIVATE ${FAST_COMPILE_FLAGS})

# Simple lerp benchmark (standalone, no dependencies)
add_executable(benchmark_lerp_simple benchmark_lerp_simple.cpp)
target_link_libraries(benchmark_lerp_simple PRIVATE fast_math)
target_compile_options(benchmark_lerp_simple PRIVATE ${FAST_COMPILE_FLAGS})

# Common (min/max/clamp) benchmarks and tests
add_executable(test_common test_common.cpp)
target_link_libraries(test_common PRIVATE fast_math)
target_compile_options(test_common PRIVATE ${FAST_COMPILE_FLAGS})

add_executable(benchmark_common benchmark_common.cpp)
target_link_libraries(benchmark_common PRIVATE fast_math)
target_compile_options(benchmark_common PRIVATE ${FAST_COMPILE_FLAGS})

# sqrt tests and benchmarks
add_executable(test_sqrt test_sqrt.cpp)
target_link_libraries(test_sqrt PRIVATE fast_math)
target_compile_options(test_sqrt PRIVATE ${FAST_COMPILE_FLAGS})

# Power (exp/log/pow) tests and benchmarks
add_executable(test_power test_power.cpp)
target_link_libraries(test_power PRIVATE fast_math)
target_compile_options(test_power PRIVATE ${FAST_COMPILE_FLAGS})

add_executable(benchmark_power_full benchmark_power_full.cpp)
target_link_libraries(benchmark_power_full PRIVATE fast_math)
target_compile_options(benchmark_power_full PRIVATE ${FAST_COMPILE_FLAGS})

# Lerp full benchmark (requires GLM and DirectXMath)
set(GLM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../glm-1.0.2" CACHE PATH "Path to GLM library")
set(DXM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../DirectXMath-apr2025" CACHE PATH "Path to DirectXMath library")

# Strict benchmark with GLM only
if(EXISTS "${GLM_DIR}/glm")
    message(STATUS "Building benchmark_lerp_strict with GLM")

    add_executable(benchmark_lerp_strict benchmark_lerp_strict.cpp)
    target_link_libraries(benchmark_lerp_strict PRIVATE fast_math)
    target_include_directories(benchmark_lerp_strict PRIVATE
        ${GLM_DIR}
    )
    target_compile_options(benchmark_lerp_strict PRIVATE ${FAST_COMPILE_FLAGS})

    if(MSVC)
        set_target_properties(benchmark_lerp_strict PROPERTIES LINK_FLAGS "/LTCG")
    endif()
else()
    message(WARNING "GLM not found, benchmark_lerp_strict disabled")
endif()

if(EXISTS "${GLM_DIR}/glm" AND EXISTS "${DXM_DIR}/Inc")
    message(STATUS "Building benchmark_lerp_full with GLM and DirectXMath")

    add_executable(benchmark_lerp_full benchmark_lerp_full.cpp)
    target_link_libraries(benchmark_lerp_full PRIVATE fast_math)
    target_include_directories(benchmark_lerp_full PRIVATE
        ${GLM_DIR}
        ${DXM_DIR}/Inc
    )
    target_compile_options(benchmark_lerp_full PRIVATE ${FAST_COMPILE_FLAGS})

    if(MSVC)
        set_target_properties(benchmark_lerp_full PROPERTIES LINK_FLAGS "/LTCG")
    endif()
else()
    message(WARNING "GLM or DirectXMath not found, benchmark_lerp_full disabled")
endif()

if(MSVC)
    set_target_properties(bench_fast bench_simd bench_vec2 bench_mat3 bench_aabb2 test_lerp test_common test_sqrt test_power benchmark_common benchmark_power_full PROPERTIES LINK_FLAGS "/LTCG")
endif()

# ============================================================================
# BitOps Tests and Benchmarks
# ============================================================================

# Bit operations unit tests
add_executable(test_bit_ops test_bit_ops.cpp)
target_link_libraries(test_bit_ops PRIVATE fast_math)
target_compile_options(test_bit_ops PRIVATE ${FAST_COMPILE_FLAGS})

# Bit operations benchmark
add_executable(benchmark_bit_ops benchmark_bit_ops.cpp)
target_link_libraries(benchmark_bit_ops PRIVATE fast_math)
target_compile_options(benchmark_bit_ops PRIVATE ${FAST_COMPILE_FLAGS})

# Performance regression test (新增 - 确保重构后性能无回退)
add_executable(performance_regression_test performance_regression_test.cpp)
target_link_libraries(performance_regression_test PRIVATE fast_math)
target_compile_options(performance_regression_test PRIVATE ${FAST_COMPILE_FLAGS})

# SIMD bottleneck diagnosis tool
add_executable(diagnose_simd_bottleneck diagnose_simd_bottleneck.cpp)
target_link_libraries(diagnose_simd_bottleneck PRIVATE fast_math)
target_compile_options(diagnose_simd_bottleneck PRIVATE ${FAST_COMPILE_FLAGS})

# Guaranteed vectorization test
add_executable(test_guaranteed_vectorization test_guaranteed_vectorization.cpp)
target_link_libraries(test_guaranteed_vectorization PRIVATE fast_math)
target_compile_options(test_guaranteed_vectorization PRIVATE ${FAST_COMPILE_FLAGS})

# BitSetView overhead test
add_executable(test_bitset_view_overhead test_bitset_view_overhead.cpp)
target_link_libraries(test_bitset_view_overhead PRIVATE fast_math)
target_compile_options(test_bitset_view_overhead PRIVATE ${FAST_COMPILE_FLAGS})

if(MSVC)
    set_target_properties(test_bit_ops benchmark_bit_ops performance_regression_test diagnose_simd_bottleneck test_guaranteed_vectorization test_bitset_view_overhead PROPERTIES LINK_FLAGS "/LTCG")
endif()

message(STATUS "BitOps tests configured:")
message(STATUS "  - test_bit_ops: Unit tests for all bit operations")
message(STATUS "  - benchmark_bit_ops: Performance benchmarks")
message(STATUS "  - performance_regression_test: Regression test (ensures no performance degradation)")
message(STATUS "  - test_bitset_view_overhead: Zero-cost abstraction verification")

# ============================================================================
# Installation
# ============================================================================

install(DIRECTORY include/ DESTINATION include)
install(TARGETS fast_math EXPORT fast_mathTargets)
install(EXPORT fast_mathTargets
    FILE fast_mathTargets.cmake
    NAMESPACE fast_math::
    DESTINATION lib/cmake/fast_math
)
