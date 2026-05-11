cmake_minimum_required(VERSION 3.20)
project(fast_math VERSION 2.0.0 LANGUAGES CXX)

# =============================================================================
# Global build settings
# =============================================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

option(FAST_MATH_ENABLE_AGGRESSIVE_FLAGS "Enable high-performance compiler flags" ON)
option(FAST_MATH_BUILD_TESTS "Build comprehensive test framework" ON)
option(FAST_MATH_BUILD_BENCHMARKS "Build comprehensive benchmark framework" ON)
option(FAST_MATH_BUILD_MODULES "Build C++20 module interface target" ON)

if(FAST_MATH_BUILD_MODULES AND CMAKE_VERSION VERSION_LESS 3.28)
    message(WARNING "FAST_MATH_BUILD_MODULES requires CMake >= 3.28; disabling modules for this configure.")
    set(FAST_MATH_BUILD_MODULES OFF)
endif()

set(FAST_MATH_DIRECTXMATH_DIR "" CACHE PATH "Optional hint path to DirectXMath root")
set(FAST_MATH_EIGEN_DIR "" CACHE PATH "Optional hint path to Eigen root or include directory")
set(FAST_MATH_GLM_DIR "" CACHE PATH "Optional hint path to GLM root or include directory")

function(fm_find_include_dir out_var header_name root_hint)
    set(_hints)
    if(root_hint)
        list(APPEND _hints "${root_hint}")
    endif()

    find_path(${out_var}
        NAMES ${header_name}
        HINTS ${_hints}
        PATH_SUFFIXES
            ""
            include
            inc
            Inc
            eigen3
        PATHS
            "${CMAKE_CURRENT_SOURCE_DIR}/external"
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
            "${CMAKE_CURRENT_SOURCE_DIR}/vendor"
        DOC "Resolved include directory for ${header_name}"
    )
    set(${out_var} "${${out_var}}" PARENT_SCOPE)
endfunction()

# =============================================================================
# Header-only library target
# =============================================================================

add_library(fast_math INTERFACE)
target_include_directories(fast_math INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(fast_math INTERFACE cxx_std_20)
target_compile_definitions(fast_math INTERFACE NOMINMAX)

function(fm_apply_aggressive_compile_flags target_name)
    if(NOT FAST_MATH_ENABLE_AGGRESSIVE_FLAGS)
        return()
    endif()

    if(MSVC)
        target_compile_options(${target_name} PRIVATE
            /O2
            /Ob3
            /Oi
            /Ot
            /GL
            /arch:AVX2
            /fp:fast
            /GS-
            /Gy
        )
    else()
        target_compile_options(${target_name} PRIVATE
            -O3
            -march=native
            -mtune=native
            -ffast-math
            -fno-math-errno
            -fno-trapping-math
            -funroll-loops
            -finline-functions
            -ftree-vectorize
            -fomit-frame-pointer
        )
    endif()
endfunction()

# =============================================================================
# C++20 modules target
# =============================================================================

if(FAST_MATH_BUILD_MODULES)
    file(GLOB FAST_MATH_MODULE_SOURCES CONFIGURE_DEPENDS
        "modules/fast_math/*.cppm"
    )

    if(FAST_MATH_MODULE_SOURCES)
        add_library(fast_math_modules)
        target_sources(fast_math_modules
            PUBLIC
                FILE_SET CXX_MODULES
                FILES ${FAST_MATH_MODULE_SOURCES}
        )
        target_link_libraries(fast_math_modules PUBLIC fast_math)
        target_include_directories(fast_math_modules PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        )
        target_compile_features(fast_math_modules PUBLIC cxx_std_20)
        target_compile_definitions(fast_math_modules PUBLIC NOMINMAX)

        if(MSVC)
            target_compile_options(fast_math_modules PRIVATE /W4 /EHsc /permissive-)
        else()
            target_compile_options(fast_math_modules PRIVATE -Wall -Wextra -Wpedantic)
        endif()

        message(STATUS "Enabled target: fast_math_modules")
    else()
        message(WARNING "No module interface files found under modules/fast_math/")
    endif()
endif()

# =============================================================================
# Third-party backend detection (for cross-library verification/benchmark)
# =============================================================================

set(FM_HAVE_DIRECTXMATH 0)
set(FM_HAVE_EIGEN 0)
set(FM_HAVE_GLM 0)

fm_find_include_dir(FAST_MATH_DIRECTXMATH_INCLUDE_DIR "DirectXMath.h" "${FAST_MATH_DIRECTXMATH_DIR}")
fm_find_include_dir(FAST_MATH_EIGEN_INCLUDE_DIR "Eigen/Dense" "${FAST_MATH_EIGEN_DIR}")
fm_find_include_dir(FAST_MATH_GLM_INCLUDE_DIR "glm/glm.hpp" "${FAST_MATH_GLM_DIR}")

if(FAST_MATH_DIRECTXMATH_INCLUDE_DIR)
    set(FM_HAVE_DIRECTXMATH 1)
    message(STATUS "DirectXMath detected: ${FAST_MATH_DIRECTXMATH_INCLUDE_DIR}")
else()
    message(STATUS "DirectXMath not found. Set FAST_MATH_DIRECTXMATH_DIR to enable cross-library checks.")
endif()

if(FAST_MATH_EIGEN_INCLUDE_DIR)
    set(FM_HAVE_EIGEN 1)
    message(STATUS "Eigen detected: ${FAST_MATH_EIGEN_INCLUDE_DIR}")
else()
    message(STATUS "Eigen not found. Set FAST_MATH_EIGEN_DIR to enable cross-library checks.")
endif()

if(FAST_MATH_GLM_INCLUDE_DIR)
    set(FM_HAVE_GLM 1)
    message(STATUS "GLM detected: ${FAST_MATH_GLM_INCLUDE_DIR}")
else()
    message(STATUS "GLM not found. Set FAST_MATH_GLM_DIR to enable cross-library checks.")
endif()

function(fm_apply_common_target_setup target_name)
    target_link_libraries(${target_name} PRIVATE fast_math)
    target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    fm_apply_aggressive_compile_flags(${target_name})

    target_compile_definitions(${target_name} PRIVATE
        FM_HAVE_DIRECTXMATH=${FM_HAVE_DIRECTXMATH}
        FM_HAVE_EIGEN=${FM_HAVE_EIGEN}
        FM_HAVE_GLM=${FM_HAVE_GLM}
    )

    if(FM_HAVE_DIRECTXMATH)
        target_include_directories(${target_name} PRIVATE "${FAST_MATH_DIRECTXMATH_INCLUDE_DIR}")
    endif()
    if(FM_HAVE_EIGEN)
        target_include_directories(${target_name} PRIVATE "${FAST_MATH_EIGEN_INCLUDE_DIR}")
    endif()
    if(FM_HAVE_GLM)
        target_include_directories(${target_name} PRIVATE "${FAST_MATH_GLM_INCLUDE_DIR}")
    endif()

    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /EHsc /permissive-)
    else()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

# =============================================================================
# Test framework target
# =============================================================================

if(FAST_MATH_BUILD_TESTS)
    file(GLOB_RECURSE FAST_MATH_TEST_SOURCES CONFIGURE_DEPENDS
        "test/*.cpp"
    )
    list(FILTER FAST_MATH_TEST_SOURCES EXCLUDE REGEX "[/\\\\]module_smoke\\.cpp$")
    list(FILTER FAST_MATH_TEST_SOURCES EXCLUDE REGEX "[/\\\\]module_parity_smoke\\.cpp$")

    if(FAST_MATH_TEST_SOURCES)
        add_executable(fast_math_tests ${FAST_MATH_TEST_SOURCES})
        fm_apply_common_target_setup(fast_math_tests)

        # Tests use stricter floating-point semantics so fast-math does not mask numerical issues.
        if(MSVC)
            target_compile_options(fast_math_tests PRIVATE /fp:precise)
        else()
            target_compile_options(fast_math_tests PRIVATE -fno-fast-math)
        endif()

        enable_testing()
        add_test(NAME fast_math_tests COMMAND fast_math_tests)

        message(STATUS "Enabled target: fast_math_tests")
    else()
        message(WARNING "No test sources found under test/")
    endif()

    if(FAST_MATH_BUILD_MODULES AND TARGET fast_math_modules)
        add_executable(fast_math_modules_smoke
            test/module_smoke.cpp
            test/module_parity_smoke.cpp
        )
        target_link_libraries(fast_math_modules_smoke PRIVATE fast_math_modules)

        if(MSVC)
            target_compile_options(fast_math_modules_smoke PRIVATE /W4 /EHsc /permissive-)
        else()
            target_compile_options(fast_math_modules_smoke PRIVATE -Wall -Wextra -Wpedantic)
        endif()

        add_test(NAME fast_math_modules_smoke COMMAND fast_math_modules_smoke)
        message(STATUS "Enabled target: fast_math_modules_smoke")
    endif()
endif()

# =============================================================================
# Benchmark framework target
# =============================================================================

if(FAST_MATH_BUILD_BENCHMARKS)
    file(GLOB_RECURSE FAST_MATH_BENCH_SOURCES CONFIGURE_DEPENDS
        "bench/*.cpp"
    )

    if(FAST_MATH_BENCH_SOURCES)
        add_executable(fast_math_bench ${FAST_MATH_BENCH_SOURCES})
        fm_apply_common_target_setup(fast_math_bench)

        message(STATUS "Enabled target: fast_math_bench")
    else()
        message(WARNING "No benchmark sources found under bench/")
    endif()
endif()

# =============================================================================
# Installation
# =============================================================================

install(DIRECTORY include/ DESTINATION include)
install(TARGETS fast_math EXPORT fast_mathTargets)
if(TARGET fast_math_modules)
    install(TARGETS fast_math_modules
        EXPORT fast_mathTargets
        FILE_SET CXX_MODULES DESTINATION share/fast_math/modules
    )
endif()
install(EXPORT fast_mathTargets
    FILE fast_mathTargets.cmake
    NAMESPACE fast_math::
    DESTINATION lib/cmake/fast_math
)
