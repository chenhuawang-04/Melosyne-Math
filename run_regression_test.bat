@echo off
REM ============================================================================
REM 位操作库性能回归测试脚本
REM ============================================================================
REM
REM 用途: 自动编译并运行性能回归测试，确保重构后性能无回退
REM
REM 使用方法:
REM   run_regression_test.bat         - 使用默认配置
REM   run_regression_test.bat clean   - 清理并重新编译
REM
REM 输出:
REM   - 编译日志
REM   - 单元测试结果
REM   - 性能回归测试结果
REM   - 基准测试结果
REM ============================================================================

setlocal enabledelayedexpansion

echo ============================================================
echo 位操作库性能回归测试
echo ============================================================
echo.

REM 检查参数
set CLEAN_BUILD=0
if "%1"=="clean" (
    set CLEAN_BUILD=1
    echo [模式] 清理后重新编译
) else (
    echo [模式] 增量编译
)
echo.

REM 设置路径
set BUILD_DIR=build
set BIN_DIR=%BUILD_DIR%\bin\Release

REM 清理旧构建
if %CLEAN_BUILD%==1 (
    echo [1/5] 清理旧构建...
    if exist %BUILD_DIR% (
        rmdir /s /q %BUILD_DIR%
        echo   ✅ 已清理
    ) else (
        echo   ℹ️  无需清理
    )
    echo.
)

REM 创建构建目录
if not exist %BUILD_DIR% (
    echo [1/5] 创建构建目录...
    mkdir %BUILD_DIR%
    echo   ✅ 已创建 %BUILD_DIR%
    echo.
) else (
    echo [1/5] 构建目录已存在
    echo.
)

REM 配置CMake
echo [2/5] 配置CMake...
cd %BUILD_DIR%
cmake .. -G "Visual Studio 17 2022" -A x64
if errorlevel 1 (
    echo   ❌ CMake配置失败
    cd ..
    pause
    exit /b 1
)
echo   ✅ CMake配置成功
cd ..
echo.

REM 编译
echo [3/5] 编译项目 (Release模式, AVX2优化)...
cmake --build %BUILD_DIR% --config Release --target performance_regression_test
if errorlevel 1 (
    echo   ❌ 编译失败
    pause
    exit /b 1
)
echo   ✅ 编译成功
echo.

REM 编译单元测试
echo [3.5/5] 编译单元测试...
cmake --build %BUILD_DIR% --config Release --target test_bit_ops
if errorlevel 1 (
    echo   ⚠️  单元测试编译失败 (继续进行性能测试)
) else (
    echo   ✅ 单元测试编译成功
)
echo.

REM 运行单元测试
if exist %BIN_DIR%\test_bit_ops.exe (
    echo [4/5] 运行单元测试...
    echo ------------------------------------------------------------
    %BIN_DIR%\test_bit_ops.exe
    if errorlevel 1 (
        echo ------------------------------------------------------------
        echo   ❌ 单元测试失败 - 存在功能性问题!
        echo   建议: 修复单元测试后再进行性能测试
        pause
        exit /b 1
    ) else (
        echo ------------------------------------------------------------
        echo   ✅ 所有单元测试通过
        echo.
    )
) else (
    echo [4/5] 跳过单元测试 (未编译)
    echo.
)

REM 运行性能回归测试
echo [5/5] 运行性能回归测试...
echo ============================================================
%BIN_DIR%\performance_regression_test.exe
if errorlevel 1 (
    echo ============================================================
    echo.
    echo   ❌ 性能回归测试失败!
    echo   检测到性能回退，需要优化代码。
    echo.
    echo   详细信息请查看上方测试输出
    echo.
    pause
    exit /b 1
) else (
    echo ============================================================
    echo.
    echo   ✅ 性能回归测试通过!
    echo   重构后的代码性能保持在基线95%%以上
    echo.
)

REM 可选: 运行完整基准测试
echo.
echo ============================================================
echo 可选: 运行完整基准测试
echo ============================================================
echo.
set /p RUN_FULL_BENCH="是否运行完整基准测试? (Y/N): "
if /i "%RUN_FULL_BENCH%"=="Y" (
    echo.
    echo 编译完整基准测试...
    cmake --build %BUILD_DIR% --config Release --target benchmark_bit_ops
    if errorlevel 1 (
        echo   ❌ 基准测试编译失败
    ) else (
        echo   ✅ 基准测试编译成功
        echo.
        echo 运行完整基准测试...
        echo ============================================================
        %BIN_DIR%\benchmark_bit_ops.exe
        echo ============================================================
    )
)

echo.
echo ============================================================
echo 测试完成
echo ============================================================
echo.
echo 结果摘要:
echo   - 编译: ✅ 成功
if exist %BIN_DIR%\test_bit_ops.exe (
    echo   - 单元测试: ✅ 通过
) else (
    echo   - 单元测试: ⏭️  跳过
)
echo   - 性能回归: ✅ 通过
echo.
echo 下一步:
echo   1. 查看上方详细性能数据
echo   2. 提交代码前确认所有测试通过
echo   3. 更新文档（如有必要）
echo.

pause
