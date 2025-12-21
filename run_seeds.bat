@echo off
REM =============================================================================
REM Multi-Seed Training Script for Windows (多随机种子训练脚本)
REM =============================================================================
REM 用于运行多个随机种子的实验，生成统计显著的结果
REM
REM Usage:
REM   run_seeds.bat baseline 5           # 运行 baseline 方法，5个种子
REM   run_seeds.bat specnorm 5           # 运行 specnorm 方法，5个种子
REM   run_seeds.bat redo 5               # 运行 redo 方法，5个种子
REM   run_seeds.bat layernorm 5          # 运行 layernorm 方法，5个种子
REM   run_seeds.bat all 5                # 运行所有方法，每个5个种子
REM   run_seeds.bat baseline 5 --resume  # 断点续训模式
REM
REM Output:
REM   results\multiseed\{method}\seed_{n}\checkpoints\{method}_0.pt
REM =============================================================================

setlocal enabledelayedexpansion

set METHOD=%1
set NUM_SEEDS=%2
set RESUME_FLAG=%3
set CONFIG=hyperparams_multiseed.yaml
set OUTPUT_BASE=results\multiseed

if "%METHOD%"=="" set METHOD=baseline
if "%NUM_SEEDS%"=="" set NUM_SEEDS=5

REM 检查是否启用断点续训
set RESUME_ARG=
if "%RESUME_FLAG%"=="--resume" set RESUME_ARG=-r
if "%RESUME_FLAG%"=="-r" set RESUME_ARG=-r

echo ==============================================
echo Multi-Seed Training Script
echo ==============================================
echo Method: %METHOD%
echo Number of seeds: %NUM_SEEDS%
echo Config: %CONFIG%
echo Output: %OUTPUT_BASE%
if defined RESUME_ARG echo Resume: ENABLED
echo ==============================================

if "%METHOD%"=="all" (
    echo Running all methods...
    for %%m in (baseline specnorm redo layernorm) do (
        call :run_method %%m %NUM_SEEDS%
    )
) else (
    call :run_method %METHOD% %NUM_SEEDS%
)

echo.
echo ==============================================
echo All experiments completed!
echo Results saved to: %OUTPUT_BASE%
echo ==============================================
goto :eof

:run_method
set m=%1
set n=%2

echo.
echo ^>^>^> Running %m% with %n% seeds...

for /L %%s in (0,1,%n%) do (
    if %%s lss %n% (
        echo.
        echo --- %m% seed %%s ---
        
        set OUTPUT_DIR=%OUTPUT_BASE%\%m%\seed_%%s
        if not exist "!OUTPUT_DIR!" mkdir "!OUTPUT_DIR!"
        
        python train.py -p %CONFIG% -c %m% -s %%s -n %m%_seed%%s -o "!OUTPUT_DIR!" %RESUME_ARG%
        
        echo √ Completed %m% seed %%s
    )
)

echo.
echo ^>^>^> Completed all seeds for %m%
goto :eof
