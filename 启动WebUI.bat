@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo   语音克隆朗读系统 WebUI
echo   GPU: NVIDIA RTX 3050 Laptop 4GB
echo ============================================================

:: 检查便携环境
if not exist "env\python.exe" (
    echo.
    echo [错误] 未找到便携环境 env\，请确保完整解压了项目压缩包。
    echo 如果这是你第一次运行，请解压 env.zip 到当前目录。
    pause
    exit /b 1
)

:: 首次运行：修复 conda-pack 环境路径（运行后会自动删除自身）
if exist "env\Scripts\conda-unpack.exe" (
    echo.
    echo 正在初始化便携环境（首次运行，约10秒）...
    call "env\Scripts\conda-unpack.exe"
    if %errorlevel% neq 0 (
        echo [警告] 环境初始化返回非零，尝试继续...
    )
    echo 初始化完成。
)

echo.
echo 正在启动，请稍候...
echo.

:: 激活便携环境（conda-pack 方式：直接设置 PATH）
set "PATH=%~dp0env;%~dp0env\Scripts;%~dp0env\Library\bin;%~dp0env\Library\mingw-w64\bin;%~dp0env\Library\usr\bin;%PATH%"
set "CONDA_PREFIX=%~dp0env"
set "CONDA_DEFAULT_ENV=voiceclone"

python scripts/webui.py
pause
