@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo   语音克隆朗读系统 WebUI
echo   GPU: NVIDIA RTX 3050 Laptop 4GB
echo ============================================================
echo.
echo 正在启动，请稍候...
echo.

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" voiceclone
python scripts/webui.py
pause
