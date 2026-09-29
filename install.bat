@echo off
setlocal enabledelayedexpansion

echo =============================================
echo   Passive Anomaly Detector -- Install Script
echo   Cross-Platform Windows Setup (CPU PyTorch)
echo =============================================
echo.

REM 1. Verify Python availability and version
python --version >nul 2>nul
if %errorlevel% neq 0 (
    py --version >nul 2>nul
    if %errorlevel% neq 0 (
        echo [!] Python not found in PATH.
        echo     Please install Python 3.10+ from https://www.python.org/
        echo     Be sure to check "Add Python to PATH" during installation.
        pause
        exit /b 1
    ) else (
        set "PY_CMD=py"
    )
) else (
    set "PY_CMD=python"
)

echo [+] Using Python interpreter: !PY_CMD!
!PY_CMD! --version

REM Check for minimum Python version (>= 3.10)
!PY_CMD! -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if %errorlevel% neq 0 (
    echo [!] ERROR: Python 3.10 or higher is required.
    pause
    exit /b 1
)

echo.
echo [*] Upgrading pip, setuptools, and wheel...
!PY_CMD! -m pip install --upgrade pip setuptools wheel

echo.
echo [*] Installing dependencies from requirements.txt...
!PY_CMD! -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

if %errorlevel% neq 0 (
    echo.
    echo [!] Installation encountered errors. See above.
    pause
    exit /b 1
)

echo.
echo [*] Verifying core packages...
!PY_CMD! -c "import torch, cv2, PIL, pyttsx3, psutil; print('[+] All core libraries loaded successfully.')" >nul 2>nul
if %errorlevel% neq 0 (
    echo [!] Warning: Some core libraries could not be imported. Please inspect the pip logs.
) else (
    echo [+] All core libraries verified!
)

echo.
echo =============================================
echo [+] SUCCESS: Installation complete!
echo.
echo     To launch the Anomaly Detector:
echo       run.bat  (or: python main.py)
echo.
echo     To launch the Background Watchdog:
echo       run_watchdog.bat  (or: python watchdog.py)
echo =============================================
echo.
pause
