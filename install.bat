@echo off
echo =============================================
echo   Passive Anomaly Detector -- Install Script
echo   (CPU-only PyTorch)
echo =============================================

python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo [!] Python not found. Please install Python 3.10+ and re-run.
    pause
    exit /b 1
)

echo.
echo [*] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [*] Installing dependencies (CPU-only PyTorch)...
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

if %errorlevel% equ 0 (
    echo.
    echo =============================================
    echo [+] SUCCESS: All dependencies installed!
    echo     Run the app with:  python main.py
    echo =============================================
) else (
    echo.
    echo [!] Installation finished with errors. See output above.
)
pause
