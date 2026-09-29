@echo off
setlocal

cd /d "%~dp0"

python --version >nul 2>nul
if %errorlevel% neq 0 (
    py --version >nul 2>nul
    if %errorlevel% neq 0 (
        echo [!] Python not found. Please run install.bat first.
        pause
        exit /b 1
    ) else (
        py main.py %*
    )
) else (
    python main.py %*
)
