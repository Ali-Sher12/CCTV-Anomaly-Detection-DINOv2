@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo     Installing Python 3.14 ^& Project Dependencies
echo ===================================================
echo.

:: -----------------------------------------------------
:: Make sure we're running as Administrator.
:: Installing Python for all users needs elevation, and
:: doing this check up front avoids a silent failure or
:: a UAC prompt appearing mid-script.
:: -----------------------------------------------------
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Administrator rights are required. Requesting elevation...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

:: -----------------------------------------------------
:: Check if Python is REALLY installed.
:: We deliberately do NOT use "where python", because a
:: fresh Windows install ships a fake python.exe stub on
:: PATH (it just opens the Microsoft Store). "where" would
:: find that stub and report success even with no real
:: Python installed. Running "python --version" forces the
:: stub to fail instead of falsely reporting success.
:: -----------------------------------------------------
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo [!] Python is not installed ^(or the PATH entry is just the Store stub^).
    echo [*] Attempting to install Python via winget...
    winget install --id Python.Python.3.14 --exact --silent --accept-package-agreements --accept-source-agreements >nul 2>&1
    if !errorlevel! neq 0 (
        echo [*] winget install failed or unavailable. Downloading Python installer manually...
        powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.14.7/python-3.14.7-amd64.exe' -OutFile '%TEMP%\python_installer.exe'"
        echo [*] Running Python installer silently...
        "%TEMP%\python_installer.exe" /quiet InstallAllUsers=1 PrependPath=1 Include_pip=1
        del "%TEMP%\python_installer.exe"
    )

    echo [*] Updating session PATH environment variable...
    set "PATH=%LOCALAPPDATA%\Programs\Python\Python314;%LOCALAPPDATA%\Programs\Python\Python314\Scripts;C:\Program Files\Python314;C:\Program Files\Python314\Scripts;%PATH%"

    :: Confirm the install actually worked before going further.
    python --version >nul 2>nul
    if !errorlevel! neq 0 (
        echo [!] ERROR: Python still isn't runnable after install. Please install Python 3.14 manually and re-run this script.
        pause
        exit /b 1
    )
) else (
    echo [+] Python detected:
    python --version
)

echo.
echo [*] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [*] Installing project dependencies...
python -m pip install torch torchvision transformers numpy opencv-python pillow psutil ultralytics pyttsx3 scipy huggingface_hub safetensors tqdm PyYAML certifi filelock fsspec packaging regex idna typing_extensions

if %errorlevel% equ 0 (
    echo.
    echo ===================================================
    echo [+] SUCCESS: All dependencies installed successfully!
    echo ===================================================
) else (
    echo.
    echo [!] ERROR: Installation finished with warnings or errors.
)

pause
