@echo off
echo ===================================================
echo     Uninstalling Project Dependencies
echo ===================================================
echo.

:: Same real-check as the installer: "python --version" fails
:: on the Store-stub python.exe, whereas "where python" would not.
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo [!] Python is not installed or not found in PATH.
    pause
    exit /b
)

echo [*] Uninstalling project libraries...
python -m pip uninstall -y torch torchvision transformers numpy opencv-python pillow psutil ultralytics pyttsx3 scipy huggingface_hub safetensors tqdm PyYAML certifi filelock fsspec packaging regex idna typing_extensions pygame-ce

echo.
echo ===================================================
echo [+] Dependencies have been successfully removed.
echo ===================================================
pause
