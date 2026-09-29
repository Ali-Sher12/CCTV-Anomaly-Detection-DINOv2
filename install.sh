#!/usr/bin/env bash
# install.sh — Cross-platform install script (Linux / macOS)
# Installs CPU-only PyTorch so no CUDA toolkit is required.
set -e

echo "============================================="
echo "  Passive Anomaly Detector — Install Script"
echo "  (CPU-only PyTorch)"
echo "============================================="

# ---- Check for Tkinter (system library, not installable via pip) ----
PYTHON=$(command -v python3 || command -v python || true)
if [ -z "$PYTHON" ]; then
    echo "[!] Python not found. Please install Python 3.10+ and re-run."
    exit 1
fi

if ! "$PYTHON" -c "import tkinter" 2>/dev/null; then
    echo ""
    echo "[!] ERROR: Tkinter is not available."
    echo "    This is a system package — pip cannot install it."
    echo "    Install it with your package manager, then re-run this script:"
    echo ""
    if command -v pacman &>/dev/null; then
        echo "      sudo pacman -S tk"
    elif command -v apt-get &>/dev/null; then
        echo "      sudo apt-get install python3-tk"
    elif command -v dnf &>/dev/null; then
        echo "      sudo dnf install python3-tkinter"
    elif command -v brew &>/dev/null; then
        echo "      brew install python-tk"
    else
        echo "      Install the 'tk' or 'python3-tk' package for your distro."
    fi
    echo ""
    exit 1
fi
echo "[+] Tkinter OK."

# ---- Check for eSpeak / TTS (needed by pyttsx3 on Linux) ----
if [[ "$OSTYPE" == "linux"* ]]; then
    if ! command -v espeak-ng &>/dev/null && ! command -v espeak &>/dev/null; then
        echo ""
        echo "[!] Warning: 'espeak-ng' not detected."
        echo "    Audio alerts on Linux require espeak-ng. Install with:"
        if command -v pacman &>/dev/null; then
            echo "      sudo pacman -S espeak-ng"
        elif command -v apt-get &>/dev/null; then
            echo "      sudo apt-get install espeak-ng"
        elif command -v dnf &>/dev/null; then
            echo "      sudo dnf install espeak-ng"
        fi
        echo ""
    else
        echo "[+] eSpeak TTS OK."
    fi
fi

# ---- Check Python >= 3.10 ----
VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
MAJOR=$(echo "$VERSION" | cut -d. -f1)
MINOR=$(echo "$VERSION" | cut -d. -f2)
if [ "$MAJOR" -lt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]; }; then
    echo "[!] Python 3.10+ required. Found $VERSION."
    exit 1
fi
echo "[+] Python $VERSION detected."

echo ""
echo "[*] Upgrading pip..."
"$PYTHON" -m pip install --upgrade pip

echo ""
echo "[*] Installing dependencies (CPU-only PyTorch)..."
"$PYTHON" -m pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

echo ""
echo "============================================="
echo "[+] SUCCESS: All dependencies installed!"
echo "    Run the app with:  python3 main.py"
echo "============================================="

