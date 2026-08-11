# Use official Python 3.14 slim image
FROM python:3.14-slim

# Prevent Python from writing .pyc files and enable unbuffered logging output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DISPLAY=:0

# Install Linux system dependencies required for OpenCV, Tkinter (GUI), Pygame sound, and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-tk \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libsdl2-2.0-0 \
    libsdl2-mixer-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Install exact Python packages matching local PC environment versions
RUN pip install --no-cache-dir \
    torch==2.13.0 \
    torchvision==0.28.0 \
    transformers==5.14.1 \
    numpy==2.5.1 \
    opencv-python==5.0.0.93 \
    pillow==12.3.0 \
    pygame-ce==2.5.7 \
    scipy==1.18.0 \
    huggingface_hub==1.25.1 \
    safetensors==0.8.0 \
    tqdm==4.69.1 \
    PyYAML==6.0.3 \
    certifi==2026.6.17 \
    filelock==3.29.0 \
    fsspec==2026.4.0 \
    packaging==26.2 \
    regex==2026.7.19 \
    idna==3.18 \
    typing_extensions==4.16.0

# Copy application files into the container
COPY . .

# Ensure logs directory exists for output saving
RUN mkdir -p logs

# Default command to run the anomaly detection application
CMD ["python", "main.py"]
