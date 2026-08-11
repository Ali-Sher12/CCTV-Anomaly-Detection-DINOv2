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

# Copy dependency requirements list first for efficient layer caching
COPY requirements.txt .

# Install Python packages matching local PC environment versions
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files into the container
COPY . .

# Ensure logs directory exists for output saving
RUN mkdir -p logs

# Default command to run the anomaly detection application
CMD ["python", "main.py"]
