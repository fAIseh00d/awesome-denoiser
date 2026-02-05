# syntax=docker/dockerfile:1
# CUDA base image with Python for GPU-accelerated audio denoising
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set Python environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_CACHE_DIR=/root/.cache/pip

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    git-lfs \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    sox \
    wget \
    && git lfs install \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with cache mount
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install --no-deps denoiser==0.1.5 resemble-enhance==0.0.1

# Copy application code
COPY . .

# Download models
RUN chmod +x download_models.sh && ./download_models.sh example_data

# Expose Gradio default port
EXPOSE 7860

# Set environment variable for Gradio to listen on all interfaces
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Run the Gradio application
CMD ["python", "main.py"]
