# ARG statement to choose base image
ARG USE_GPU=false

# Use the appropriate base image based on the USE_GPU build arg
# FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0rc0-gpu-cuda12.3-cudnn9.0-trt8.6 AS gpu-base
# FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0rc0 AS cpu-base

FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0rc0

# Select the final base image
# FROM gpu-base AS gpu
# FROM cpu-base AS cpu

# Continue with the final base image based on USE_GPU
# FROM ${USE_GPU:+gpu}${USE_GPU:-cpu}

# Set working directory
WORKDIR /app

# Install system dependencies
# RUN apt-get update && apt-get install -y \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender-dev \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PaddleX
RUN pip install --no-cache-dir https://paddle-model-ecology.bj.bcebos.com/paddlex/whl/paddlex-3.0.0rc0-py3-none-any.whl

# Copy the application code
COPY app.py .

# Create directories for uploads and outputs
RUN mkdir -p /app/uploads 

# Expose the port the app runs on
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
# This will be overridden by docker-compose or docker run command
ENV USE_GPU=${USE_GPU:-false}

# Command to run the application
CMD ["python", "app.py"]