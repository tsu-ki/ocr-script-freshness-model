# Use Python 3.11 slim as the base image
FROM python:3.11-slim

# Set environment variables to minimize Python output
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies for OpenCV, PaddleOCR, and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libhdf5-dev \
    libgomp1 \
    git \
    libgl1-mesa-glx && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# Set the working directory
WORKDIR /app

# Copy the entire project directory (use .dockerignore to exclude unnecessary files)
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir paddlepaddle


# Expose the application's port
EXPOSE 5500

# Define the command to run the application
CMD ["python", "app.py"]