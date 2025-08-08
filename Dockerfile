# Nvidia RTX 50X only works with pytorch using 12.8.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

# Set the working directory
WORKDIR /app

# Copy and install your project's requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt