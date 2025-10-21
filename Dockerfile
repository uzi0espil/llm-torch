# Nvidia RTX 50X only works with pytorch using 12.8.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

# Set the working directory
WORKDIR /app

# adding folder to python path
ENV PYTHONPATH="/app/llm_torch:${PYTHONPATH}"

# Copy dependency manifests
COPY requirements.txt requirements-dev.txt ./

# Install runtime dependencies
RUN pip install --no-cache-dir -r requirements.txt

ARG BUILD_DEV=false

# Install dev tooling when requested
RUN if [ "${BUILD_DEV}" = "true" ]; then \
        pip install --no-cache-dir -r requirements-dev.txt; \
    fi

# Copy source into the image
COPY llm_torch ./llm_torch
COPY scripts ./scripts
