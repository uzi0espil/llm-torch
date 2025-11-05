# LLM-Torch: A Framework for Building LLMs from Scratch

## Project Purpose

Welcome to **LLM-Torch**! This repository is a learning-focused project dedicated to building a flexible and modular framework for training Large Language Models (LLMs) from the ground up using PyTorch. The primary goal is to demystify the inner workings of modern LLMs by providing a clean, well-documented, and extensible codebase.

This project is created for educational purposes. While the aim is to build a robust framework, it is not intended to be a production-ready solution. It serves as a hands-on guide for anyone interested in understanding the architecture, training, and implementation details of models like GPT-2.

## Implemented Models

This framework currently includes the following model architectures:

*   **GPT-2**: A from-scratch implementation of the GPT-2 architecture. It uses a standard `TransformerBlock` with `MultiHeadAttention`, `LayerNorm`, and a `GELU` activation function in the feed-forward network.

*   **Llama2**: An implementation of the Llama 2 architecture. It utilizes `RoPEMHA` (Rotary Positional Embedding Multi-Head Attention), `RMSNorm` for normalization, and a `SwiGLUBlock` with `SiLU` activation.

*   **Llama3**: An implementation of the Llama 3 architecture, which features `RoPEGOA` (Rotary Positional Embedding Grouped Query Attention).

*   **Llama3.1**: A Llama 3.1 implementation that uses `YarnGOA` (Yet another RoPE with Grouped Query Attention), which is an extension of the RoPE method for better handling of long contexts.

*   **Llama3.2**: It shrunk in size comparing to Llama3.1, increased the rescaling factor of Yarn, and they added back the weight tying between the input token embedding and the output layer.

*   **GPT-OSS**: It pairs `NTKSWA` (NTK-scaled sliding-window attention) with freely configurable head dimensions and a Mixture-of-Experts feed-forward stack that routes tokens through *bounded* `SwiGLU` experts, following the 20B reference design.

*   **Qwen3**: Similar to Llama3.*, it uses `YarnGOA`, but instead of `SwiGLUBlock` it uses Mixture-of-expert `MoEBlock` for the feedforward block, where a router will choose handful of experts for every token. In addition, it adds `RMSNorm` on Key and Query of GOA. Finally, Although it uses bfloat16 as data type. However, the mean and std of `RMSNorm` is computed with float32.

## Project Structure

The repository is organized into a modular structure to ensure a clear separation of concerns, making it easy to navigate, extend, and experiment with different components.

```
llm-torch/
├── data/                  # Scripts and notebooks for data processing
├── notebooks/             # Jupyter notebooks for experimentation
├── scripts/               # Standalone scripts (e.g., run_training.py)
├── llm_torch/
│   ├── __init__.py
│   ├── architectures/   # Model definitions (gpt, llama, etc.)
│   ├── components/      # Shared building blocks and layers.
│   ├── data/            # Dataloaders and preprocessing pipelines
│   ├── engine/          # Core training and evaluation loops
│   ├── tokenizer/       # Tokenizer implementations or wrappers
│   └── utils/           # Utility functions
├── tests/                 # All tests
├── .gitignore
├── docker-compose.yml     # Docker Compose file for easy setup
├── Dockerfile             # Dockerfile for building the environment
├── pyproject.toml         # Project metadata and dependency declarations
├── requirements.txt       # Convenience entry point for `pip install -e .`
└── README.md              # This file
```

### Key Directories:

*   **`llm_torch/`**: This is the core Python package containing all the source code.
    *   **`architectures/`**: Defines the high-level model architectures (e.g., `GPT2`).
    *   **`components/`**: Contains the fundamental building blocks of the models, such as attention mechanisms, normalization layers, and feed-forward blocks.
    *   **`configs/`**: Manages all configurations using `dataclasses`, allowing for type-safe and flexible setup of models, trainers, and datasets.
    *   **`data/`**: Includes the `Dataset` and `DataLoader` implementations for feeding data to the model.
    *   **`engine/`**: Houses the main `Trainer` and `Predictor` classes, which control the training loop and text generation.
    *   **`utils/`**: A collection of helper functions for tasks like plotting and logging.
*   **`scripts/`**: Contains standalone Python scripts for running key tasks like training (`train.py`) and testing (`test.py`).
*   **`data/`**: Stores raw data files (e.g., `.txt` files for training).
*   **`notebooks/`**: Jupyter notebooks for experimentation, visualization, and analysis.

## Installation & Tooling

Create a virtual environment (or attach to the provided Docker image) and install the runtime dependencies:

```bash
pip install -r requirements.txt
```

For development tooling, including Ruff and pytest, install the additional requirements:

```bash
pip install -r requirements-dev.txt
```

Run linting with `ruff check .` and execute tests with `pytest` once suites are added.

## Docker Usage

For a reproducible and isolated environment, you can use the provided `Dockerfile` and `docker-compose.yml`.

### NVIDIA RTX 50 Series Compatibility

**Please Note:** The current `Dockerfile` is configured to work with **NVIDIA RTX 50 series GPUs**. The base image and CUDA versions have been selected to be compatible with this hardware. If you are using a different GPU series, you may need to adjust the base image in the `Dockerfile` to match your CUDA driver version.

To build and run the container, use Docker Compose. By default it installs runtime dependencies only; pass `BUILD_DEV=true` to bake in linting and test tooling for CI runs:

```bash
# Build the default runtime image
docker-compose build

# Build a dev/testing image with Ruff + pytest installed
docker-compose build --build-arg BUILD_DEV=true

# Start a container and get a shell inside it
docker-compose run --rm dev python scripts/train.py --data-path data/the-verdict.txt --llm gpt2 --size 124M
```
