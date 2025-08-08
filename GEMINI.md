# Gemini Collaboration Guide: llm-torch

This document outlines the principles, conventions, and roadmap for our collaboration on the `llm-torch` project. My goal is to act as an efficient and helpful AI software engineering assistant.

## Project Overview

`llm-torch` is a modular framework for building and training Large Language Models (LLMs) using PyTorch. The design should be flexible enough to easily incorporate different model architectures (e.g., GPT, Llama, etc.), tokenizers, and training configurations.

## Core Principles & Conventions

To ensure a high-quality and maintainable codebase, I will adhere to the following standards:

### 1. Code Style & Linting
- **Standard:** All Python code will follow the PEP 8 style guide.
- **Tooling:** We will use `ruff` for both linting and automatic formatting. This ensures consistency and catches potential errors early.

### 2. Typing
- **Standard:** All new code must include type hints (`typing` module).
- **Tooling:** We will use `mypy` for static type checking to maintain type safety across the project.

### 3. Testing
- **Framework:** We will use `pytest` for writing and running tests.
- **Requirement:** Every new feature, component, or bug fix must be accompanied by corresponding unit or integration tests. Tests will be located in the `tests/` directory.

### 4. Documentation
- **Docstrings:** All modules, classes, and functions must have clear and concise Google-style docstrings.
- **READMEs:** The main `README.md` will serve as the primary entry point for users and contributors.

### 5. Commit Messages
- **Standard:** We will follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification for clear and automated versioning. Example: `feat(architectures): add initial gpt2 model structure`.

## Technology Stack

- **Language:** Python 3.10+
- **Core Library:** PyTorch
- **Linting & Formatting:** `ruff`
- **Type Checking:** `mypy`
- **Testing:** `pytest`

## Proposed Project Structure

To keep the project organized and scalable, I will use the following structure:

```
llm-torch/
├── configs/               # Training/model configurations (e.g., .yaml files)
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
├── GEMINI.md              # Our collaboration guide
└── README.md
```

## Initial Goal

Our first concrete goal is to:
**Implement a minimal, from-scratch GPT-2-like model architecture.**

This involves:
1.  Setting up the proposed project structure.
2.  Creating the basic model components (e.g., attention block, feed-forward network) in `src/llm_torch/architectures/gpt.py`.
3.  Adding initial tests for the model components.
