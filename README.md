# Service Application for Recommender Systems with Sparse Autoencoders

This project is a research-oriented experimental platform for designing, executing, and analyzing recommender system pipelines based on Sparse Autoencoders (SAE). It is developed within the research initiative at Charles University.

## Purpose
The goal of this project is to provide a modular, reproducible, and extensible framework that simplifies experimentation with SAE-enhanced recommender systems, with a focus on interpretability, evaluation, and steering of recommendations.

## Key Features
- Plugin-based pipeline architecture for flexible experiment composition
- Support for multi-step pipelines, including:
  - data loading and preprocessing
  - training collaborative filtering autoencoders
  - training embedded sparse autoencoders
  - neuron labeling and labeling evaluation
  - inspection and steering of recommendations
- Experiment tracking and reproducibility via MLflow
- Web-based UI for pipeline creation, execution, and result inspection
- Reuse of intermediate results from previous experiments

## Context
The platform generalizes prior SAE-based recommender research to enable systematic comparison of methods and efficient collaboration within the research group.

## Development

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) - Fast Python package installer and resolver
- [Node.js](https://nodejs.org/) 18+ and npm - For the frontend
- [just](https://github.com/casey/just) - Command runner (optional but recommended)

### Tooling
This project uses modern Python tooling for development:

- **uv**: Fast package management and virtual environment handling
- **ruff**: Lightning-fast Python linter and formatter (replaces black, isort, flake8)
- **ty**: Fast Python type checker (alternative to mypy)
- **pre-commit**: Git hooks for automated code quality checks
- **justfile**: Task runner for common development commands

### Quick Start

Install dependencies:
```bash
uv sync
```

Set up pre-commit hooks:
```bash
just install-hooks
# or: uv run pre-commit install
```

Run the backend:
```bash
just run
# or: cd src && uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Install and run the frontend:
```bash
just frontend-install
just frontend-dev
# or: cd frontend && npm install && npm run dev
```

The backend runs on `http://localhost:8000` and the frontend on `http://localhost:5173`.

### Common Commands

With `just` installed, you can use these commands:

```bash
just install          # Install dependencies
just install-hooks    # Install pre-commit hooks
just run              # Run the FastAPI application
just format           # Format code with ruff
just lint             # Lint code with ruff
just lint-fix         # Lint and auto-fix issues
just type-check       # Type check with ty
just pre-commit       # Run pre-commit hooks on all files
just check            # Run all checks (lint + type-check)
just fix              # Format and fix all issues
just pre-commit-fix   # Install hooks and run pre-commit
just clean            # Clean up generated files
just frontend-install # Install frontend dependencies
just frontend-dev     # Run frontend dev server (Vite)
just frontend-build   # Build frontend for production
just mlflow           # Start MLflow server
just download-movielens  # Download MovieLens dataset
just download-lastfm     # Download LastFM dataset
```

Run `just` without arguments to see all available commands.

### Manual Commands

If you prefer not to use `just`:

```bash
# Install dependencies
uv sync

# Run application
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Format code
uv run ruff format .

# Lint code
uv run ruff check .
uv run ruff check --fix .  # with auto-fix

# Type check
uv run ty

# Pre-commit hooks
uv run pre-commit install
uv run pre-commit run --all-files
```
