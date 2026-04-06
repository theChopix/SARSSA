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

Run the application:
```bash
just run
# or: cd src && uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

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
just mlflow           # Start MLflow server
just download-movielens  # Download MovieLens dataset
just download-lastfm     # Download LastFM dataset
```

Run `just` without arguments to see all available commands.

### Running Backend + Frontend (Development)

You need **two terminals** — one for the backend API server and one for the frontend dev server.

#### Terminal 1 — Backend (FastAPI)

```bash
# From the project root
cd src
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or with `just`:
```bash
just run
```

The API will be available at **http://localhost:8000**. You can check the interactive docs at http://localhost:8000/docs.

#### Terminal 2 — Frontend (Vite + React)

```bash
cd frontend
npm install   # first time only
npm run dev
```

The frontend will be available at **http://localhost:5173** and will proxy API requests to the backend via CORS.

#### Optional — MLflow UI

To browse past experiment results:
```bash
just mlflow
# or: cd src && uv run mlflow ui --port 5000
```

MLflow UI will be available at **http://localhost:5000**.

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
