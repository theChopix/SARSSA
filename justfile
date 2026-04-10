# Default recipe to display help information
default:
    @just --list

# Install dependencies using uv
install:
    uv sync

# Install dev dependencies
install-dev:
    uv sync --all-extras --dev

# Run the FastAPI application
run:
    cd src && uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Run tests
test *ARGS:
    cd src && uv run pytest {{ARGS}}

# Run unit tests only
test-unit:
    cd src && uv run pytest tests/app/unit

# Run integration tests only
test-integration:
    cd src && uv run pytest tests/app/integration

# Format code with ruff
format:
    uv run ruff format .

# Lint code with ruff
lint:
    uv run ruff check .

# Lint and fix auto-fixable issues
lint-fix:
    uv run ruff check --fix .

# Type check with ty
type-check:
    uv run ty check

# Install pre-commit hooks
install-hooks:
    uv run pre-commit install

# Run pre-commit hooks on all files
pre-commit:
    uv run pre-commit run --all-files

# Run all checks (lint + type-check)
check: lint type-check

# Format and fix all issues
fix: format lint-fix

# Run pre-commit hooks and fix issues
pre-commit-fix: install-hooks pre-commit

# Clean up generated files
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    find . -type d -name "*.egg-info" -exec rm -rf {} +
    find . -type d -name ".pytest_cache" -exec rm -rf {} +
    find . -type d -name ".ruff_cache" -exec rm -rf {} +

# Install frontend dependencies
frontend-install:
    cd frontend && npm install

# Run frontend dev server
frontend-dev:
    cd frontend && npm run dev

# Build frontend for production
frontend-build:
    cd frontend && npm run build

# Start MLflow server
mlflow:
    cd src && uv run mlflow server --host 127.0.0.1 --port 8006

# Download MovieLens dataset
download-movielens:
    bash scripts/download_movieLens_dataset.sh

# Download LastFM dataset
download-lastfm:
    bash scripts/download_lastFm1k_dataset.sh

# Sync dependencies (update lock file)
sync:
    uv sync

# Sync dependencies in frozen mode (exact versions from lock file)
sync-frozen:
    uv sync --frozen

# Add a new dependency
add PACKAGE:
    uv add {{PACKAGE}}

# Add a new dev dependency
add-dev PACKAGE:
    uv add --dev {{PACKAGE}}

# Remove a dependency
remove PACKAGE:
    uv remove {{PACKAGE}}

# Update all dependencies
update:
    uv sync --upgrade
