# CogniBench/Dockerfile
# Version: 0.1 (Phase 4 - API Containerization)

# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Set uv environment variables for non-interactive use
ENV UV_NO_INTERACTIVE 1

# Set the working directory in the container
WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency definition files
COPY pyproject.toml uv.lock* ./
# uv.lock* handles cases where the lock file might not exist yet

# Install dependencies using uv
# --system: Install into the global Python environment (no venv needed in container)
# --no-cache: Avoid caching downloads to keep image size smaller
# --exclude-editable: Install the package normally, not in editable mode
RUN uv pip install --system --no-cache --exclude-editable .

# Copy the rest of the application source code
# Adjust these paths based on what the API actually needs at runtime
COPY core ./core
COPY api ./api
COPY prompts ./prompts
# Consider if data needs to be copied or mounted as a volume
# COPY data ./data

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application
# Use 0.0.0.0 to make it accessible from outside the container
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]