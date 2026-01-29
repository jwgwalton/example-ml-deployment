# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files for better caching
COPY pyproject.toml uv.lock ./

# Sync dependencies using UV (installs from lock file)
RUN uv sync --frozen --no-dev

# Copy the trained model
COPY model/ ./model/

# Set environment variables
ENV MODEL_PATH=/app/model
ENV PORT=8080

# Expose port
EXPOSE 8080

# Run MLFlow model server
CMD mlflow models serve -m ${MODEL_PATH} -h 0.0.0.0 -p ${PORT} --no-conda
