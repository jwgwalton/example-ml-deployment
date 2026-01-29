# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the trained model
COPY model/ ./model/

# Set environment variables
ENV MODEL_PATH=/app/model
ENV PORT=8080

# Expose port
EXPOSE 8080

# Run MLFlow model server
CMD mlflow models serve -m ${MODEL_PATH} -h 0.0.0.0 -p ${PORT} --no-conda
