#!/bin/bash
# Script to run tests locally with Docker
set -e

echo "Building Wine Classifier Docker image..."
docker build -t wine-classifier:test .

echo "Starting Wine Classifier container..."
docker run -d -p 8080:8080 --name wine-classifier-test wine-classifier:test

echo "Waiting for container to be ready..."
sleep 5

echo "Running tests..."
if command -v uv &> /dev/null; then
    echo "Using UV to run tests..."
    API_URL=http://localhost:8080 uv run --with requests python tests/test_api.py
else
    echo "UV not found, using python directly (requires requests to be installed)..."
    API_URL=http://localhost:8080 python tests/test_api.py
fi

# Cleanup
echo "Cleaning up..."
docker stop wine-classifier-test
docker rm wine-classifier-test

echo "Done!"
