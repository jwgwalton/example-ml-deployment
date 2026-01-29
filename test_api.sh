#!/bin/bash
# Test script for Wine Classifier API
set -e  # Exit on error

API_URL="${API_URL:-http://localhost:8080}"

echo "Testing Wine Classifier API at $API_URL"
echo "========================================"
echo ""

# Check if python is available
if ! command -v python &> /dev/null; then
    echo "Error: python is not installed or not in PATH"
    exit 1
fi

# Test health endpoint
echo "1. Testing /health endpoint..."
if ! curl -f -s "$API_URL/health" | python -m json.tool; then
    echo "Error: Health endpoint failed"
    exit 1
fi
echo ""

# Test info endpoint
echo "2. Testing /info endpoint..."
if ! curl -f -s "$API_URL/info" | python -m json.tool; then
    echo "Error: Info endpoint failed"
    exit 1
fi
echo ""

# Test prediction endpoint
echo "3. Testing /predict endpoint..."
if ! curl -f -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[13.2, 2.77, 2.51, 18.5, 96.6, 1.09, 0.52, 0.2, 0.29, 1.98, 0.13, 1.51, 660]]
  }' | python -m json.tool; then
    echo "Error: Predict endpoint failed"
    exit 1
fi
echo ""

echo "All tests completed successfully!"
