#!/bin/bash
# Test script for Wine Classifier API

API_URL="${API_URL:-http://localhost:8080}"

echo "Testing Wine Classifier API at $API_URL"
echo "========================================"
echo ""

# Test health endpoint
echo "1. Testing /health endpoint..."
curl -s "$API_URL/health" | python -m json.tool
echo ""

# Test info endpoint
echo "2. Testing /info endpoint..."
curl -s "$API_URL/info" | python -m json.tool
echo ""

# Test prediction endpoint
echo "3. Testing /predict endpoint..."
curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[13.2, 2.77, 2.51, 18.5, 96.6, 1.09, 0.52, 0.2, 0.29, 1.98, 0.13, 1.51, 660]]
  }' | python -m json.tool
echo ""

echo "All tests completed!"
