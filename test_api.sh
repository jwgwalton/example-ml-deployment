#!/bin/bash
# Test script for Wine Classifier API (MLFlow Serving)
set -e  # Exit on error

API_URL="${API_URL:-http://localhost:8080}"

echo "Testing Wine Classifier API (MLFlow Serving) at $API_URL"
echo "=========================================================="
echo ""

# Check if python is available
if ! command -v python &> /dev/null; then
    echo "Error: python is not installed or not in PATH"
    exit 1
fi

# Test health endpoint (MLFlow uses /health but returns empty response)
echo "1. Testing /health endpoint..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/health")
if [ "$HTTP_CODE" -eq 200 ]; then
    echo "Health check passed (HTTP $HTTP_CODE)"
else
    echo "Error: Health endpoint returned HTTP $HTTP_CODE"
    exit 1
fi
echo ""

# Test prediction endpoint (MLFlow uses /invocations)
echo "2. Testing /invocations endpoint..."
if ! curl -f -s -X POST "$API_URL/invocations" \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"],
      "data": [[13.2, 2.77, 2.51, 18.5, 96.6, 1.09, 0.52, 0.2, 0.29, 1.98, 0.13, 1.51, 660]]
    }
  }' | python -m json.tool; then
    echo "Error: Invocations endpoint failed"
    exit 1
fi
echo ""

echo "All tests completed successfully!"
