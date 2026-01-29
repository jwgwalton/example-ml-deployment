"""
Test script for Wine Classifier API (MLFlow Serving)

This script tests the MLFlow serving API endpoints to ensure:
1. Health check endpoint is responsive
2. Prediction endpoint accepts requests and returns valid predictions
"""
import sys
import os
import json
import time
import requests
from typing import Dict, Any, List


def test_health_endpoint(api_url: str) -> bool:
    """
    Test the health endpoint.
    
    Args:
        api_url: Base URL of the API
        
    Returns:
        True if health check passes, False otherwise
    """
    endpoint = f"{api_url}/health"
    print(f"Testing /health endpoint at {endpoint}...")
    
    try:
        response = requests.get(endpoint, timeout=10)
        if response.status_code == 200:
            print(f"✓ Health check passed (HTTP {response.status_code})")
            return True
        else:
            print(f"✗ Health endpoint returned HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_prediction_endpoint(api_url: str) -> bool:
    """
    Test the prediction endpoint with sample wine data.
    
    Args:
        api_url: Base URL of the API
        
    Returns:
        True if prediction succeeds, False otherwise
    """
    endpoint = f"{api_url}/invocations"
    print(f"Testing /invocations endpoint at {endpoint}...")
    
    # Sample wine data (13 features)
    payload = {
        "dataframe_split": {
            "columns": [
                "alcohol", "malic_acid", "ash", "alcalinity_of_ash",
                "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols",
                "proanthocyanins", "color_intensity", "hue",
                "od280/od315_of_diluted_wines", "proline"
            ],
            "data": [
                [13.2, 2.77, 2.51, 18.5, 96.6, 1.09, 0.52, 0.2, 0.29, 1.98, 0.13, 1.51, 660]
            ]
        }
    }
    
    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Prediction successful: {json.dumps(result, indent=2)}")
            
            # Validate response structure
            if "predictions" in result:
                predictions = result["predictions"]
                if isinstance(predictions, list) and len(predictions) > 0:
                    print(f"✓ Valid prediction response with {len(predictions)} prediction(s)")
                    return True
                else:
                    print("✗ Invalid predictions format")
                    return False
            else:
                print("✗ Response missing 'predictions' key")
                return False
        else:
            print(f"✗ Prediction failed with HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Prediction request failed: {e}")
        return False


def test_multiple_predictions(api_url: str) -> bool:
    """
    Test the prediction endpoint with multiple wine samples.
    
    Args:
        api_url: Base URL of the API
        
    Returns:
        True if all predictions succeed, False otherwise
    """
    endpoint = f"{api_url}/invocations"
    print(f"Testing /invocations endpoint with multiple samples...")
    
    # Multiple wine samples
    payload = {
        "dataframe_split": {
            "columns": [
                "alcohol", "malic_acid", "ash", "alcalinity_of_ash",
                "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols",
                "proanthocyanins", "color_intensity", "hue",
                "od280/od315_of_diluted_wines", "proline"
            ],
            "data": [
                [13.2, 2.77, 2.51, 18.5, 96.6, 1.09, 0.52, 0.2, 0.29, 1.98, 0.13, 1.51, 660],
                [12.37, 1.07, 2.1, 18.5, 88.0, 3.52, 3.75, 0.24, 1.95, 4.5, 1.04, 2.77, 660],
                [13.73, 4.36, 2.26, 22.5, 88.0, 1.28, 0.47, 0.52, 1.15, 6.62, 0.78, 1.75, 520]
            ]
        }
    }
    
    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            predictions = result.get("predictions", [])
            
            if len(predictions) == 3:
                print(f"✓ Received {len(predictions)} predictions: {predictions}")
                return True
            else:
                print(f"✗ Expected 3 predictions, got {len(predictions)}")
                return False
        else:
            print(f"✗ Multiple predictions failed with HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Multiple predictions request failed: {e}")
        return False


def wait_for_service(api_url: str, max_attempts: int = 30, delay: int = 2) -> bool:
    """
    Wait for the service to become available.
    
    Args:
        api_url: Base URL of the API
        max_attempts: Maximum number of retry attempts
        delay: Delay in seconds between attempts
        
    Returns:
        True if service becomes available, False otherwise
    """
    print(f"Waiting for service at {api_url} to become available...")
    
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✓ Service is ready (attempt {attempt}/{max_attempts})")
                return True
        except requests.exceptions.RequestException:
            pass
        
        if attempt < max_attempts:
            print(f"  Attempt {attempt}/{max_attempts}: Service not ready, waiting {delay}s...")
            time.sleep(delay)
    
    print(f"✗ Service did not become available after {max_attempts} attempts")
    return False


def main():
    """Main test execution function."""
    # Get API URL from environment variable or use default
    api_url = os.getenv("API_URL", "http://localhost:8080")
    
    print("=" * 70)
    print(f"Wine Classifier API Test Suite")
    print(f"Testing API at: {api_url}")
    print("=" * 70)
    print()
    
    # Wait for service to be ready
    if not wait_for_service(api_url):
        print("\n✗ Service failed to start. Aborting tests.")
        sys.exit(1)
    
    print()
    
    # Run tests
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Health endpoint
    print("Test 1: Health Endpoint")
    print("-" * 70)
    if test_health_endpoint(api_url):
        tests_passed += 1
    else:
        tests_failed += 1
    print()
    
    # Test 2: Single prediction
    print("Test 2: Single Prediction")
    print("-" * 70)
    if test_prediction_endpoint(api_url):
        tests_passed += 1
    else:
        tests_failed += 1
    print()
    
    # Test 3: Multiple predictions
    print("Test 3: Multiple Predictions")
    print("-" * 70)
    if test_multiple_predictions(api_url):
        tests_passed += 1
    else:
        tests_failed += 1
    print()
    
    # Print summary
    print("=" * 70)
    print(f"Test Results: {tests_passed} passed, {tests_failed} failed")
    print("=" * 70)
    
    if tests_failed > 0:
        print("\n✗ Some tests failed")
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
