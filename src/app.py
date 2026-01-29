"""
Flask API for serving ML model predictions.
"""
from flask import Flask, request, jsonify
import mlflow.sklearn
import numpy as np
import os

app = Flask(__name__)

# Load the model
MODEL_PATH = os.getenv("MODEL_PATH", "model")
model = None

def load_model():
    """Load the MLFlow model."""
    global model
    try:
        model = mlflow.sklearn.load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


# Load model on startup
load_model()


@app.route("/", methods=["GET"])
def home():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "message": "Wine Classification API"
    })


@app.route("/health", methods=["GET"])
def health():
    """Detailed health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict wine class from input features.
    
    Expected JSON format:
    {
        "features": [[5.9, 1.04, 0.05, 1.8, 0.084, 5.0, 13.0, 0.9927, 3.22, 0.55, 9.9]]
    }
    
    Returns:
    {
        "predictions": [0],
        "probabilities": [[0.95, 0.03, 0.02]]
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Get input data
        data = request.get_json()
        
        if "features" not in data:
            return jsonify({"error": "Missing 'features' in request body"}), 400
        
        features = np.array(data["features"])
        
        # Validate input shape
        if len(features.shape) != 2:
            return jsonify({"error": "Features must be a 2D array"}), 400
        
        # Make prediction
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        
        return jsonify({
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist()
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/info", methods=["GET"])
def info():
    """Get model information."""
    wine_classes = ["class_0", "class_1", "class_2"]
    feature_names = [
        "alcohol", "malic_acid", "ash", "alcalinity_of_ash",
        "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols",
        "proanthocyanins", "color_intensity", "hue", 
        "od280/od315_of_diluted_wines", "proline"
    ]
    
    return jsonify({
        "model_type": "Wine Classification",
        "classes": wine_classes,
        "n_features": 13,
        "feature_names": feature_names,
        "description": "Classifies wine into three classes based on chemical properties"
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
