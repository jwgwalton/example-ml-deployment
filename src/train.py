"""
Training script for ML model using scikit-learn Pipeline API and MLFlow.
"""
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os

from transformer import CustomFeatureTransformer


def train_model():
    """
    Train a machine learning model using sklearn Pipeline API with custom transformer.
    Logs the model and metrics to MLFlow.
    """
    # Set MLFlow tracking URI and experiment name
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("wine-classification")
    
    # Load the Wine dataset (a toy dataset from sklearn)
    print("Loading Wine dataset...")
    wine = load_wine()
    X, y = wine.data, wine.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Feature names: {wine.feature_names}")
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Start MLFlow run
    with mlflow.start_run():
        # Log dataset information
        mlflow.log_param("dataset", "wine")
        mlflow.log_param("n_samples", X.shape[0])
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_classes", len(np.unique(y)))
        mlflow.log_param("test_size", 0.2)
        
        # Create the pipeline with custom transformer and classifier
        print("\nBuilding pipeline with custom transformer...")
        pipeline = Pipeline([
            ('custom_features', CustomFeatureTransformer(
                add_squared=True, 
                add_interactions=True
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        # Log model parameters
        mlflow.log_param("custom_transformer", "CustomFeatureTransformer")
        mlflow.log_param("add_squared", True)
        mlflow.log_param("add_interactions", True)
        mlflow.log_param("classifier", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        
        # Train the model
        print("\nTraining model...")
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        print("\nEvaluating model...")
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        print(f"\nTrain Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        
        # Print classification report
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, y_pred_test, target_names=wine.target_names))
        
        # Log the model
        print("\nLogging model to MLFlow...")
        mlflow.sklearn.log_model(
            pipeline, 
            "model",
            registered_model_name="wine-classifier"
        )
        
        # Save model locally for deployment
        model_path = "../model"
        os.makedirs(model_path, exist_ok=True)
        mlflow.sklearn.save_model(pipeline, model_path)
        print(f"\nModel saved to {model_path}")
        
        print("\n" + "="*50)
        print("Training completed successfully!")
        print("="*50)
        
        return pipeline, test_accuracy


if __name__ == "__main__":
    train_model()
