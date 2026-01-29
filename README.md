# example-ml-deployment

An example repo showing how to train Sci-Kit Learn pipelines locally and deploy to Google Cloud Run.

## Overview

This project demonstrates a complete machine learning deployment pipeline featuring:

- **Custom Feature Transformer**: A scikit-learn transformer that creates polynomial and interaction features
- **ML Pipeline**: Uses sklearn's Pipeline API for clean, reproducible model training
- **MLFlow Integration**: Tracks experiments and saves models as MLFlow artifacts
- **REST API**: Flask-based API for serving predictions
- **Containerization**: Docker container for consistent deployment
- **CI/CD Pipeline**: GitHub Actions workflow for automated training and deployment to Google Cloud Run

## Project Structure

```
.
├── .github/
│   └── workflows/
│       └── train-and-deploy.yml  # CI/CD pipeline
├── src/
│   ├── transformer.py            # Custom feature transformer
│   ├── train.py                  # Model training script
│   └── app.py                    # Flask API server
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container configuration
└── README.md                     # This file
```

## Features

### Custom Feature Transformer

The `CustomFeatureTransformer` creates additional features:
- Squared features for non-linear relationships
- Pairwise interaction terms between features
- Automatic normalization (zero mean, unit variance)

### Wine Classification Model

- Uses the **Wine dataset** from sklearn (178 samples, 13 features, 3 classes)
- Trains a Random Forest classifier
- Achieves high accuracy on this well-separated dataset

## Local Development

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/jwgwalton/example-ml-deployment.git
cd example-ml-deployment

# Install dependencies
pip install -r requirements.txt
```

### Training the Model

```bash
cd src
python train.py
```

This will:
1. Load the Wine dataset
2. Create train/test split
3. Build a pipeline with custom transformer and Random Forest classifier
4. Train the model
5. Log metrics and model to MLFlow
6. Save the model to the `model/` directory

### Running the API Locally

```bash
# Make sure you've trained the model first
cd src
python app.py
```

The API will be available at `http://localhost:8080`

### Testing the API

```bash
# Health check
curl http://localhost:8080/health

# Get model info
curl http://localhost:8080/info

# Make a prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[13.2, 2.77, 2.51, 18.5, 96.6, 1.09, 0.52, 0.2, 0.29, 1.98, 0.13, 1.51, 660]]
  }'
```

## API Endpoints

### GET /
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "Wine Classification API"
}
```

### GET /health
Detailed health check

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "./model"
}
```

### GET /info
Model information

**Response:**
```json
{
  "model_type": "Wine Classification",
  "classes": ["class_0", "class_1", "class_2"],
  "n_features": 13,
  "feature_names": ["alcohol", "malic_acid", "ash", ...]
}
```

### POST /predict
Make predictions

**Request:**
```json
{
  "features": [[13.2, 2.77, 2.51, 18.5, 96.6, 1.09, 0.52, 0.2, 0.29, 1.98, 0.13, 1.51, 660]]
}
```

**Response:**
```json
{
  "predictions": [0],
  "probabilities": [[0.95, 0.03, 0.02]]
}
```

## Docker

### Build the Docker Image

```bash
# Train the model first
cd src && python train.py && cd ..

# Build the image
docker build -t wine-classifier .
```

### Run the Container

```bash
docker run -p 8080:8080 wine-classifier
```

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/train-and-deploy.yml`) automates:

1. **Training Job**: 
   - Installs dependencies
   - Trains the model with MLFlow
   - Uploads model artifacts

2. **Deployment Job** (only on main branch):
   - Downloads trained model
   - Builds Docker image
   - Pushes to Google Artifact Registry
   - Deploys to Google Cloud Run

### Required GitHub Secrets

To enable deployment to Google Cloud Run, configure these secrets in your GitHub repository:

- `GCP_PROJECT_ID`: Your Google Cloud project ID
- `GCP_SA_KEY`: Service account key JSON with permissions:
  - Cloud Run Admin
  - Artifact Registry Writer
  - Service Account User

**Security Note:** The default workflow configuration deploys the API with `--allow-unauthenticated` for demonstration purposes. For production use, you should:
- Remove the `--allow-unauthenticated` flag
- Implement authentication (API keys, OAuth, Cloud IAM)
- Add rate limiting to prevent abuse
- Consider using Cloud Armor for DDoS protection

### Setting Up Google Cloud

```bash
# Create a service account
gcloud iam service-accounts create github-actions \
  --display-name="GitHub Actions"

# Grant necessary permissions
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:github-actions@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:github-actions@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:github-actions@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

# Create key
gcloud iam service-accounts keys create key.json \
  --iam-account=github-actions@PROJECT_ID.iam.gserviceaccount.com

# Create Artifact Registry repository
gcloud artifacts repositories create wine-classifier \
  --repository-format=docker \
  --location=us-central1
```

## MLFlow

The training script logs all experiments to MLFlow. To view the MLFlow UI:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

## Model Details

### Input Features (Wine Dataset)

1. Alcohol
2. Malic acid
3. Ash
4. Alcalinity of ash
5. Magnesium
6. Total phenols
7. Flavanoids
8. Nonflavanoid phenols
9. Proanthocyanins
10. Color intensity
11. Hue
12. OD280/OD315 of diluted wines
13. Proline

### Output Classes

- Class 0: Wine type 0
- Class 1: Wine type 1
- Class 2: Wine type 2

## License

MIT
