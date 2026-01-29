# example-ml-deployment

An example repo showing how to train Sci-Kit Learn pipelines locally and deploy to Google Cloud Run.

## Overview

This project demonstrates a complete machine learning deployment pipeline featuring:

- **Custom Feature Transformer**: A scikit-learn transformer that creates polynomial and interaction features
- **ML Pipeline**: Uses sklearn's Pipeline API for clean, reproducible model training
- **MLFlow Integration**: Tracks experiments and saves models as MLFlow artifacts (using MLflow 3.5.0+ for security)
- **MLFlow Serving**: Uses MLFlow's built-in model serving for standardized REST API
- **Containerization**: Docker container for consistent deployment
- **CI/CD Pipeline**: GitHub Actions workflow for automated training and deployment to Google Cloud Run
- **Security**: All dependencies updated to latest secure versions

## Project Structure

```
.
├── .github/
│   └── workflows/
│       └── train-and-deploy.yml  # CI/CD pipeline
├── src/
│   ├── transformer.py            # Custom feature transformer
│   └── train.py                  # Model training script
├── tests/
│   └── test_api.py               # Python API test suite
├── examples/
│   └── sample_request.json       # Sample API request
├── pyproject.toml                # UV package configuration
├── uv.lock                       # UV lock file for reproducible builds
├── Dockerfile                    # Container configuration
├── run_tests.sh                  # Automated test script with Docker
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
- [UV](https://github.com/astral-sh/uv) (recommended package manager)

### Installation

**Using UV (recommended):**
```bash
# Clone the repository
git clone https://github.com/jwgwalton/example-ml-deployment.git
cd example-ml-deployment

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (including dev dependencies)
uv pip install -e ".[dev]"
```

**Using pip:**
```bash
# Clone the repository
git clone https://github.com/jwgwalton/example-ml-deployment.git
cd example-ml-deployment

# Install dependencies
pip install -e ".[dev]"
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

### Running the API Locally with MLFlow Serving

```bash
# Make sure you've trained the model first
# From the repository root
mlflow models serve -m ./model -h 0.0.0.0 -p 8080 --no-conda
```

The API will be available at `http://localhost:8080`

### Testing the API

MLFlow serving provides the following endpoints:

```bash
# Health check
curl http://localhost:8080/health

# Make a prediction using MLFlow's /invocations endpoint
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"],
      "data": [[13.2, 2.77, 2.51, 18.5, 96.6, 1.09, 0.52, 0.2, 0.29, 1.98, 0.13, 1.51, 660]]
    }
  }'

# Or use the Python test suite (requires dev dependencies)
uv run --with requests python tests/test_api.py
# Or with activated virtual environment:
python tests/test_api.py
```

### Running Tests

The project includes a comprehensive Python test suite that validates the API endpoints.

**Quick Start (using the test script):**
```bash
# Train the model first
cd src && python train.py && cd ..

# Run all tests with Docker (builds, runs container, tests, and cleans up)
./run_tests.sh
```

**Using UV (recommended):**
```bash
# Start the MLFlow server (in one terminal)
mlflow models serve -m ./model -h 0.0.0.0 -p 8080 --no-conda

# Run tests (in another terminal)
uv run --with requests python tests/test_api.py
```

**Using Docker:**
```bash
# Build and run the Docker container
docker build -t wine-classifier .
docker run -d -p 8080:8080 --name wine-test wine-classifier

# Run tests
uv run --with requests python tests/test_api.py

# Clean up
docker stop wine-test && docker rm wine-test
```

The test suite includes:
- Health endpoint validation
- Single prediction test
- Multiple predictions test
- Automatic service availability checking with retry logic

## API Endpoints

MLFlow serving provides a standardized REST API with the following endpoints:

### GET /health

Health check endpoint to verify the service is running.

**Response:**
```json
{
  "status": "OK"
}
```

### POST /invocations

Main prediction endpoint. Accepts input data in multiple formats.

**Request Format (dataframe_split):**
```json
{
  "dataframe_split": {
    "columns": [
      "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
      "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
      "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"
    ],
    "data": [
      [13.2, 2.77, 2.51, 18.5, 96.6, 1.09, 0.52, 0.2, 0.29, 1.98, 0.13, 1.51, 660]
    ]
  }
}
```

**Response:**
```json
{
  "predictions": [1]
}
```

**Alternative Input Formats:**

MLFlow also supports these input formats:
- `dataframe_records`: `{"dataframe_records": [{"col1": val1, "col2": val2, ...}]}`
- `instances`: `{"instances": [[val1, val2, ...]]}`
- `inputs`: `{"inputs": [[val1, val2, ...]]}`


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
   - Installs dependencies using UV
   - Trains the model with MLFlow
   - Uploads model artifacts

2. **Test Job**:
   - Downloads trained model
   - Builds Docker container
   - Runs container locally
   - Executes Python test suite using UV with `--dev` flag
   - Validates health and prediction endpoints
   - Cleans up container

3. **Deployment Job** (only on main branch, runs after tests pass):
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
