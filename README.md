# Heart Disease UCI – End-to-End MLOps Pipeline

An end-to-end MLOps pipeline for the **Heart Disease UCI dataset**, covering data preprocessing and EDA, model training with experiment tracking, automated testing and CI/CD, containerized API serving, Kubernetes deployment, and monitoring/logging.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Setup Instructions](#setup-instructions)
- [Training the Model](#training-the-model)
- [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
- [Running Tests & Linting](#running-tests--linting)
- [API Usage](#api-usage)
- [Docker](#docker)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Monitoring & Logging](#monitoring--logging)
- [CI/CD Pipeline](#cicd-pipeline)
- [Report & Video](#report--video)
- [Tech Stack](#tech-stack)
- [Repository Link](#repository-link)

## Project Overview

This project builds a classifier that predicts whether a patient is likely to have heart disease, then productionizes it end-to-end. The pipeline covers:

- Data acquisition, cleaning, and exploratory data analysis (EDA).
- Feature engineering and training of multiple classification models.
- Experiment tracking with MLflow.
- Model packaging for reproducible inference.
- Automated testing and CI/CD via GitHub Actions.
- Model serving through a FastAPI REST API.
- Containerization with Docker.
- Deployment to a local Kubernetes cluster via Minikube.
- Monitoring with Prometheus and Grafana, plus application/Kubernetes logging.

## Repository Structure

The main folders and files are organized as follows:

```
mlops-heart-disease-uci/
├── data/                      # Raw and cleaned dataset
├── notebooks/                 # EDA, Model Development notebook and generated plots
├── src/                       # Training, inference, and API source code
│   ├── model_training.py      # ML training & MLflow logging
│   ├── api.py                 # FastAPI application
│   └── ...                    # Any additional utility modules
├── tests/                     # Unit tests for data, models, and API
├── models/                    # Serialized final model and preprocessing pipeline
│   └── final_model_rf.pkl
├── k8s/                       # Kubernetes manifests (Deployment, Service, ServiceMonitor)
│   ├── deployment.yaml
│   ├── service.yaml
│   └── servicemonitor.yaml
├── screenshots/               # Screenshots referenced in the final report
├── .github/
│   └── workflows/             # GitHub Actions CI/CD workflow
├── Dockerfile                 # Container image definition for the API
├── requirements.txt           # Python dependencies
├── REPORT.pdf                 # Exported report
└── README.md                  # This file
```

If any of the optional files are missing in your clone (for example `REPORT.pdf` or a `data/README.md`), they can be added later without affecting the core pipeline.

## Dataset

This project uses the **Heart Disease UCI dataset** from the UCI Machine Learning Repository. It contains patient-level clinical features such as age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG results, maximum heart rate, exercise-induced angina, ST depression, slope of the peak exercise ST segment, number of major vessels, and thalassemia, along with a target label for presence/absence of heart disease.

The dataset is downloaded and cleaned as part of the pipeline. See the `data/` folder and your report for the exact source URL and any preprocessing notes.

## Setup Instructions

### Prerequisites

- Python 3.10+
- Git
- Docker
- Minikube (local Kubernetes)
- kubectl
- A modern browser (for MLflow UI, Grafana, and Prometheus)

### Clone the Repository

```bash
git clone git@github.com:donald01johnson/mlops-heart-disease-uci.git
cd mlops-heart-disease-uci
```

### Create a Virtual Environment

```bash
python -m venv heartdisease
source heartdisease/bin/activate

pip install -r requirements.txt
```

## Training the Model

```bash
source heartdisease/bin/activate
export MLFLOW_ALLOW_FILE_STORE=true
python -m src.model_training
```

This loads the dataset, applies preprocessing, trains multiple models (Logistic Regression and Random Forest), logs experiments to MLflow, and saves the final selected model to `models/final_model_rf.pkl`.

## Experiment Tracking with MLflow

```bash
source heartdisease/bin/activate
export MLFLOW_ALLOW_FILE_STORE=true
mlflow ui --backend-store-uri ./mlruns
```

Open in a browser:

```
http://127.0.0.1:5000
```

Select the `heart-disease-uci-experiments` experiment to view runs, parameters, metrics, and artifacts.

## Running Tests & Linting

```bash
flake8 src/ tests/
pytest tests/ -v
```

## API Usage

The FastAPI application exposes `/predict`, `/health`, and `/metrics` endpoints.

Run locally without Docker:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Sample prediction request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 54, "sex": 1, "cp": 0, "trestbps": 130, "chol": 246,
    "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 2
  }'
```

The OpenAPI/Swagger UI is available at:

```
http://localhost:8000/docs
```

## Docker

Build and run the API container:

```bash
docker build -t heart-disease-api:v2 .

docker run -d -p 8000:8000 --name heart-disease-api-local heart-disease-api:v2

curl -i http://localhost:8000/health
```

## Kubernetes Deployment

```bash
minikube start
minikube status

minikube image load heart-disease-api:v2

kubectl apply -f k8s/
kubectl get pods
```

Access the deployed API via the Minikube service URL:

```bash
URL=$(minikube service heart-disease-api-svc --url)

curl -i "$URL/health"

curl -X POST "$URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 54, "sex": 1, "cp": 0, "trestbps": 130, "chol": 246,
    "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 2
  }'
```

## Monitoring & Logging

- **Prometheus** scrapes metrics from the API's `/metrics` endpoint via a `ServiceMonitor` (`k8s/servicemonitor.yaml`), tracking `api_requests_total`, `api_request_duration_seconds`, and `prediction_requests_total`.
- **Grafana** is connected to Prometheus as a data source and visualizes request rates, latency, and prediction counts.
- **Application/Kubernetes logs** can be inspected with:

```bash
kubectl logs deployment/heart-disease-api-deployment --tail=50
```

## CI/CD Pipeline

A GitHub Actions workflow (`.github/workflows/`) runs automatically on every push/pull request to `main`:

- Sets up the Python environment.
- Installs dependencies from `requirements.txt`.
- Runs `flake8` linting.
- Runs `pytest` unit tests.

The pipeline fails and reports errors clearly if linting or tests do not pass, enforcing a quality gate before merging.

## Report & Video

- Final written report: [`REPORT.md`](./REPORT.md) in the repo root.
- Screenshots referenced in the report are stored in the [`screenshots/`](./screenshots) folder.
- Short video walkthrough of the end-to-end pipeline: **add link here once recorded**.

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| ML/Data | scikit-learn, pandas, numpy |
| Experiment Tracking | MLflow |
| API | FastAPI, Pydantic, Uvicorn |
| Testing/Linting | pytest, flake8 |
| CI/CD | GitHub Actions |
| Containerization | Docker |
| Orchestration | Kubernetes (Minikube) |
| Monitoring | Prometheus, Grafana |

## Repository Link

```
https://github.com/donald01johnson/mlops-heart-disease-uci
```
