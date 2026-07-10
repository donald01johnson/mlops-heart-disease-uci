# ❤️ Heart Disease UCI — End-to-End MLOps Pipeline

[![CI](https://github.com/donald01johnson/mlops-heart-disease-uci/actions/workflows/ci.yml/badge.svg)](https://github.com/donald01johnson/mlops-heart-disease-uci/actions/workflows/ci.yml)
![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?logo=kubernetes&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?logo=prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-F46800?logo=grafana&logoColor=white)

>
> A production-grade machine learning pipeline that predicts heart disease risk using the UCI Cleveland Heart Disease dataset. The project spans the full MLOps lifecycle: data engineering, model development with experiment tracking, automated testing & CI/CD, containerised API serving, Kubernetes orchestration, and observability with Prometheus + Grafana.

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Repository Structure](#-repository-structure)
- [Dataset](#-dataset)
- [Setup & Installation](#-setup--installation)
- [Data Pipeline & EDA](#-data-pipeline--eda)
- [Model Training & Experiment Tracking](#-model-training--experiment-tracking)
- [API Serving](#-api-serving)
- [Testing & Linting](#-testing--linting)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Docker](#-docker)
- [Kubernetes Deployment](#-kubernetes-deployment)
- [Monitoring & Logging](#-monitoring--logging)
- [Screenshots & Evidence](#-screenshots--evidence)
- [Tech Stack](#-tech-stack)
- [Author](#-author)

---

## 🔍 Project Overview

This project implements a **binary classification model** that predicts whether a patient is likely to have heart disease (`target = 1`) or not (`target = 0`). It goes beyond a Jupyter notebook prototype to demonstrate **production MLOps best practices**:

| Stage | What's Covered |
|---|---|
| **Data Engineering** | Automated data download from UCI ML Repository, cleaning, and feature pipeline |
| **EDA** | Histograms, correlation heatmaps, class-distribution charts, feature-vs-target analysis |
| **Model Development** | Logistic Regression & Random Forest with `GridSearchCV` + `StratifiedKFold` |
| **Experiment Tracking** | MLflow logging of parameters, metrics (accuracy, precision, recall, F1, AUC-ROC), artifacts, and model registry |
| **API Serving** | FastAPI REST endpoint with Pydantic validation & Prometheus metrics instrumentation |
| **Testing & Linting** | `pytest` unit tests for data pipeline and model training; `flake8` linting |
| **CI/CD** | GitHub Actions workflow triggered on push/PR to `main` |
| **Containerisation** | Docker image based on `python:3.10-slim` |
| **Orchestration** | Kubernetes Deployment + Service + ServiceMonitor on Minikube |
| **Monitoring** | Prometheus scraping custom `/metrics` endpoint; Grafana dashboards for latency, request counts & prediction status |
| **Logging** | Structured application logging + Kubernetes pod logging via `kubectl logs` |

---

## 🏗 Architecture

```
┌──────────────┐    ┌─────────────────┐    ┌────────────────┐
│  UCI ML Repo │───▶│  Data Pipeline  │───▶│  EDA Notebook  │
│  (download)  │    │  (clean & feat) │    │  (visualize)   │
└──────────────┘    └────────┬────────┘    └────────────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  Model Training     │
                  │  (LR + RF + GridCV) │
                  │  + MLflow Tracking  │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  FastAPI Serving    │
                  │  POST /predict      │
                  │  GET  /metrics      │
                  └──────────┬──────────┘
                             │
              ┌──────────────┼───────────────┐
              ▼              ▼               ▼
     ┌──────────────┐ ┌────────────┐ ┌──────────────┐
     │   Docker     │ │ Kubernetes │ │  Prometheus  │
     │   Container  │ │ (Minikube) │ │  + Grafana   │
     └──────────────┘ └────────────┘ └──────────────┘
              │              │               │
              └──────────────┼───────────────┘
                             ▼
                  ┌─────────────────────┐
                  │   GitHub Actions    │
                  │   CI/CD Pipeline    │
                  └─────────────────────┘
```

---

## 📁 Repository Structure

```
mlops-heart-disease-uci/
│
├── .github/
│   └── workflows/
│       └── ci.yml                        # GitHub Actions CI pipeline
│
├── data/
│   └── cleveland_heart.csv               # Cleaned Cleveland Heart Disease dataset
│
├── notebooks/
│   ├── 01_eda_heart_disease.ipynb        # Exploratory Data Analysis notebook
│   ├── 02_model_development.ipynb        # Model training & evaluation notebook
│   ├── correlation_heatmap.png           # Feature correlation heatmap
│   ├── class_distribution.png            # Target class balance chart
│   ├── histogram.png                     # Feature distribution histograms
│   ├── missing_values.png                # Missing values analysis
│   ├── agevshd.png                       # Age vs Heart Disease
│   ├── cholvshd.png                      # Cholesterol vs Heart Disease
│   └── thalachvshd.png                   # Max Heart Rate vs Heart Disease
│
├── src/
│   ├── __init__.py
│   ├── download_data.py                  # Script to download data from UCI repo
│   ├── data_pipeline.py                  # Feature definitions, loading & preprocessing
│   ├── model_training.py                 # Training, hyperparameter tuning & MLflow logging
│   └── api.py                            # FastAPI app with Prometheus instrumentation
│
├── tests/
│   ├── conftest.py                       # Shared pytest fixtures
│   ├── test_data_pipeline.py             # Unit tests for data loading & preprocessing
│   └── test_model_training.py            # Unit tests for model training functions
│
├── models/
│   └── final_model_rf.pkl                # Serialized best Random Forest model
│
├── k8s/
│   ├── deployment.yaml                   # Kubernetes Deployment manifest
│   ├── service.yaml                      # Kubernetes Service (NodePort)
│   └── servicemonitor.yaml               # Prometheus ServiceMonitor CRD
│
├── screenshots/                          # Evidence screenshots for the report
│   ├── MLflow*.png                       # MLflow UI screenshots
│   ├── KD*.png                           # Kubernetes deployment screenshots
│   ├── Grafana*.png                      # Grafana dashboard screenshots
│   ├── Docker_build*.png                 # Docker build screenshots
│   ├── Apilogs*.png                      # API logging screenshots
│   ├── Failure*.png                      # CI failure evidence screenshots
│   └── Error*.png                        # Error handling screenshots
│
├── Dockerfile                            # Container image definition
├── requirements.txt                      # Python dependencies
├── .gitignore
└── README.md                             # ← You are here
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Name** | [Heart Disease UCI (Cleveland)](https://archive.ics.uci.edu/dataset/45/heart+disease) |
| **Source** | UCI Machine Learning Repository |
| **Instances** | 303 |
| **Features** | 13 clinical attributes |
| **Target** | Binary — `0` (no disease) / `1` (disease present) |

### Feature Descriptions

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Age in years |
| `sex` | Categorical | Sex (1 = male; 0 = female) |
| `cp` | Categorical | Chest pain type (0–3) |
| `trestbps` | Numeric | Resting blood pressure (mm Hg) |
| `chol` | Numeric | Serum cholesterol (mg/dl) |
| `fbs` | Categorical | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) |
| `restecg` | Categorical | Resting ECG results (0–2) |
| `thalach` | Numeric | Maximum heart rate achieved |
| `exang` | Categorical | Exercise-induced angina (1 = yes; 0 = no) |
| `oldpeak` | Numeric | ST depression induced by exercise |
| `slope` | Categorical | Slope of the peak exercise ST segment |
| `ca` | Categorical | Number of major vessels coloured by fluoroscopy (0–3) |
| `thal` | Categorical | Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect) |

---

## ⚙ Setup & Installation

### Prerequisites

- Python 3.10+
- pip
- Docker (for containerisation)
- Minikube & kubectl (for Kubernetes deployment)
- Git

### Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/donald01johnson/mlops-heart-disease-uci.git
cd mlops-heart-disease-uci

# 2. Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Download Dataset (Optional — already included)

```bash
python -m src.download_data
```

---

## 📈 Data Pipeline & EDA

The data pipeline is implemented in [`src/data_pipeline.py`](src/data_pipeline.py) and handles:

1. **Data Loading** — Reads `data/cleveland_heart.csv` into a Pandas DataFrame.
2. **Feature Separation** — Splits into numeric features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) and categorical features (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`).
3. **Preprocessing Pipeline** — Uses `sklearn.compose.ColumnTransformer`:
   - **Numeric**: `SimpleImputer` (median strategy) → `StandardScaler`
   - **Categorical**: `SimpleImputer` (most frequent) → `OneHotEncoder`

### EDA Highlights

The exploratory analysis is in [`notebooks/01_eda_heart_disease.ipynb`](notebooks/01_eda_heart_disease.ipynb) and covers:

- **Missing value analysis** — No missing values found in the cleaned Cleveland subset
- **Class distribution** — Near-balanced classes (54% disease, 46% no disease)
- **Feature distributions** — Histograms of all 13 features
- **Correlation heatmap** — Identifies relationships between numeric features
- **Feature-vs-target plots** — Age, cholesterol, and max heart rate vs. disease presence

---

## 🤖 Model Training & Experiment Tracking

### Model Development

Implemented in [`src/model_training.py`](src/model_training.py):

| Model | Details |
|---|---|
| **Logistic Regression** | `solver='liblinear'`, `max_iter=1000` |
| **Random Forest** | `random_state=42`, with hyperparameter grid search |

**Hyperparameter Tuning:**
- `GridSearchCV` with `StratifiedKFold` (5 folds)
- Scoring metric: `roc_auc`
- Parallelised with `n_jobs=-1`

**Best Model:** Random Forest (serialized as `models/final_model_rf.pkl`)

### MLflow Experiment Tracking

All training runs are tracked using **MLflow**:

```bash
# Start the MLflow UI
mlflow ui --port 5000
```

**What's logged per run:**
- Hyperparameters (e.g., `n_estimators`, `max_depth`)
- Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Confusion matrix plots
- ROC curve artifacts
- Trained model artifacts

---

## 🌐 API Serving

The prediction API is built with **FastAPI** and defined in [`src/api.py`](src/api.py).

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Returns heart disease prediction |
| `GET` | `/metrics` | Prometheus-compatible metrics |

### Running the API Locally

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### Sample Prediction Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
  }'
```

### Prometheus Instrumentation

The API exposes the following custom metrics at `/metrics`:

| Metric | Type | Description |
|---|---|---|
| `api_requests_total` | Counter | Total API requests (by method, endpoint, status) |
| `api_request_duration_seconds` | Histogram | Request latency distribution (by endpoint) |
| `prediction_requests_total` | Counter | Total prediction requests (by success/error) |

---

## 🧪 Testing & Linting

### Unit Tests

Tests are located in the [`tests/`](tests/) directory and use **pytest**:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage (optional)
pytest tests/ -v --cov=src
```

| Test File | What It Covers |
|---|---|
| `test_data_pipeline.py` | Data loading, feature counts, preprocessing output shape |
| `test_model_training.py` | Train/test split, model training, prediction output format |

### Linting

```bash
# Run flake8 linting
flake8 src/ tests/
```

---

## 🔄 CI/CD Pipeline

The CI pipeline is defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) and runs on every **push** and **pull request** to the `main` branch.

### Pipeline Steps

```
Push/PR to main
       │
       ▼
┌──────────────┐
│  Checkout    │
│  Repository  │
├──────────────┤
│  Setup       │
│  Python 3.10 │
├──────────────┤
│  Install     │
│  Dependencies│
├──────────────┤
│  Run Linting │
│  (flake8)    │
├──────────────┤
│  Run Tests   │
│  (pytest)    │
└──────────────┘
```

---

## 🐳 Docker

### Build the Image

```bash
docker build -t heart-disease-api:latest .
```

### Run the Container

```bash
docker run -d -p 8000:8000 --name heart-api heart-disease-api:latest
```

### Verify

```bash
# Health check
curl http://localhost:8000/

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":55,"sex":1,"cp":2,"trestbps":130,"chol":250,"fbs":0,"restecg":1,"thalach":160,"exang":0,"oldpeak":1.4,"slope":2,"ca":1,"thal":3}'
```

### Dockerfile Overview

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY models/ ./models/
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## ☸ Kubernetes Deployment

### Prerequisites

- [Minikube](https://minikube.sigs.k8s.io/docs/start/) installed and running
- `kubectl` configured

### Deploy to Minikube

```bash
# 1. Start Minikube
minikube start

# 2. Use Minikube's Docker daemon
eval $(minikube docker-env)      # Linux/macOS
# For Windows PowerShell:
# & minikube -p minikube docker-env --shell powershell | Invoke-Expression

# 3. Build the Docker image inside Minikube
docker build -t heart-disease-api:latest .

# 4. Apply Kubernetes manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/servicemonitor.yaml

# 5. Check deployment status
kubectl get pods
kubectl get services

# 6. Access the API
minikube service heart-disease-api-service --url
```

### Kubernetes Manifests

| File | Purpose |
|---|---|
| `k8s/deployment.yaml` | Deploys the API container with replica management |
| `k8s/service.yaml` | Exposes the deployment via NodePort |
| `k8s/servicemonitor.yaml` | Prometheus ServiceMonitor for auto-scraping `/metrics` |

---

## 📡 Monitoring & Logging

### Prometheus

- Scrapes the custom `/metrics` endpoint exposed by FastAPI
- **ServiceMonitor** auto-discovers the Kubernetes service
- Tracks: request count, latency histograms, prediction success/error rates

### Grafana

Grafana dashboards visualise the Prometheus metrics:

- **API Request Rate** — Requests per second over time
- **Request Latency (p50/p95/p99)** — Response time distribution
- **Prediction Counter** — Success vs. error predictions
- **HTTP Status Codes** — Breakdown by 2xx, 4xx, 5xx

### Application Logging

Structured logging is built into the FastAPI app using Python's `logging` module:

```bash
# View logs from the running container
docker logs heart-api

# View logs from Kubernetes pods
kubectl logs -f deployment/heart-disease-api
```

---

## 📸 Screenshots & Evidence

All evidence screenshots are located in the [`screenshots/`](screenshots/) directory:

| Category | Files | Description |
|---|---|---|
| **MLflow** | `MLflow1.png` – `MLflow13.png` | Experiment runs, metrics, model registry |
| **Kubernetes** | `KD1.png` – `KD18.png` | Pod status, deployment rollouts, service exposure |
| **Grafana** | `Grafana1.png` – `Grafana7.png` | Dashboard panels for latency, throughput, predictions |
| **Docker** | `Docker_build1.png` – `Docker_build2.png` | Image build & run evidence |
| **API Logs** | `Apilogs1.png` – `Apilogs2.png` | Structured application logging |
| **CI Failures** | `Failure1.png` – `Failure11.png` | Evidence of failed builds and fixes |
| **Errors** | `Error1.png` – `Error3.png` | Error handling & debugging evidence |

---

## 🛠 Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.10 |
| **Data Processing** | Pandas, NumPy |
| **ML Framework** | scikit-learn |
| **Experiment Tracking** | MLflow |
| **API Framework** | FastAPI + Uvicorn |
| **Data Validation** | Pydantic |
| **Visualisation** | Matplotlib, Seaborn |
| **Testing** | pytest |
| **Linting** | flake8 |
| **Containerisation** | Docker |
| **Orchestration** | Kubernetes (Minikube) |
| **Monitoring** | Prometheus, Grafana |
| **Metrics Library** | prometheus-client |
| **Serialisation** | joblib |
| **CI/CD** | GitHub Actions |
| **Version Control** | Git, GitHub |

---

## 👤 Author

**Donald Johnson A**
Roll No: 2024AB05326

🔗 **GitHub Repository:** [donald01johnson/mlops-heart-disease-uci](https://github.com/donald01johnson/mlops-heart-disease-uci)

---

## 📜 License

This project is developed as part of an academic assignment. Please refer to the repository for licensing details.

---

<p align="center">
  Made with ❤️ for MLOps
</p>
