# 🩺 Diabetes Prediction Model – Your First MLOps Project (FastAPI + Docker + K8s)


This project helps you learn **Building and Deploying an ML Model** using a simple and real-world use case: predicting whether a person is diabetic based on health metrics. We’ll go from:

- ✅ Model Training
- ✅ Building the Model locally
- ✅ API Deployment with FastAPI
- ✅ Dockerization
- ✅ Kubernetes Deployment

---

## 📊 Problem Statement

Predict if a person is diabetic based on:
- Pregnancies
- Glucose
- Blood Pressure
- BMI
- Age

We use a Random Forest Classifier trained on the **Pima Indians Diabetes Dataset**.

---

## 🚀 Quick Start

### 1. Clone the Repo

```bash
git clone https://github.com/Karthik016996148/Diabetes-Prediction.git
cd Diabetes-Prediction
```

### 2. Create Virtual Environment

#### Windows (PowerShell)

```powershell
py -m venv .mlops
.\.mlops\Scripts\Activate.ps1
```

#### Linux/macOS (bash/zsh)

```bash
python3 -m venv .mlops
source .mlops/bin/activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

## Train the Model

```
python train.py
```

This creates deployable artifacts in `artifacts/` (model + metadata + metrics).

## Run the API Locally

```
uvicorn main:app --reload
```

### Sample Input for /predict

```
{
  "Pregnancies": 2,
  "Glucose": 130,
  "BloodPressure": 70,
  "BMI": 28.5,
  "Age": 45
}
```

## Dockerize the API

### Build the Docker Image

```
docker build -t diabetes-prediction:latest .
```

### Run the Container

```
docker run -p 8000:8000 diabetes-prediction:latest
```

## Deploy to Kubernetes

```
kubectl apply -f k8s-deploy.yml
```


