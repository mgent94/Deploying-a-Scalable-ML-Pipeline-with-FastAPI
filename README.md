# Deploying a Scalable ML Pipeline with FastAPI

This repository contains an end-to-end MLOps-style project that trains and deploys a machine learning model to predict whether an individual's income is `<=50K` or `>50K` based on census data.

The project includes:

- Data processing (`ml/data.py`)
- Model training and evaluation (`ml/model.py`, `train_model.py`)
- Slice-based model performance analysis (`slice_output.txt`)
- Unit tests (`test_ml.py`)
- A FastAPI service for inference (`main.py`)
- A local client script to call the API (`local_api.py`)
- CI pipeline using GitHub Actions (`.github/workflows/ci.yml`)
- A model card (`model_card.md`)

---

## Environment Setup

1. Install Anaconda or Miniconda.
2. Clone this repository.
3. Create and activate the environment (example):

   ```bash
   conda create -n fastapi python=3.10
   conda activate fastapi
   pip install -r requirements.txt

   or 

   conda env create -f environment.yml
   conda activate fastapi

> Note: Windows users do not require gunicorn; it is intentionally omitted
---

# How to Use and Run This Project

This repository contains an end-to-end machine learning pipeline for training, evaluating, and deploying a Census Income classification model using FastAPI. The sections below describe how to install, run, and test the project.



# How to Run the Program

## 1. Train the Model
Run:
    python train_model.py

Outputs:
- model/model.pkl
- model/encoder.pkl
- model/lb.pkl
- slice_output.txt

---

## 2. Start the FastAPI Service
Run:
    uvicorn main:app --reload

Service URLs:
- http://127.0.0.1:8000/
- http://127.0.0.1:8000/docs

---

## 3. Run the Local API Client
Run:
    python local_api.py

---

---

## Run Unit Tests
Run:
    pytest

---

## Lint the Project
Run:
    flake8

---

## Continuous Integration (CI)
Workflow file:
    .github/workflows/ci.yml


---

## Supporting Files
- model_card.md
- slice_output.txt
- screenshots/
