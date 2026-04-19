# Bankruptcy Prevention AI: Production-Ready REST API

This project aims to predict the likelihood of business bankruptcy based on six key risk factors. It has been evolved from a basic script to a **Production-Ready REST API** using **FastAPI**, optimized for serverless deployment on **Google Cloud Platform (GCP)**.

## Project Structure

- `bankruptcy-prevention.xlsx`: Raw input data containing semicolon-separated risk scores.
- `data_pipeline.py`: Cleans raw data, parses feature strings, and removes duplicates (Result: 103 unique samples).
- `final_bankruptcy_model.py`: Performs an expanded comparison of **10 different machine learning models** using 10-fold cross-validation.
- `dashboard.py`: An interactive **Streamlit Dashboard** with live predictions and analytics.
- `main.py`: A professional **FastAPI backend** (API fallback).
- `Dockerfile`: A containerization configuration optimized for **GCP Cloud Run** (now launches the dashboard).
- `requirements.txt`: Minimal dependencies including `streamlit` and `plotly` for the UI.
- `run_all.py`: Master script to execute the entire training pipeline in sequence.

## Features (Data Dictionary)

All features are scored on a risk scale: `0` (Low Risk), `0.5` (Medium Risk), `1.0` (High Risk).

1. **Industrial Risk**: Risk related to the specific industry sector.
2. **Management Risk**: Risk associated with the quality of business leadership.
3. **Financial Flexibility**: The capability of the business to adapt financially (**Secondary Predictor**).
4. **Credibility**: Measured trustworthiness and historical financial reliability.
5. **Competitiveness**: Market positioning and competitive edge (**Primary Predictor**).
6. **Operating Risk**: Risks related to daily business operations.

## Model Performance (Best Model: KNN)

We evaluated 10 different algorithms (Random Forest, SVM, Logistic Regression, etc.). **K-Nearest Neighbors (KNN)** emerged as the most robust champion.

- **Cross-Validation Accuracy**: 99.09% (Measured across 10 folds)
- **Consistency**: Extremely high stability (+/- 5.45% variance)
- **Primary Driver**: `Competitiveness` (~53% influence on prediction).

## How to Run

### 1. Execute the Training Pipeline
This will clean the data and save the best model (KNN) into `bankruptcy_model.joblib`:
```bash
python run_all.py
```

### 2. Run the API Locally
Start the FastAPI production server:
```bash
python main.py
```
*Wait for the server to start on Port 8080.*

### 3. Access API Documentation
Once the server is running, visit:
`http://localhost:8080/docs`
This provides an interactive interface to test the `/predict` endpoint.

## Deployment & CI/CD

This repository is designed for **Google Cloud Run** with automatic deployment via GitHub.

### Automatic Deployment (Preferred)
1. **Push to GitHub**: Connect your local repo to GitHub and push to the `main` branch.
2. **Setup Cloud Run Service**:
   - Go to GCP Console -> Cloud Run -> Create Service.
   - Select **"Continuously deploy from a repository"**.
   - Connect your GitHub repo.
   - Choose **Dockerfile** as the build configuration.
3. Every time you `git push origin main`, GCP will automatically build and deploy a new version of your API.

### Manual Deployment (CLI)
If you prefer manual deployment via terminal:
```bash
gcloud run deploy bankruptcy-api --source . --region us-central1 --allow-unauthenticated
```


---
*Developed for the Bankruptcy Prevention Task • Production Version 1.1*
