from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# Initialize FastAPI app
app = FastAPI(
    title="Bankruptcy Prevention API",
    description="A production-ready API for predicting bankruptcy risk using Machine Learning.",
    version="1.0.0"
)

# Load the trained model
MODEL_PATH = "bankruptcy_model.joblib"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

# Input Data Model
class CompanyRiskFeatures(BaseModel):
    industrial_risk: float
    management_risk: float
    financial_flexibility: float
    credibility: float
    competitiveness: float
    operating_risk: float

@app.get("/")
def home():
    return {
        "status": "online",
        "project": "Bankruptcy Prevention AI",
        "api_docs": "/docs"
    }

@app.post("/predict")
def predict_bankruptcy(features: CompanyRiskFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Machine Learning model not found on server.")
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([features.dict()])
    
    # Make Prediction
    try:
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0].tolist()
        
        # Mapping: 0 = Bankruptcy, 1 = Non-Bankruptcy
        status = "Bankruptcy" if prediction == 0 else "Non-Bankruptcy"
        confidence = probabilities[prediction]
        
        return {
            "prediction_code": prediction,
            "status": status,
            "confidence": f"{confidence * 100:.2f}%",
            "probabilities": {
                "Bankruptcy": probabilities[0],
                "Non-Bankruptcy": probabilities[1]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Port 8080 is the default for GCP Cloud Run
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
