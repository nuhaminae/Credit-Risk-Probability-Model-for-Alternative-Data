from fastapi import FastAPI
from scripts.api.pydantic_models import RiskRequest, RiskResponse
import mlflow.sklearn
import joblib
import os
import pandas as pd

# Ensure you have the necessary packages installed:
app = FastAPI(title="Credit Risk API")

# Load trained feature list
features_path = "models/features_list.pkl"
features_used = joblib.load(features_path)

def preprocess_input(request_dict):
    """
    Aligns input JSON to trained feature structure, fills missing values with zeros.
    """
    df = pd.DataFrame([request_dict])
    df = df.reindex(columns=features_used, fill_value=0)
    return df

# Load the best model locally — ensure this matches your trained model type and saved filename
model_dir = "models"
known_models = [
    "gradientboostingclassifier",
    "randomforestclassifier",
    "logisticregression",
    "decisiontreeclassifier"
]

# Find the most recently saved model file
model = None
for model_id in known_models:
    candidate = os.path.join(model_dir, f"{model_id}_credit_model_boosted.pkl")
    if os.path.exists(candidate):
        print(f"Found model: {candidate}")
        #model = mlflow.sklearn.load_model(candidate)
        model = joblib.load(candidate)
        print(f"Loaded model: {model_id}")
        break

if model is None:
    raise RuntimeError("No trained model found in 'models' directory.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Credit Risk API. Use /docs to interact."}

@app.post("/predict", response_model=RiskResponse)
def predict_risk(request: RiskRequest):
    try:
        request_dict = request.model_dump()
        input_df = preprocess_input(request_dict)
        risk_prob = model.predict_proba(input_df)[0][1]
        return RiskResponse(risk_probability=round(risk_prob, 4))
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
