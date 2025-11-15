from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from .schemas import ChurnInput
from .predict import preprocess_input, predict_and_explain

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predict customer churn with SHAP explanations",
    version="1.0.0"
)

@app.post("/predict", response_model=dict)
async def predict_churn(payload: ChurnInput):
    try:
        X = preprocess_input(payload)
        result = predict_and_explain(X)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Churn Prediction API is running. Use POST /predict"}