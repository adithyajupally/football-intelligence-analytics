from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.match_prediction import predict_match


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "match_prediction_model.pkl"
DATA_PATH = BASE_DIR / "data" / "matches_prepared.csv"


# Load model and historical data
import joblib

model = joblib.load(MODEL_PATH)
history_df = pd.read_csv(DATA_PATH)


# Create FastAPI app
app = FastAPI(
    title="Football Intelligence Prediction API",
    description="API for international football match predictions.",
    version="1.0.0",
)


# Allow the Next.js frontend to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://football-intelligence-analytics.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    neutral: bool = False
    tournament: str = "Friendly"


@app.get("/")
def root():
    return {
        "message": "Football Intelligence Prediction API",
        "status": "running",
        "endpoint": "/predict",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "data_loaded": history_df is not None,
    }


@app.post("/predict")
def predict(request: MatchRequest):

    if not request.home_team.strip():
        raise HTTPException(status_code=400, detail="Home team is required.")

    if not request.away_team.strip():
        raise HTTPException(status_code=400, detail="Away team is required.")

    if request.home_team.strip().lower() == request.away_team.strip().lower():
        raise HTTPException(
            status_code=400,
            detail="Home team and away team must be different.",
        )

    try:
        result = predict_match(
            model=model,
            history_df=history_df,
            home_team=request.home_team,
            away_team=request.away_team,
            neutral=request.neutral,
            tournament=request.tournament,
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )