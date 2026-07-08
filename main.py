import json
from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from src import preprocessing

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'models/xgboost.pkl'
FEATURE_COLUMNS_PATH = BASE_DIR / 'models/feature_columns.json'

model = joblib.load(MODEL_PATH)
with open(FEATURE_COLUMNS_PATH) as f:
    feature_columns = json.load(f)

app = FastAPI(title="Ride-Wise Churn Prediction API")

class RiderRawInput(BaseModel):
    age: float = Field(..., examples=[30])
    loyalty_status: str = Field(..., description="Bronze, Silver, Gold, or Platinum", examples=["Gold"])
    city: str = Field(..., description="Cairo, Nairobi, or Lagos", examples=["Nairobi"])
    avg_rating_given: float = Field(..., examples=[4.5])
    account_age_days: int = Field(..., description="Days since signup", examples=[400])
    total_trips: int = Field(..., description="Lifetime trip count", examples=[30])
    total_spent: float = Field(..., description="Lifetime fare spend", examples=[250.5])
    total_tip: float = Field(..., description="Lifetime tip amount", examples=[10])
    avg_surge: float = Field(..., description="Average surge multiplier across trips", examples=[1.1])
    recent_trips: int = Field(..., description="Trips in the last 30 days before the snapshot date", examples=[4])
    prior_trips: int = Field(..., description="Trips 31-60 days before the snapshot date", examples=[6])
    trip_trend: int = Field(..., description="recent_trips - prior_trips", examples=[-2])
    recency_at_cutoff: int = Field(..., description="Days since the rider's last trip", examples=[10])
    session_count: int = Field(..., description="Number of app sessions", examples=[20])
    avg_time_on_app: float = Field(..., description="Average seconds spent per session", examples=[90])
    avg_pages_visited: float = Field(..., description="Average pages visited per session", examples=[3])
    session_conversion_rate: float = Field(..., description="Share of sessions that converted to a booking", examples=[0.3]
    )


@app.get('/')
def health():
    return {"status": "ok", "model": MODEL_PATH.name}


@app.post('/predict')
def predict(rider: RiderRawInput):
    row = preprocessing.encode_new_rider(rider.model_dump(), feature_columns)
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])
    return {"churn_prediction": prediction, "churn_probability": probability}
