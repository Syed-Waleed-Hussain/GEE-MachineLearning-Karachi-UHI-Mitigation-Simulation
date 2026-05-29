from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import os
import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")
BASELINE_DATA_PATH = os.path.join(BASE_DIR, "baseline_data.csv")
METADATA_PATH = os.path.join(BASE_DIR, "model_metadata.json")
FORECAST_PATH = os.path.join(BASE_DIR, "time_series_forecast.json")

try:
    best_model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    
    if metadata.get("requires_scaling"):
        scaler = joblib.load(SCALER_PATH)
    else:
        scaler = None

    baseline_df = pd.read_csv(BASELINE_DATA_PATH)
except Exception as e:
    print(f"Error loading models or data: {e}")
    best_model = None
    scaler = None
    baseline_df = pd.DataFrame()

class SimulationParams(BaseModel):
    delta_ndvi_percent: float = 0.0  
    delta_albedo: float = 0.0        
    delta_ndbi_percent: float = 0.0  

@app.get("/api/baseline")
def get_baseline():
    if baseline_df.empty:
        raise HTTPException(status_code=500, detail="Baseline data not available.")
    data = baseline_df[['latitude', 'longitude', 'LST', 'NDBI', 'NDVI']].to_dict(orient="records")
    return {"data": data}

@app.post("/api/simulate")
def simulate(params: SimulationParams):
    if baseline_df.empty or best_model is None:
        raise HTTPException(status_code=500, detail="Models or data not available.")
    
    sim_df = baseline_df.copy()

    # Apply changes
    sim_df['NDVI'] = sim_df['NDVI'] * (1 + params.delta_ndvi_percent / 100.0)
    
    # Assuming Albedo is scaled by 10000, 1.0 intensity = +2000 albedo (equivalent to +0.2 in raw 0-1 albedo)
    sim_df['Albedo'] = sim_df['Albedo'] + (params.delta_albedo * 2000.0)

    sim_df['NDBI'] = sim_df['NDBI'] * (1 + params.delta_ndbi_percent / 100.0)

    features = ['Albedo', 'NDBI', 'NDVI', 'NDWI']
    X_sim = sim_df[features]

    if scaler is not None:
        X_sim = scaler.transform(X_sim)
    
    new_lst = best_model.predict(X_sim)
    
    result_df = sim_df[['latitude', 'longitude', 'NDBI', 'NDVI']].copy()
    result_df['LST'] = new_lst
    
    return {"data": result_df.to_dict(orient="records")}

@app.get("/api/forecast")
def get_forecast():
    try:
        with open(FORECAST_PATH, "r") as f:
            forecast_data = json.load(f)
        return forecast_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast data not available: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
