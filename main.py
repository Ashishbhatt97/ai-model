from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle
from forecastlogic import forecast_top_medicines

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Allows all origins; replace with specific origins for production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load pickled data
with open("forecast_model.pkl", "rb") as f:
    df, _ = pickle.load(f)  # only load the df, use the imported function


@app.get("/")
def root():
    return {"message": "Welcome to the Medicine Forecast API!"}


@app.get("/predict/")
def predict(category: str = Query(..., description="Enter the medicine category")):
    try:
        result = forecast_top_medicines(df.copy(), category)
        predictions = [
            {
                "medicine": row["Medicine"],
                "predicted_quantity": int(round(row["Predicted_Quantity"])),
            }
            for _, row in result.iterrows()
        ]
        return {"category": category, "top_10_predictions": predictions}
    except Exception as e:
        return {"error": str(e)}
