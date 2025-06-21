# save_pickle.py
import pandas as pd
import pickle
from forecastlogic import forecast_top_medicines  # import from module

# Load your CSV file
df = pd.read_csv("finalmodel.csv")

# Pickle the function + dataframe
with open("forecast_model.pkl", "wb") as f:
    pickle.dump((df, forecast_top_medicines), f)

print("Pickle file saved successfully.")
