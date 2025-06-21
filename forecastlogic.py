# forecast_logic.py
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")

def forecast_top_medicines(df, category):
    df['MONTH'] = df['MONTH'].str.strip().str.title()
    df['MONTH'] = pd.to_datetime(df['MONTH'], format='%B', errors='coerce')
    df['Year'] = 2023
    df.dropna(subset=['MONTH'], inplace=True)
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['MONTH'].dt.month.astype(str), format='%Y-%m')
    df.set_index('Date', inplace=True)

    category = category.strip().lower()
    filtered_df = df[df['CATEGORY'].str.lower().str.strip() == category]

    predictions = []

    for medicine in filtered_df['MEDICINE'].unique():
        med_df = filtered_df[filtered_df['MEDICINE'] == medicine]
        ts = med_df['Sales'].resample('M').sum()
        unit_price = med_df['UNITPRICE'].iloc[-1]

        if len(ts.dropna()) < 6 or unit_price == 0:
            continue

        try:
            model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            result = model.fit(disp=False)
            forecast = result.forecast(steps=1)
            predicted_sales = forecast.iloc[0]
            predicted_quantity = predicted_sales / unit_price
            predictions.append((medicine, predicted_quantity))
        except:
            continue

    predicted_df = pd.DataFrame(predictions, columns=['Medicine', 'Predicted_Quantity'])
    top_10 = predicted_df.sort_values(by='Predicted_Quantity', ascending=False).head(10)

    return top_10
