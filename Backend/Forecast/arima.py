import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from pmdarima import auto_arima

from Forecast.model import predict
from Forecast import data_processing


def run():
    data_processing.run()
    # Load the dataset
    df = pd.read_csv("/home/captain/Desktop/NLP_FISAC/Backend/Forecast/processed_sentiment.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")  # Ensure chronological order
    df.set_index("Date", inplace=True)

    # Normalize Sentiment Values (for better performance)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df["Sentiment"] = scaler.fit_transform(df[["Sentiment"]])

    # Split data into train and test (Before 2024 for training)
    train = df[df.index < "2024-01-01"]
    test = df[df.index >= "2024-01-01"]

    best_model = auto_arima(train["Sentiment"], seasonal=False, trace=True)

    model = ARIMA(train["Sentiment"], order=(0,0,0))
    model_fitted = model.fit()

    # Forecast
    forecast = model_fitted.forecast(steps=len(test))

    # Inverse transform predictions
    forecast = scaler.inverse_transform(forecast.values.reshape(-1, 1)).flatten()

    forecast = predict(test)

    # Plot Predictions vs. Actual
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test["Sentiment"], label="Actual Sentiment", color="blue")
    plt.plot(test.index, forecast, label="Predicted Sentiment", color="red", linestyle="dashed")
    plt.xlabel("Date")
    plt.ylabel("Sentiment Score")
    plt.title("ARIMA Sentiment Prediction")
    plt.legend()
    plt.grid(True)
    plt.savefig("/home/captain/Desktop/NLP_FISAC/Backend/Forecast/arima_sentiment_prediction.png", dpi=300)

if __name__ == '__main__':
    run()