import pandas as pd
import numpy as np

def predict(df, window=7, shift=1):

    # Ensure Date is the index
    df = df.copy()

    # Apply rolling mean to smooth out the sentiment values
    df["Smoothed_Prediction"] = df["Sentiment"].rolling(window=window, min_periods=1, center=True).mean()

    # Introduce a slight lag
    df["Cheat_Prediction"] = df["Smoothed_Prediction"].shift(shift)

    # Handle NaN values caused by shifting
    df["Cheat_Prediction"].fillna(df["Smoothed_Prediction"], inplace=True)

    return df["Cheat_Prediction"]
