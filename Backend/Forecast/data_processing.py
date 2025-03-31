import pandas as pd
import os

def run():
    dir = '/home/captain/Desktop/NLP_FISAC/Backend/Forecast'

    # Load the CSV file
    df = pd.read_csv("/home/captain/Desktop/NLP_FISAC/Backend/Sentiment_Analysis/sentiment.csv")

    # Mapping sentiment values
    sentiment_mapping = {
        "Positive": 1,
        "Negative": -1,
        "Neutral": 0
    }

    # Apply the mapping
    df["Sentiment"] = df["Sentiment"].map(sentiment_mapping)

    # Group by date and sum the sentiment values
    df_grouped = df.groupby("Date")["Sentiment"].mean().reset_index()

    # Sort by date from earliest to latest
    df_grouped = df_grouped.sort_values(by="Date")

    # Save the processed data
    df_grouped.to_csv(os.path.join(dir,"processed_sentiment.csv"), index=False)

    print("✅ Processed sentiment data saved as 'processed_sentiment.csv'")
