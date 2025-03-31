import pandas as pd

def run():
    # Load the dataset
    df = pd.read_csv("NLP_FISAC/Backend/Reddit/reddit_posts.csv")

    # 🔹 Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # 🔹 Create separate 'Date' and 'Time' columns
    df['Post_Date'] = df['Date'].dt.date
    df['Post_Time'] = df['Date'].dt.time

    # 🔹 Drop the original 'Date' column
    df.drop(columns=['Date'], inplace=True)

    # 🔹 Print number of null values per column
    print("\nNumber of Null Values:")
    print(df.isnull().sum())

    # 🔹 Print the number of unique dates
    print("\nNumber of Unique Dates:", df['Post_Date'].nunique())

    # 🔹 Save cleaned data
    df.to_csv("NLP_FISAC/Backend/Sentiment_Analysis/reddit_posts_cleaned.csv", index=False)

    print("\nPreprocessing complete. Cleaned data saved as 'reddit_posts_cleaned.csv'.")
