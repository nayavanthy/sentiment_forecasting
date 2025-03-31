import pandas as pd
import openai
import time
from tqdm import tqdm
import re

from Sentiment_Analysis import data_preprocessing

# OpenAI API Key (Replace with your key)
OPENAI_API_KEY = ""

# Function to get sentiment from OpenAI
def get_sentiment(titles):
    prompt = "Classify the following Reddit post titles into 'Positive', 'Neutral', or 'Negative':\n\n"
    prompt += "\n".join([f"{i+1}. {title}" for i, title in enumerate(titles)])
    chat_client = openai.OpenAI(api_key=OPENAI_API_KEY)

    response = chat_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a sentiment analysis model."},
                  {"role": "user", "content": prompt}],
        temperature=0
    )
    
    # Extract sentiment labels
    sentiments = response.choices[0].message.content.strip().split("\n")  # Split into a list of lines
    sentiments = [re.sub(r"^\d+\.\s*", "", line) for line in sentiments]

    return sentiments

def run():

    data_preprocessing.run()

    # Load dataset
    df = pd.read_csv("NLP_FISAC/Backend/Sentiment_Analysis/reddit_posts_cleaned.csv")

    # Ensure 'Post_Date' is in datetime format
    df['Post_Date'] = pd.to_datetime(df['Post_Date'])

    # Process in batches
    batch_size = 100
    sentiment_results = []

    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i:i+batch_size]  # Select valid batch
        batch_titles = batch_df['Title'].tolist()

        if not batch_titles:  # Skip empty batch
            continue
        
        try:
            batch_sentiments = get_sentiment(batch_titles)
            for j, sentiment in enumerate(batch_sentiments):
                sentiment_results.append([df.iloc[i+j]['Post_Date'], df.iloc[i+j]['Title'], sentiment])
        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {e}")
            time.sleep(5)  # Wait before retrying

    # Convert results to DataFrame
    sentiment_df = pd.DataFrame(sentiment_results, columns=['Date', 'Title', 'Sentiment'])

    # Save to CSV
    sentiment_df.to_csv("NLP_FISAC/Backend/Sentiment_Analysis/sentiment.csv", index=False)

    print("Sentiment analysis completed and saved to sentiment.csv")

if __name__ == '__main__':
    run()
