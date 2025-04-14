import pandas as pd
import requests
import time
from tqdm import tqdm

#import data_preprocessing
#import cloud_run
from Sentiment_Analysis import data_preprocessing
from Sentiment_Analysis import cloud_run

def get_sentiment(titles):
    payload = [
            {'texts': titles}
        ]

    try:
        response = cloud_run.predict_custom_trained_model_sample(project = "bert-vertex-ai", endpoint_id= "3877006642398625792", instances=payload)
        return response
    except Exception as e:
        print(f"Error during prediction: {e}")
        return ["Error"] * len(titles)

def run():
    data_preprocessing.run()

    # Load dataset
    df = pd.read_csv("/home/captain/Desktop/NLP_FISAC/Backend/Sentiment_Analysis/reddit_posts_cleaned.csv")

    # Ensure 'Post_Date' is in datetime format
    df['Post_Date'] = pd.to_datetime(df['Post_Date'])

    batch_size = 200
    sentiment_results = []

    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i:i+batch_size]
        batch_titles = batch_df['Title'].tolist()

        if not batch_titles:
            continue
        
        try:
            batch_sentiments = get_sentiment(batch_titles)
            for j, sentiment in enumerate(batch_sentiments):
                sentiment_results.append([df.iloc[i+j]['Post_Date'], df.iloc[i+j]['Title'], sentiment])
        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {e}")
            time.sleep(5)

    sentiment_df = pd.DataFrame(sentiment_results, columns=['Date', 'Title', 'Sentiment'])
    sentiment_df.to_csv("/home/captain/Desktop/NLP_FISAC/Backend/Sentiment_Analysis/sentiment.csv", index=False)

    print("Sentiment analysis completed and saved to sentiment.csv")

if __name__ == '__main__':
    run()

    #print(get_sentiment(["i love this", 'i hate this']))
