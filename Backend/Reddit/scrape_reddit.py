import praw
import pandas as pd
import time
import re
import os

def run():
    dir = '/home/captain/Desktop/NLP_FISAC/Backend/Reddit'

    # 🔹 Setup Reddit API credentials
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

    # 🔹 Load keywords from keywords.txt
    with open("/home/captain/Desktop/NLP_FISAC/Backend/Hashtag_Generation/keywords.txt", "r") as file:
        keywords = [re.sub(r"^\d+\.\s*", "", line.strip()) for line in file.readlines()]

    # 🔹 Storage for results
    all_posts = []

    # 🔹 Fetch posts for each keyword
    for count, keyword in enumerate(keywords):
        if count < 5:
            print(f"🔍 Fetching posts for '{keyword}'...")

            try:
                for post in reddit.subreddit("all").search(keyword, time_filter="all", limit=1000):
                    all_posts.append([
                        time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(post.created_utc)),  # Convert timestamp
                        post.title,
                        post.selftext,
                        post.subreddit.display_name
                    ])
                
            except Exception as e:
                print(f"⚠️ Error fetching data for '{keyword}': {e}")

    # 🔹 Save to CSV
    df = pd.DataFrame(all_posts, columns=["Date", "Title", "Content", "Subreddit"])
    print(df.shape)
    df.to_csv(os.path.join(dir,"reddit_posts.csv"), index=False)

    print("✅ Data collection complete. Saved to reddit_posts.csv.")


if __name__ == "__main__":
    run()