import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("/home/captain/Desktop/NLP_FISAC/Backend/Sentiment_Analysis/reddit_posts_cleaned.csv")

# Convert 'Post_Date' to datetime format
df['Post_Date'] = pd.to_datetime(df['Post_Date'])

# Plot the distribution of posts over time
plt.figure(figsize=(12, 6))
df['Post_Date'].hist(bins=50, edgecolor='black')  # Adjust bins for granularity

# Formatting the plot
plt.xlabel("Date")
plt.ylabel("Number of Posts")
plt.title("Distribution of Reddit Posts Over Time")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()
