import pandas as pd
import matplotlib.pyplot as plt
import os

dir = 'NLP_FISAC/Backend/Forecast'

# Load the processed sentiment data
df = pd.read_csv("NLP_FISAC/Backend/Forecast/processed_sentiment.csv")

# Convert Date column to datetime for proper plotting
df["Date"] = pd.to_datetime(df["Date"])

# Plot the sentiment trend over time
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Sentiment"], linestyle="-", color="b", label="Sentiment Score")

# Customize the plot
plt.xlabel("Date")
plt.ylabel("Sentiment Score")
plt.title("Sentiment Trend Over Time")
plt.xticks(rotation=45)
plt.axhline(0, color="gray", linestyle="--", linewidth=1)  # Reference line at 0
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig(os.path.join(dir,"sentiment_trend.png"), dpi=300, bbox_inches="tight")

# Show the plot
plt.show()
