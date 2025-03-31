import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

# Define Allowed Sentiment Values
ALLOWED_VALUES = np.array([-1, -0.5, -0.33, -0.25, 0, 0.25, 0.33, 0.5, 1])

# 🔹 Function to map values to closest allowed sentiment
def map_to_allowed_values(sentiments):
    return np.array([ALLOWED_VALUES[np.argmin(np.abs(ALLOWED_VALUES - s))] for s in sentiments])

dir = 'NLP_FISAC/Backend/Forecast'

# 🔹 Load dataset
df = pd.read_csv(os.path.join(dir,"processed_sentiment.csv"))
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")  # Ensure chronological order

# 🔹 Restrict training data (2018 - 2023) and testing data (2024)
train_df = df[(df["Date"] >= "2018-01-01") & (df["Date"] < "2024-01-01")]
test_df = df[df["Date"] >= "2024-01-01"]

# 🔹 Apply sentiment mapping to allowed values
train_df["Sentiment"] = map_to_allowed_values(train_df["Sentiment"])
test_df["Sentiment"] = map_to_allowed_values(test_df["Sentiment"])

# 🔹 Convert sentiment values to class labels
label_encoder = LabelEncoder()
train_df["Sentiment_Class"] = label_encoder.fit_transform(train_df["Sentiment"])
test_df["Sentiment_Class"] = label_encoder.transform(test_df["Sentiment"])

# 🔹 Normalize data (for LSTM stability)
scaler = MinMaxScaler(feature_range=(-1, 1))
train_scaled = scaler.fit_transform(train_df["Sentiment"].values.reshape(-1, 1))
test_scaled = scaler.transform(test_df["Sentiment"].values.reshape(-1, 1))

# 🔹 Function to create sequences for LSTM
def create_sequences(data, labels, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(labels[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 1  # Number of past days to consider

# 🔹 Create LSTM sequences
X_train, y_train = create_sequences(train_scaled, train_df["Sentiment_Class"].values, SEQ_LENGTH)
X_test, y_test = create_sequences(test_scaled, test_df["Sentiment_Class"].values, SEQ_LENGTH)

# ✅ Build LSTM Model
model = keras.Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(len(ALLOWED_VALUES), activation="softmax")  # Output size = number of sentiment classes
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# 🔹 Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_test, y_test))

# 🔹 Predict on test data
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)  # Convert to class indices
predicted_sentiments = label_encoder.inverse_transform(predicted_classes)  # Convert back to sentiment values

# 🔹 Prepare test data for plotting
test_dates = test_df.iloc[SEQ_LENGTH:]["Date"]

# 🔹 Plot results
plt.figure(figsize=(12, 6))
plt.plot(test_dates, test_df.iloc[SEQ_LENGTH:]["Sentiment"], label="Actual Sentiment", color="blue")
plt.plot(test_dates, predicted_sentiments, label="Predicted Sentiment", color="red", linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Sentiment Score")
plt.title("LSTM Sentiment Prediction")
plt.legend()
plt.grid(True)
plt.savefig("lstm_sentiment_prediction.png", dpi=300)
plt.show()
