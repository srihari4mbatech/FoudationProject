import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
 
# Load the pre-trained model and scaler
loaded_model = load_model("stock_price_prediction_model.h5")
scaler = MinMaxScaler()  # Assuming you have saved the scaler during training
 
def preprocess_input_data(open_val, high_val, low_val, volume_val, close_val,
                          headline, content):
    # Assume you have a function to process the news data and extract sentiment scores
    positive_sentiment_val, negative_sentiment_val, neutral_sentiment_val = process_news_data(headline, content)
 
    # Reshape input data
    input_data = np.array([[open_val, high_val, low_val, volume_val, close_val,
                            positive_sentiment_val, negative_sentiment_val, neutral_sentiment_val,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
 
    # Scale the input data
    scaled_input_data = scaler.transform(input_data)
 
    return scaled_input_data
 
def process_news_data(headline, content):
    # Placeholder function, replace with actual sentiment analysis logic
    positive_sentiment_val = 0.5
    negative_sentiment_val = 0.3
    neutral_sentiment_val = 0.2
    return positive_sentiment_val, negative_sentiment_val, neutral_sentiment_val
 
def main():
    st.title("Stock Price Prediction App")
 
    # Input form for user input
    open_val = st.number_input("Enter Open Value:")
    high_val = st.number_input("Enter High Value:")
    low_val = st.number_input("Enter Low Value:")
    volume_val = st.number_input("Enter Volume Value:")
    close_val = st.number_input("Enter Close Value:")
 
    headline = st.text_input("Enter News Article Headline:")
    content = st.text_area("Enter News Article Content:")
 
    # Button to generate prediction
    if st.button("Generate Prediction"):
        # Preprocess input data
        input_data = preprocess_input_data(open_val, high_val, low_val, volume_val, close_val,
                                           headline, content)
 
        # Make prediction using the loaded model
        prediction = loaded_model.predict(input_data)
 
        # Display the prediction
        st.success(f"Predicted Next_Week_Close: {prediction.flatten()[0]}")
 
if __name__ == "__main__":
    main()