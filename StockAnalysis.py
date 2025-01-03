# Answers questions related to stock data like the highest/least price of a particular stock ticker
# Also answers for prediction questions using LSTM

from neo4j import GraphDatabase
import pandas as pd
import numpy as np
import openai
import json
import neo4j.time
from keras import models, layers
from sklearn.preprocessing import MinMaxScaler

import os
from dotenv import load_dotenv

load_dotenv()

FMP_API_KEY = os.getenv("FMP_API_KEY")
OPEN_API_KEY = os.getenv("OPEN_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Set OpenAI API Key
openai.api_key = OPEN_API_KEY
# Neo4j connection setup
driver = GraphDatabase.driver(uri=NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Function to extract ticker(s) from user question
def extract_ticker_with_llm(question):
    messages = [
        {"role": "system", "content": "You are a financial assistant that extracts stock tickers from user questions. "
                                      "Return only the stock tickers as a comma-separated list."},
        {"role": "user", "content": f"Extract the stock tickers from this question: {question}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages
    )
    extracted_tickers = response.choices[0].message.content.strip()
    return [ticker.strip() for ticker in extracted_tickers.split(",") if ticker.strip()]


# Function to fetch stock data from Neo4j
def fetch_stock_data_from_neo4j(ticker):
    with driver.session() as session:
        result = session.run('''
        MATCH (s:Stock_Ticker {ticker: $ticker})-[:HAS_DAILY_PRICE]->(p:DailyPrice)
        RETURN 
            s.ticker AS Ticker, 
            p.date AS Date, 
            p.open AS Open, 
            p.high AS High, 
            p.low AS Low, 
            p.close AS Close, 
            p.volume AS Volume
        ORDER BY p.date
    ''', {"ticker": ticker})
        data = []
        for record in result:
            record_dict = record.data()
            if "Date" in record_dict and isinstance(record_dict["Date"], neo4j.time.DateTime):
                record_dict["Date"] = record_dict["Date"].to_native()
            data.append(record_dict)
        return pd.DataFrame(data)


# Function to analyze stock data using GPT-4o Mini
def analyze_stock_data_with_llm(question, ticker, stock_data):
    # Convert stock data to JSON
    stock_data_json = json.dumps(stock_data, default=str)

    # Use GPT-4o Mini for analysis
    messages = [
        {"role": "system",
         "content": "You are a financial assistant analyzing stock data. The user has asked a question, "
                    "and you need to analyze the provided data to generate an answer for only the given ticker. Don't summarize"},
        {"role": "user", "content": f"Here is the stock data:\n{stock_data_json}\n\n"
                                    f"Question: {question}\n and Ticker: {ticker}\n"
                                    f"Please analyze the data and provide a clear and concise answer for the ticker {ticker}."}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content.strip()


# Function to check if the user is asking for price prediction and what feature
def is_price_prediction_question(question):
    # Keywords indicating a prediction intent
    prediction_keywords = ["predict", "forecast", "estimate", "future", "will", "project"]

    # Keywords mapping to specific features
    feature_keywords = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "change": " Change ",
        "change_percent": "Change Percent"

    }

    # Check if the question has prediction intent
    has_prediction_intent = any(keyword in question.lower() for keyword in prediction_keywords)

    # Check if the question refers to a specific feature
    for keyword, feature in feature_keywords.items():
        if keyword in question.lower() and has_prediction_intent:
            return feature
    return None


# Function to predict tomorrow's feature value using LSTM
def predict_tomorrows_value_lstm(stock_data, feature):
    if stock_data.empty or len(stock_data) < 100:
        return f"Insufficient data for LSTM prediction for {feature}."

    # Prepare data
    feature_data = stock_data[feature].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(feature_data)
    lookback = 60
    x, y = [], []
    for i in range(lookback, len(scaled_data)):
        x.append(scaled_data[i - lookback:i, 0])
        y.append(scaled_data[i, 0])

    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    # Build the LSTM model
    model = models.Sequential()
    model.add(layers.LSTM(units=50, return_sequences=True, input_shape=(x.shape[1], 1)))
    model.add(layers.LSTM(units=50, return_sequences=False))
    model.add(layers.Dense(units=25))
    model.add(layers.Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x, y, batch_size=32, epochs=10, verbose=0)

    # Predict the next day's feature value
    last_60_days = scaled_data[-lookback:]
    x_test = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))
    predicted_value = model.predict(x_test)
    predicted_value = scaler.inverse_transform(predicted_value)

    return f"Predicted {feature} for tomorrow: {predicted_value[0, 0]:.2f}"


# Main function
def stock_analysis_main(user_question):
    try:
        # Extract ticker(s) from the question
        tickers = extract_ticker_with_llm(user_question)
        print(f"Extracted tickers: {tickers}")

        if not tickers:
            print("No stock tickers were detected in your question. Please try again.")
            return
        all_answers = []
        for ticker in tickers:
            # Fetch stock data for the ticker
            stock_data = fetch_stock_data_from_neo4j(ticker)

            if stock_data.empty:
                print(f"No data found for ticker: {ticker}")
                continue

            # Check if the question is for prediction and for which feature
            feature = is_price_prediction_question(user_question)

            if feature:
                prediction = predict_tomorrows_value_lstm(stock_data, feature)
                print(f"\n{prediction}")
                all_answers.append(prediction)
            else:
                # Use GPT-4o Mini to analyze the data and answer the question
                answer = analyze_stock_data_with_llm(user_question, ticker, stock_data)
                print(f"\nAnswer for {ticker}:\n{answer}")
                all_answers.append(answer)
        return all_answers
    except Exception as e:
        print(f"An error occurred: {e}")
    return