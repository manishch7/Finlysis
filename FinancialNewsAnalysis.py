# Answers questions related to Financial News

from neo4j import GraphDatabase
import openai
import json
import os
from dotenv import load_dotenv

load_dotenv()

OPEN_API_KEY = os.getenv("OPEN_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Set OpenAI API Key
openai.api_key = OPEN_API_KEY

# Neo4j connection setup
driver = None
try:
    driver = GraphDatabase.driver(uri=NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Your existing code her

finally:
    if driver:
        driver.close()

# Function to extract ticker(s) and determine question type
def analyze_user_question(question):
    # Determine question type
    financial_keywords = ["income", "balance sheet", "cash flow", "profit", "revenue", "debt", "expenses", "valuation","finance","financial"]
    news_keywords = ["news", "sentiment", "lawsuit", "regulation", "competition", "product", "earnings"]

    question_type = None
    if any(keyword in question.lower() for keyword in financial_keywords):
        question_type = "financial_statements"
    elif any(keyword in question.lower() for keyword in news_keywords):
        question_type = "news"

    # Extract ticker(s) using GPT
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
    tickers = [ticker.strip() for ticker in extracted_tickers.split(",") if ticker.strip()]

    return question_type, tickers


# Function to fetch news data from Neo4j
def fetch_news_data(ticker):
    with driver.session() as session:
        result = session.run('''
         MATCH (stock:Stock_Ticker {ticker: $ticker})
        OPTIONAL MATCH (stock)-[:HAS_DAILY_PRICE]->(price:DailyPrice)
        OPTIONAL MATCH (price)-[:HAS_NEWS]->(news:News)
        OPTIONAL MATCH (stock)-[:HAS_NO_DAILY_PRICE]->(dateNode:Date)-[:HAS_NEWS]->(newsWithoutPrice:News)
        WITH DISTINCT news, newsWithoutPrice
        ORDER BY news.date_of_news DESC, newsWithoutPrice.date_of_news DESC
        RETURN 
            COLLECT(news {.*, sentiment: news.sentiment, sentiment_score: news.sentiment_score }) AS newsListWithPrice,
            COLLECT(newsWithoutPrice {.*, sentiment: newsWithoutPrice.sentiment, sentiment_score: newsWithoutPrice.sentiment_score }) AS newsListWithoutPrice
        ''', {"ticker": ticker})
        data = []
        for record in result:
            data.append(record.data())

        # Flatten and deduplicate news
        all_news = {frozenset(news.items()): news for news_list in data for news in
                    (news_list.get('newsListWithPrice', []) + news_list.get('newsListWithoutPrice', []))}
        deduplicated_news = list(all_news.values())
        return deduplicated_news


# Function to analyze news data using GPT
def analyze_news_data_with_llm(question, ticker, news_data):
    news_data_json = json.dumps(news_data, default=str)
    messages = [
        {"role": "system", "content": "You are a financial assistant analyzing news data. The user has asked a question, "
                                      "and you need to analyze the provided data to generate an answer for the given ticker."},
        {"role": "user", "content": f"Here is the news data:\n{news_data_json}\n\n"
                                    f"Question: {question}\n and Ticker: {ticker}\n"
                                    "Please analyze the data and provide a clear and concise answer without mentioning missing data."}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content.strip()

# Main function
def financial_news_analysis_main(user_question):
    try:
        # Analyze user question and extract relevant tickers
        question_type, tickers = analyze_user_question(user_question)
        print(f"Extracted tickers: {tickers}")

        if not tickers:
            print("No stock tickers were detected in your question. Please try again.")
            return

        all_answers = []

        for ticker in tickers:

            if question_type == "news":
                news_data = fetch_news_data(ticker)
                if news_data:
                    answer = analyze_news_data_with_llm(user_question, ticker, news_data)
                    print(f"\nAnswer for {ticker}:\n{answer}")
                    all_answers.append(answer)
                else:
                    all_answers.append(f"No news data found for ticker: {ticker}")

            else:
                all_answers.append("Unable to determine the type of question. Please refine your query.")
        return all_answers
    except Exception as e:
        print(f"An error occurred: {e}")
    return