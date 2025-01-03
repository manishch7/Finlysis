import os
from dotenv import load_dotenv
import pandas as pd
import requests
import re
from datetime import datetime, timedelta
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

load_dotenv()

FMP_API_KEY = os.getenv("FMP_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def convert_column_names(df):
    """Convert column names from camelCase to Title Case."""
    df.columns = [' '.join([word.capitalize() for word in re.findall(r'[a-z]+|[A-Z][a-z]*', col)]) for col in df.columns]
    return df

def get_company_profile(stock_tickers):
    """Fetch company profile data for given stock tickers."""
    company_profiles = []

    for ticker in stock_tickers:
        url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={FMP_API_KEY}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data:
                    for record in data:
                        company_profiles.append({
                            "symbol": record.get("symbol", ""),
                            "companyName": record.get("companyName", ""),
                            "mktCap": record.get("mktCap", 0),
                            "industry": record.get("industry", ""),
                            "sector": record.get("sector", ""),
                            "ceo": record.get("ceo", "")
                        })
            else:
                print(f"Error fetching company profile for {ticker}: {response.status_code}")
        except Exception as e:
            print(f"Unexpected error fetching profile for {ticker}: {e}")

    df_profiles = pd.DataFrame(company_profiles)
    if not df_profiles.empty:
        df_profiles = convert_column_names(df_profiles)
    return df_profiles

def get_stock_data(stock_tickers):
    """Fetch historical stock data for given tickers."""
    stock_data = []
    start_date = "2023-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    for ticker in stock_tickers:
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={start_date}&to={end_date}&apikey={FMP_API_KEY}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if "historical" in data:
                    for record in data["historical"]:
                        stock_data.append({
                            "Date": record["date"],
                            "Ticker": ticker,
                            "Open": record["open"],
                            "High": record["high"],
                            "Low": record["low"],
                            "Close": record["adjClose"],
                            "Volume": record["volume"],
                            "Change": record.get("change", 0),
                            "ChangePercent": record.get("changePercent", 0)
                        })
            else:
                print(f"Error fetching stock data for {ticker}: {response.status_code}")
        except Exception as e:
            print(f"Unexpected error fetching stock data for {ticker}: {e}")

    df_stock = pd.DataFrame(stock_data)
    if not df_stock.empty:
        df_stock = convert_column_names(df_stock)
    return df_stock

def get_financial_data(stock_tickers, limit =5):
    """Fetch financial statement data (income, balance, cash flow) for given tickers."""
    financial_statements = {
        "income_statement": [],
        "balance_sheet": [],
        "cash_flow": []
    }

    for ticker in stock_tickers:
        for statement_type, data_list in zip(
            ["income-statement", "balance-sheet-statement", "cash-flow-statement"],
            [financial_statements["income_statement"], financial_statements["balance_sheet"], financial_statements["cash_flow"]]
        ):
            url = f"https://financialmodelingprep.com/api/v3/{statement_type}/{ticker}?limit={limit}&apikey={FMP_API_KEY}"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and data:
                        for record in data:
                            record["Ticker"] = ticker
                            data_list.append(record)
                    else:
                        print(f"No data returned for {statement_type} of {ticker}.")
                else:
                    print(f"Error fetching {statement_type} for {ticker}: {response.status_code}")
            except Exception as e:
                print(f"Unexpected error fetching {statement_type} for {ticker}: {e}")

    return {
        name: convert_column_names(pd.DataFrame(data)) if data else pd.DataFrame()
        for name, data in financial_statements.items()
    }

def get_news_data(stock_tickers):
    """Fetch stock-related news and perform sentiment analysis."""
    news_data = []
    today = datetime.now()
    start_date = today - timedelta(days=30)
    print(
        f"Fetching news for {len(stock_tickers)} tickers from {start_date.strftime('%Y-%m-%d')} to {today.strftime('%Y-%m-%d')}..."
    )

    for ticker in stock_tickers:
        url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={ticker}&limit=300&apikey={FMP_API_KEY}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                articles = response.json()
                for article in articles:
                    published_date = datetime.strptime(article["publishedDate"], "%Y-%m-%d %H:%M:%S")
                    if start_date <= published_date <= today:
                        news_data.append({
                            "Ticker": ticker,
                            "Headline": article.get("title", ""),
                            "Date of News": article.get("publishedDate", ""),
                            "Source": article.get("site", ""),
                            "URL": article.get("url", "")
                        })
            else:
                print(f"Error fetching news for {ticker}: {response.status_code}")
        except Exception as e:
            print(f"Unexpected error fetching news for {ticker}: {e}")

    df_news = pd.DataFrame(news_data)
    if not df_news.empty:
        # Load the sentiment analysis model and tokenizer
        model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Create a pipeline for sentiment analysis (running on CPU by default)
        sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

        # Apply sentiment analysis to the 'Headline' column
        tqdm.pandas()  # Enables progress bar for pandas apply
        df_news[['Sentiment', 'Sentiment_Score']] = df_news['Headline'].progress_apply(
            lambda x: pd.Series(sentiment_pipeline(x, truncation=True, padding='max_length', max_length=512)[0])
        )

        # Select relevant columns for final output
        df_news = df_news[['Ticker', 'Headline', 'Source', 'Date of News', 'URL', 'Sentiment', 'Sentiment_Score']]
    return df_news

