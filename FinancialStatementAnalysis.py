from neo4j import GraphDatabase
import openai
import json
import os
import neo4j.time

from dotenv import load_dotenv

load_dotenv()

# Load API keys and Neo4j credentials from environment variables
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


    # Function to fetch data from Neo4j
    def fetch_data(query, ticker):
        with driver.session() as session:
            result = session.run(query, {"ticker": ticker})
            data = []
            for record in result:
                record_dict = record.data()
                # Handle Neo4j's time.Date objects and convert to Python-native datetime.date
                for key in ["CurrentDate", "NextDate"]:
                    if key in record_dict and isinstance(record_dict[key], neo4j.time.Date):
                        record_dict[key] = record_dict[key].to_native()
                data.append(record_dict)
            return data


    # Function to extract ticker(s) from the user question
    def extract_ticker_with_llm(question):
        messages = [
            {"role": "system",
             "content": "You are a financial assistant that extracts stock tickers from user questions. "
                        "Return only the stock tickers as a comma-separated list."},
            {"role": "user", "content": f"Extract the stock tickers from this question: {question}"}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages
        )
        extracted_tickers = response.choices[0].message.content.strip()
        return [ticker.strip() for ticker in extracted_tickers.split(",") if ticker.strip()]


    # Function to fetch financial data from Neo4j
    def fetch_financial_data_from_neo4j(ticker):
        query = '''
        MATCH (s:Stock_Ticker {ticker: $ticker})-[:HAS_FINANCIAL_STATEMENT_FOR_YEAR]->(y:Year)
        OPTIONAL MATCH (y)-[:HAS_INCOME_STATEMENT]->(income:Income_Statement)
        OPTIONAL MATCH (y)-[:HAS_BALANCE_SHEET]->(balance:Balance_Sheet)
        OPTIONAL MATCH (y)-[:HAS_CASH_FLOW_STATEMENT]->(cashflow:Cash_Flow)
        RETURN 
            y.year AS Year,
            income AS IncomeStatement,
            balance AS BalanceSheet,
            cashflow AS CashFlowStatement
        ORDER BY y.year DESC
        '''
        return fetch_data(query, ticker)


    # Function to chunk the data into smaller parts
    def chunk_data(data, max_chunk_size=16000):
        chunks = []
        current_chunk = []
        current_length = 0

        for record in data:
            record_str = json.dumps(record, default=str)
            record_length = len(record_str)

            if current_length + record_length > max_chunk_size:
                chunks.append(current_chunk)
                current_chunk = []
                current_length = 0

            current_chunk.append(record)
            current_length += record_length

        if current_chunk:
            chunks.append(current_chunk)

        return chunks


    # Function to analyze financial data using GPT-4o Mini
    def analyze_financial_data_with_llm(question, ticker, financial_data):
        financial_data_json = json.dumps(financial_data, default=str)

        # Use GPT-4o Mini for analysis
        messages = [
            {"role": "system", "content": """ 
                                             You are a financial analyst tasked with evaluating a company’s financial statements. Based on the user’s input:

                                            General Overview (Always Provide):
                                            Summarize the company’s financial health using insights from the income statement, balance sheet, cash flow statement, and notes to financial statements.
                                            Highlight both strengths and concerns, ensuring a balanced assessment.
                                            Green Flags (Only if requested):
                                            Income Statement Strengths: Mention stable or increasing revenue trends, controlled expenses resulting in strong profit margins, and low tax or interest burdens relative to income.
                                            Balance Sheet Strengths: Highlight strong liquidity ratios, moderate leverage, a diversified asset base, and minimal impairments or write-offs.
                                            Cash Flow Strengths: Identify positive and sustainable operating cash flows, sufficient capital expenditures, and minimal reliance on debt or equity for funding.
                                            Notes and Contextual Insights: Emphasize conservative accounting practices, low contingent liabilities, and positive audit remarks with no going concern issues.
                                            Red Flags (Only if requested):
                                            Income Statement Concerns: Identify declining or inconsistent revenue growth, unexplained increases in SG&A or one-time charges, and shrinking profit margins.
                                            Balance Sheet Concerns: Point out liquidity issues, high debt-to-equity ratios, reliance on intangible assets, or frequent write-offs.
                                            Cash Flow Concerns: Highlight negative operating cash flows despite reported profits, underinvestment in CapEx, and over-reliance on external financing.
                                            Notes and Contextual Risks: Address aggressive accounting practices, high contingent liabilities, related-party transactions, or auditor warnings about going concern issues.   

                                            """},

            {"role": "user", "content": f"Here is the financial data:\n{financial_data_json}\n\n"
                                        f"Question: {question}\nTicker: {ticker}\n"
                                        "Please analyze the data and provide a justified clear and concise answer for the ticker."}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content.strip()


    # Main function
    def financial_statements_analysis_main(user_question):
        try:
            # Extract ticker(s) from the question
            tickers = extract_ticker_with_llm(user_question)
            print(f"Extracted tickers: {tickers}")

            if not tickers:
                print("No stock tickers were detected in your question. Please try again.")
                return "No stock tickers were detected in your question. Please try again."

            all_answers = []
            for ticker in tickers:
                # Fetch financial data for the ticker
                financial_data = fetch_financial_data_from_neo4j(ticker)

                if not financial_data:
                    print(f"No financial data found for ticker: {ticker}")
                    all_answers.append(f"No financial data found for ticker: {ticker}")
                    continue

                # Use GPT-4o Mini to analyze the data and answer the question
                answer = analyze_financial_data_with_llm(user_question, ticker, financial_data)
                print(f"\nAnswer for {ticker}:\n{answer}")
                all_answers.append(answer)

            return all_answers
        except Exception as e:
            print(f"An error occurred: {e}")
            return f"An error occurred while processing your request: {e}"

finally:
    if driver:
        driver.close()