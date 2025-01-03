from neo4j import GraphDatabase
import pandas as pd
import FetchData
from dotenv import load_dotenv
import os

load_dotenv()

FMP_API_KEY = os.getenv("FMP_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


# Initialize Neo4j Driver
driver = GraphDatabase.driver(uri=NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def execute_query(query, parameters=None):
    with driver.session() as session:
        session.run(query, parameters or {})

def load_company_profiles(company_profiles):
    try:
        for _, row in company_profiles.iterrows():
            execute_query('''
            MERGE (stock:Stock_Ticker {ticker: $symbol})
            SET stock.company_name = $company_name,
                stock.mkt_cap = $mkt_cap,
                stock.industry = $industry,
                stock.sector = $sector,
                stock.description = $description,
                stock.ceo = $ceo
            ''', {
                "symbol": row['Symbol'],
                "company_name": row['Company Name'],
                "mkt_cap": row['Mkt Cap'],
                "industry": row['Industry'],
                "sector": row['Sector'],
                "description": row['Description'],
                "ceo": row['Ceo']
            })
        print("Company profiles loaded successfully.")
    except Exception as e:
        print("Error loading company profiles:", e)

def load_stock_and_prices(stock_data):
    try:
        stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.strftime('%Y-%m-%d')

        for ticker, ticker_data in stock_data.groupby('Ticker'):
            daily_prices = [
                {
                    "date": row['Date'],
                    "open": row['Open'],
                    "high": row['High'],
                    "low": row['Low'],
                    "close": row['Close'],
                    "volume": row['Volume'],
                    "change": row['Change'],
                    "change_percent": row['Change Percent']
                }
                for _, row in ticker_data.iterrows()
            ]

            execute_query('''
            MERGE (stock:Stock_Ticker {ticker: $ticker})
            WITH stock
            UNWIND $daily_prices AS dailyPrice
            MERGE (price:DailyPrice {date: date(dailyPrice.date), ticker: $ticker})
            SET price.open = dailyPrice.open,
                price.high = dailyPrice.high,
                price.low = dailyPrice.low,
                price.close = dailyPrice.close,
                price.volume = dailyPrice.volume,
                price.change = dailyPrice.change,
                price.change_percent = dailyPrice.change_percent
            MERGE (stock)-[:HAS_DAILY_PRICE]->(price)
            ''', {"ticker": ticker, "daily_prices": daily_prices})

            execute_query('''
            MATCH (stock:Stock_Ticker {ticker: $ticker})-[:HAS_DAILY_PRICE]->(price:DailyPrice)
            WITH price
            ORDER BY price.date
            WITH collect(price) AS prices
            UNWIND RANGE(0, size(prices) - 2) AS i
            WITH prices[i] AS current, prices[i + 1] AS next
            MERGE (current)-[:NEXT]->(next)
            ''', {"ticker": ticker})
        print("Stock and daily prices loaded successfully.")
    except Exception as e:
        print("Error loading stock and prices:", e)

def load_financial_data(income_statement, balance_sheet, cash_flow):
    try:
        tickers = income_statement['Symbol'].unique()

        for ticker in tickers:
            for year in income_statement.loc[income_statement['Symbol'] == ticker, 'Calendar Year'].unique():
                # Extract the income statement row for this ticker and year
                income = income_statement[
                    (income_statement['Symbol'] == ticker) & (income_statement['Calendar Year'] == year)
                ].iloc[0].to_dict()

                # Extract the balance sheet row
                balance = {}
                if not balance_sheet.empty:
                    balance_rows = balance_sheet[
                        (balance_sheet['Symbol'] == ticker) & (balance_sheet['Calendar Year'] == year)
                    ]
                    if not balance_rows.empty:
                        balance = balance_rows.iloc[0].to_dict()

                # Extract the cash flow row
                cashflow = {}
                if not cash_flow.empty:
                    cashflow_rows = cash_flow[
                        (cash_flow['Symbol'] == ticker) & (cash_flow['Calendar Year'] == year)
                    ]
                    if not cashflow_rows.empty:
                        cashflow = cashflow_rows.iloc[0].to_dict()

                # Prepare parameters for income statement
                income_params = {
                    "ticker": ticker,
                    "year": int(year),
                    "date": income.get("Date", None),
                    "symbol": income.get("Symbol", None),
                    "reported_currency": income.get("Reported Currency", None),
                    "cik": income.get("Cik", None),
                    "filling_date": income.get("Filling Date", None),
                    "accepted_date": income.get("Accepted Date", None),
                    "calendar_year": income.get("Calendar Year", None),
                    "period": income.get("Period", None),
                    "revenue": income.get("Revenue", 0),
                    "cost_of_revenue": income.get("Cost Of Revenue", 0),
                    "gross_profit": income.get("Gross Profit", 0),
                    "gross_profit_ratio": income.get("Gross Profit Ratio", 0.0),
                    "research_and_development_expenses": income.get("Research And Development Expenses", 0.0),
                    "general_and_administrative_expenses": income.get("General And Administrative Expenses", 0),
                    "selling_and_marketing_expenses": income.get("Selling And Marketing Expenses", 0),
                    "selling_general_and_administrative_expenses": income.get("Selling General And Administrative Expenses", 0),
                    "other_expenses": income.get("Other Expenses", 0),
                    "operating_expenses": income.get("Operating Expenses", 0),
                    "cost_and_expenses": income.get("Cost And Expenses", 0),
                    "interest_income": income.get("Interest Income", 0),
                    "interest_expense": income.get("Interest Expense", 0),
                    "depreciation_and_amortization": income.get("Depreciation And Amortization", 0),
                    "ebitda": income.get("Ebitda", 0),
                    "ebitdaratio": income.get("Ebitdaratio", 0.0),
                    "operating_income": income.get("Operating Income", 0),
                    "operating_income_ratio": income.get("Operating Income Ratio", 0.0),
                    "total_other_income_expenses_net": income.get("Total Other Income Expenses Net", 0),
                    "income_before_tax": income.get("Income Before Tax", 0),
                    "income_before_tax_ratio": income.get("Income Before Tax Ratio", 0.0),
                    "income_tax_expense": income.get("Income Tax Expense", 0),
                    "net_income": income.get("Net Income", 0),
                    "net_income_ratio": income.get("Net Income Ratio", 0.0),
                    "eps": income.get("Eps", 0.0),
                    "epsdiluted": income.get("Epsdiluted", 0.0),
                    "weighted_average_shs_out": income.get("Weighted Average Shs Out", 0),
                    "weighted_average_shs_out_dil": income.get("Weighted Average Shs Out Dil", 0),
                    "link": income.get("Link", None),
                    "final_link": income.get("Final Link", None)
                }

                # Prepare parameters for balance sheet
                balance_params = {
                    "date": balance.get("Date", None),
                    "symbol": balance.get("Symbol", None),
                    "reported_currency": balance.get("Reported Currency", None),
                    "cik": balance.get("Cik", None),
                    "filling_date": balance.get("Filling Date", None),
                    "accepted_date": balance.get("Accepted Date", None),
                    "calendar_year": balance.get("Calendar Year", None),
                    "period": balance.get("Period", None),
                    "cash_and_cash_equivalents": balance.get("Cash And Cash Equivalents", 0),
                    "short_term_investments": balance.get("Short Term Investments", 0),
                    "cash_and_short_term_investments": balance.get("Cash And Short Term Investments", 0),
                    "net_receivables": balance.get("Net Receivables", 0),
                    "inventory": balance.get("Inventory", 0),
                    "other_current_assets": balance.get("Other Current Assets", 0),
                    "total_current_assets": balance.get("Total Current Assets", 0),
                    "property_plant_equipment_net": balance.get("Property Plant Equipment Net", 0),
                    "goodwill": balance.get("Goodwill", 0),
                    "intangible_assets": balance.get("Intangible Assets", 0),
                    "goodwill_and_intangible_assets": balance.get("Goodwill And Intangible Assets", 0),
                    "long_term_investments": balance.get("Long Term Investments", 0),
                    "tax_assets": balance.get("Tax Assets", 0),
                    "other_non_current_assets": balance.get("Other Non Current Assets", 0),
                    "total_non_current_assets": balance.get("Total Non Current Assets", 0),
                    "other_assets": balance.get("Other Assets", 0),
                    "total_assets": balance.get("Total Assets", 0),
                    "account_payables": balance.get("Account Payables", 0),
                    "short_term_debt": balance.get("Short Term Debt", 0),
                    "tax_payables": balance.get("Tax Payables", 0),
                    "deferred_revenue": balance.get("Deferred Revenue", 0),
                    "other_current_liabilities": balance.get("Other Current Liabilities", 0),
                    "total_current_liabilities": balance.get("Total Current Liabilities", 0),
                    "long_term_debt": balance.get("Long Term Debt", 0),
                    "deferred_revenue_non_current": balance.get("Deferred Revenue Non Current", 0),
                    "deferred_tax_liabilities_non_current": balance.get("Deferred Tax Liabilities Non Current", 0),
                    "other_non_current_liabilities": balance.get("Other Non Current Liabilities", 0),
                    "total_non_current_liabilities": balance.get("Total Non Current Liabilities", 0),
                    "other_liabilities": balance.get("Other Liabilities", 0),
                    "capital_lease_obligations": balance.get("Capital Lease Obligations", 0),
                    "total_liabilities": balance.get("Total Liabilities", 0),
                    "preferred_stock": balance.get("Preferred Stock", 0),
                    "common_stock": balance.get("Common Stock", 0),
                    "retained_earnings": balance.get("Retained Earnings", 0),
                    "accumulated_other_comprehensive_income_loss": balance.get("Accumulated Other Comprehensive Income Loss", 0),
                    "othertotal_stockholders_equity": balance.get("Othertotal Stockholders Equity", 0),
                    "total_stockholders_equity": balance.get("Total Stockholders Equity", 0),
                    "total_equity": balance.get("Total Equity", 0),
                    "total_liabilities_and_stockholders_equity": balance.get("Total Liabilities And Stockholders Equity", 0),
                    "minority_interest": balance.get("Minority Interest", 0),
                    "total_liabilities_and_total_equity": balance.get("Total Liabilities And Total Equity", 0),
                    "total_investments": balance.get("Total Investments", 0),
                    "total_debt": balance.get("Total Debt", 0),
                    "net_debt": balance.get("Net Debt", 0),
                    "link": balance.get("Link", None),
                    "final_link": balance.get("Final Link", None)
                }

                # Prepare parameters for cash flow
                cashflow_params = {
                    "date": cashflow.get("Date", None),
                    "symbol": cashflow.get("Symbol", None),
                    "reported_currency": cashflow.get("Reported Currency", None),
                    "cik": cashflow.get("Cik", None),
                    "filling_date": cashflow.get("Filling Date", None),
                    "accepted_date": cashflow.get("Accepted Date", None),
                    "calendar_year": cashflow.get("Calendar Year", None),
                    "period": cashflow.get("Period", None),
                    "net_income": cashflow.get("Net Income", 0),
                    "depreciation_and_amortization": cashflow.get("Depreciation And Amortization", 0),
                    "deferred_income_tax": cashflow.get("Deferred Income Tax", 0),
                    "stock_based_compensation": cashflow.get("Stock Based Compensation", 0),
                    "change_in_working_capital": cashflow.get("Change In Working Capital", 0),
                    "accounts_receivables": cashflow.get("Accounts Receivables", 0),
                    "inventory": cashflow.get("Inventory", 0),
                    "accounts_payables": cashflow.get("Accounts Payables", 0),
                    "other_working_capital": cashflow.get("Other Working Capital", 0),
                    "other_non_cash_items": cashflow.get("Other Non Cash Items", 0),
                    "net_cash_provided_by_operating_activities": cashflow.get("Net Cash Provided By Operating Activities", 0),
                    "investments_in_property_plant_and_equipment": cashflow.get("Investments In Property Plant And Equipment", 0),
                    "acquisitions_net": cashflow.get("Acquisitions Net", 0),
                    "purchases_of_investments": cashflow.get("Purchases Of Investments", 0),
                    "sales_maturities_of_investments": cashflow.get("Sales Maturities Of Investments", 0),
                    "other_investing_activites": cashflow.get("Other Investing Activites", 0),
                    "net_cash_used_for_investing_activites": cashflow.get("Net Cash Used For Investing Activites", 0),
                    "debt_repayment": cashflow.get("Debt Repayment", 0),
                    "common_stock_issued": cashflow.get("Common Stock Issued", 0),
                    "common_stock_repurchased": cashflow.get("Common Stock Repurchased", 0),
                    "dividends_paid": cashflow.get("Dividends Paid", 0),
                    "other_financing_activites": cashflow.get("Other Financing Activites", 0),
                    "net_cash_used_provided_by_financing_activities": cashflow.get("Net Cash Used Provided By Financing Activities", 0),
                    "effect_of_forex_changes_on_cash": cashflow.get("Effect Of Forex Changes On Cash", 0),
                    "net_change_in_cash": cashflow.get("Net Change In Cash", 0),
                    "cash_at_end_of_period": cashflow.get("Cash At End Of Period", 0),
                    "cash_at_beginning_of_period": cashflow.get("Cash At Beginning Of Period", 0),
                    "operating_cash_flow": cashflow.get("Operating Cash Flow", 0),
                    "capital_expenditure": cashflow.get("Capital Expenditure", 0),
                    "free_cash_flow": cashflow.get("Free Cash Flow", 0),
                    "link": cashflow.get("Link", None),
                    "final_link": cashflow.get("Final Link", None)
                }

                # Combine all parameters for query execution
                params = {**income_params, **balance_params, **cashflow_params}

                query = '''
                MERGE (stock:Stock_Ticker {ticker: $ticker})
                MERGE (year:Year {year: $year, ticker: $ticker})

                MERGE (income:Income_Statement {year: $year, ticker: $ticker})
                SET income.date = $date,
                    income.symbol = $symbol,
                    income.reported_currency = $reported_currency,
                    income.cik = $cik,
                    income.filling_date = $filling_date,
                    income.accepted_date = $accepted_date,
                    income.calendar_year = $calendar_year,
                    income.period = $period,
                    income.revenue = $revenue,
                    income.cost_of_revenue = $cost_of_revenue,
                    income.gross_profit = $gross_profit,
                    income.gross_profit_ratio = $gross_profit_ratio,
                    income.research_and_development_expenses = $research_and_development_expenses,
                    income.general_and_administrative_expenses = $general_and_administrative_expenses,
                    income.selling_and_marketing_expenses = $selling_and_marketing_expenses,
                    income.selling_general_and_administrative_expenses = $selling_general_and_administrative_expenses,
                    income.other_expenses = $other_expenses,
                    income.operating_expenses = $operating_expenses,
                    income.cost_and_expenses = $cost_and_expenses,
                    income.interest_income = $interest_income,
                    income.interest_expense = $interest_expense,
                    income.depreciation_and_amortization = $depreciation_and_amortization,
                    income.ebitda = $ebitda,
                    income.ebitdaratio = $ebitdaratio,
                    income.operating_income = $operating_income,
                    income.operating_income_ratio = $operating_income_ratio,
                    income.total_other_income_expenses_net = $total_other_income_expenses_net,
                    income.income_before_tax = $income_before_tax,
                    income.income_before_tax_ratio = $income_before_tax_ratio,
                    income.income_tax_expense = $income_tax_expense,
                    income.net_income = $net_income,
                    income.net_income_ratio = $net_income_ratio,
                    income.eps = $eps,
                    income.epsdiluted = $epsdiluted,
                    income.weighted_average_shs_out = $weighted_average_shs_out,
                    income.weighted_average_shs_out_dil = $weighted_average_shs_out_dil,
                    income.link = $link,
                    income.final_link = $final_link

                MERGE (balance:Balance_Sheet {year: $year, ticker: $ticker})
                SET balance.date = $date,
                    balance.symbol = $symbol,
                    balance.reported_currency = $reported_currency,
                    balance.cik = $cik,
                    balance.filling_date = $filling_date,
                    balance.accepted_date = $accepted_date,
                    balance.calendar_year = $calendar_year,
                    balance.period = $period,
                    balance.cash_and_cash_equivalents = $cash_and_cash_equivalents,
                    balance.short_term_investments = $short_term_investments,
                    balance.cash_and_short_term_investments = $cash_and_short_term_investments,
                    balance.net_receivables = $net_receivables,
                    balance.inventory = $inventory,
                    balance.other_current_assets = $other_current_assets,
                    balance.total_current_assets = $total_current_assets,
                    balance.property_plant_equipment_net = $property_plant_equipment_net,
                    balance.goodwill = $goodwill,
                    balance.intangible_assets = $intangible_assets,
                    balance.goodwill_and_intangible_assets = $goodwill_and_intangible_assets,
                    balance.long_term_investments = $long_term_investments,
                    balance.tax_assets = $tax_assets,
                    balance.other_non_current_assets = $other_non_current_assets,
                    balance.total_non_current_assets = $total_non_current_assets,
                    balance.other_assets = $other_assets,
                    balance.total_assets = $total_assets,
                    balance.account_payables = $account_payables,
                    balance.short_term_debt = $short_term_debt,
                    balance.tax_payables = $tax_payables,
                    balance.deferred_revenue = $deferred_revenue,
                    balance.other_current_liabilities = $other_current_liabilities,
                    balance.total_current_liabilities = $total_current_liabilities,
                    balance.long_term_debt = $long_term_debt,
                    balance.deferred_revenue_non_current = $deferred_revenue_non_current,
                    balance.deferred_tax_liabilities_non_current = $deferred_tax_liabilities_non_current,
                    balance.other_non_current_liabilities = $other_non_current_liabilities,
                    balance.total_non_current_liabilities = $total_non_current_liabilities,
                    balance.other_liabilities = $other_liabilities,
                    balance.capital_lease_obligations = $capital_lease_obligations,
                    balance.total_liabilities = $total_liabilities,
                    balance.preferred_stock = $preferred_stock,
                    balance.common_stock = $common_stock,
                    balance.retained_earnings = $retained_earnings,
                    balance.accumulated_other_comprehensive_income_loss = $accumulated_other_comprehensive_income_loss,
                    balance.othertotal_stockholders_equity = $othertotal_stockholders_equity,
                    balance.total_stockholders_equity = $total_stockholders_equity,
                    balance.total_equity = $total_equity,
                    balance.total_liabilities_and_stockholders_equity = $total_liabilities_and_stockholders_equity,
                    balance.minority_interest = $minority_interest,
                    balance.total_liabilities_and_total_equity = $total_liabilities_and_total_equity,
                    balance.total_investments = $total_investments,
                    balance.total_debt = $total_debt,
                    balance.net_debt = $net_debt,
                    balance.link = $link,
                    balance.final_link = $final_link

                MERGE (cashflow:Cash_Flow {year: $year, ticker: $ticker})
                SET cashflow.date = $date,
                    cashflow.symbol = $symbol,
                    cashflow.reported_currency = $reported_currency,
                    cashflow.cik = $cik,
                    cashflow.filling_date = $filling_date,
                    cashflow.accepted_date = $accepted_date,
                    cashflow.calendar_year = $calendar_year,
                    cashflow.period = $period,
                    cashflow.net_income = $net_income,
                    cashflow.depreciation_and_amortization = $depreciation_and_amortization,
                    cashflow.deferred_income_tax = $deferred_income_tax,
                    cashflow.stock_based_compensation = $stock_based_compensation,
                    cashflow.change_in_working_capital = $change_in_working_capital,
                    cashflow.accounts_receivables = $accounts_receivables,
                    cashflow.inventory = $inventory,
                    cashflow.accounts_payables = $accounts_payables,
                    cashflow.other_working_capital = $other_working_capital,
                    cashflow.other_non_cash_items = $other_non_cash_items,
                    cashflow.net_cash_provided_by_operating_activities = $net_cash_provided_by_operating_activities,
                    cashflow.investments_in_property_plant_and_equipment = $investments_in_property_plant_and_equipment,
                    cashflow.acquisitions_net = $acquisitions_net,
                    cashflow.purchases_of_investments = $purchases_of_investments,
                    cashflow.sales_maturities_of_investments = $sales_maturities_of_investments,
                    cashflow.other_investing_activites = $other_investing_activites,
                    cashflow.net_cash_used_for_investing_activites = $net_cash_used_for_investing_activites,
                    cashflow.debt_repayment = $debt_repayment,
                    cashflow.common_stock_issued = $common_stock_issued,
                    cashflow.common_stock_repurchased = $common_stock_repurchased,
                    cashflow.dividends_paid = $dividends_paid,
                    cashflow.other_financing_activites = $other_financing_activites,
                    cashflow.net_cash_used_provided_by_financing_activities = $net_cash_used_provided_by_financing_activities,
                    cashflow.effect_of_forex_changes_on_cash = $effect_of_forex_changes_on_cash,
                    cashflow.net_change_in_cash = $net_change_in_cash,
                    cashflow.cash_at_end_of_period = $cash_at_end_of_period,
                    cashflow.cash_at_beginning_of_period = $cash_at_beginning_of_period,
                    cashflow.operating_cash_flow = $operating_cash_flow,
                    cashflow.capital_expenditure = $capital_expenditure,
                    cashflow.free_cash_flow = $free_cash_flow,
                    cashflow.link = $link,
                    cashflow.final_link = $final_link

                MERGE (stock)-[:HAS_FINANCIAL_STATEMENT_FOR_YEAR]->(year)
                MERGE (year)-[:HAS_INCOME_STATEMENT]->(income)
                MERGE (year)-[:HAS_BALANCE_SHEET]->(balance)
                MERGE (year)-[:HAS_CASH_FLOW_STATEMENT]->(cashflow)
                '''

                execute_query(query, params)
        print("Financial data loaded successfully.")
    except Exception as e:
        print("Error loading financial data:", e)

def load_news_data(news_data):
    try:
        news_data['Date of News'] = pd.to_datetime(news_data['Date of News']).dt.strftime('%Y-%m-%d')

        for _, row in news_data.iterrows():
            execute_query('''
            MATCH (stock:Stock_Ticker {ticker: $ticker})
            OPTIONAL MATCH (stock)-[:HAS_DAILY_PRICE]->(price:DailyPrice {date: date($date_of_news)})
            WITH stock, price
            FOREACH (_ IN CASE WHEN price IS NOT NULL THEN [1] ELSE [] END |
                MERGE (news:News {
                    ticker: $ticker,
                    headline: $headline,
                    source: $source,
                    date_of_news: date($date_of_news),
                    url: $url,
                    sentiment: $sentiment,
                    sentiment_score: $sentiment_score
                })
                MERGE (price)-[:HAS_NEWS]->(news)
            )
            FOREACH (_ IN CASE WHEN price IS NULL THEN [1] ELSE [] END |
                MERGE (news:News {
                    ticker: $ticker,
                    headline: $headline,
                    source: $source,
                    date_of_news: date($date_of_news),
                    url: $url,
                    sentiment: $sentiment,
                    sentiment_score: $sentiment_score
                })
                MERGE (dateNode:Date {date: date($date_of_news), ticker: $ticker})
                MERGE (stock)-[:HAS_NO_DAILY_PRICE]->(dateNode)
                MERGE (dateNode)-[:HAS_NEWS]->(news)
            )
            ''', {
                "ticker": row['Ticker'],
                "headline": row['Headline'],
                "source": row['Source'],
                "date_of_news": row['Date of News'],
                "url": row['URL'],
                "sentiment": row['Sentiment'],
                "sentiment_score": row['Sentiment_Score']
            })
        print("News data loaded successfully.")
    except Exception as e:
        print("Error loading news data:", e)

def clear_database():
    try:
        execute_query('MATCH ()-[r]->() DELETE r')
        execute_query('MATCH (n) DELETE n')
        print("Database cleared successfully.")
    except Exception as e:
        print("Error clearing database:", e)

def main():
    try:
        clear_database()
        tickers = ['AAPL', 'NVDA', 'AMD', 'JPM', 'BX', 'AMZN', 'TSLA', 'SBUX', 'JNJ', 'MRK']

        # Fetch your data
        company_profiles = FetchData.get_company_profile(tickers)
        stock_data = FetchData.get_stock_data(tickers)
        financial_data = FetchData.get_financial_data(tickers)
        news_data = FetchData.get_news_data(tickers)

        income_statement = financial_data['income_statement']
        balance_sheet = financial_data['balance_sheet']
        cash_flow = financial_data['cash_flow']

        load_company_profiles(company_profiles)
        load_stock_and_prices(stock_data)
        load_financial_data(income_statement, balance_sheet, cash_flow)
        load_news_data(news_data)

        print("All data loaded successfully.")
    except Exception as e:
        print("Error in main execution:", e)

if __name__ == "__main__":
    main()
