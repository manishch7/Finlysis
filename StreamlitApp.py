import streamlit as st
import openai

import FinancialStatementAnalysis
import InvestmentDecisionAssistant
import StockAnalysis
import FinancialNewsAnalysis
import warnings
warnings.filterwarnings("ignore")
import os

OPEN_API_KEY = os.getenv("OPEN_API_KEY")

openai.api_key = OPEN_API_KEY

# HELPER FUNCTION !!
def get_question_type(question):
    messages = [
        {"role": "system", "content": (
            "You are an assistant that classifies questions. The only possible types are: "
            "'Investment-related', 'Stock-related', 'Financial-News-related'. "
            "If the question is about financial statements (e.g., income statement, balance sheet), classify it as 'Financial-Statement-related'."
        )},
        {"role": "user", "content": f"Classify this question: {question}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content.strip()

def get_answer(question, question_type):
    answer = []
    try:
        if question_type == "Investment-related":
            answer = InvestmentDecisionAssistant.invest_decision_main(question)
            return answer
        elif question_type == "Stock-related":
            answer = StockAnalysis.stock_analysis_main(question)
        elif question_type == "Financial-News-related":
            answer = FinancialNewsAnalysis.financial_news_analysis_main(question)
        elif question_type ==  "Financial-Statement-related":
            answer = FinancialStatementAnalysis.financial_statements_analysis_main(question)

    except Exception as e:
        return f"Error: {e}"
    return answer

# HELPER FUNCTION ENDS !!

# Streamlit application
def main():
    # Application Title
    st.title("💹 Investment Decision Assistant")
    st.markdown(
        """
        <center><h5>Welcome! </h5>
        <body>Here to provide insightful answers to help you make an investment decision for the right company!
        </body></center><br>
        """,
    unsafe_allow_html=True
    )

    # Initialize the session state for the loop
    if "exit_triggered" not in st.session_state:
        st.session_state["exit_triggered"] = False

    user_question = st.text_input("Enter your question:")
    question_type = get_question_type(user_question)
    if user_question:
        if user_question.lower() == "exit":
            st.session_state["exit_triggered"] = True
            st.write("Goodbye!")
        else:
            with st.spinner("Thinking..."):
                if question_type == "Investment-related":
                    answer = get_answer(user_question, question_type)
                    st.write(answer)

                else:
                    all_answer = get_answer(user_question, question_type)
                    for answer in all_answer:
                        st.write(answer)

    if st.button("Exit"):
        st.write("Thank you for using the Investment Decision Assistant.")
        st.write("Exiting application...")
        st.session_state["exit_triggered"] = True
        st.stop()

if __name__ == "__main__":
    main()