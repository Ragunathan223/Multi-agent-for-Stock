# Multi-agent-for-Stock

This project is a Streamlit web application designed for analyzing stocks listed on the National Stock Exchange of India (NSE). It utilizes multiple agents and tools to provide a comprehensive overview of stock performance, predictions, sentiment analysis, and company information.

## Features

-   **Stock Data Visualization:** Fetches and displays historical stock data from Yahoo Finance, including candlestick charts.
-   **7-Day Stock Price Prediction:** Predicts stock prices for the next 7 days using a linear regression model.
-   **Sentiment Analysis:** Analyzes news articles related to the stock to determine overall sentiment (positive, negative, or neutral).
-   **LLM Analysis:** Uses a Large Language Model (LLM) through Groq to provide in-depth analysis of stock trends and predictions.
-   **Company Information:** Displays key company information such as sector, industry, market cap, and financial ratios.
-   **Multi-Agent Architecture:** Integrates multiple agents for fetching stock data, making predictions, analyzing sentiment, and providing LLM insights.
-   **News Fetching:** Fetches news from NewsAPI, Google News, and Financial Express.
-   **Error Handling:** Robust error handling for API calls and data processing.

## Screenshots

<img width="923" alt="image" src="https://github.com/user-attachments/assets/b3cbc180-abd4-4098-9e5f-9e98dd2fa950" />

<img width="489" alt="image" src="https://github.com/user-attachments/assets/ca8188c8-5842-46e3-a5fe-9b2b1f098eae" />

<img width="518" alt="image" src="https://github.com/user-attachments/assets/425e8faa-cca2-4124-8528-c86a7c6731e2" />

<img width="461" alt="image" src="https://github.com/user-attachments/assets/22cb4188-b967-4203-8a72-42049e3e82cb" />

<img width="427" alt="image" src="https://github.com/user-attachments/assets/942c40e3-7e8f-4bb0-983e-b406b83cc91e" />

## Prerequisites

Before running the application, ensure you have the following installed:

-   Python 3.6 or higher
-   Streamlit
-   yfinance
-   pandas
-   plotly
-   scikit-learn
-   langchain-groq
-   nltk
-   newspaper3k
-   requests
-   beautifulsoup4

You can install these dependencies using pip:

```bash
pip install streamlit yfinance pandas plotly scikit-learn langchain-groq nltk newspaper3k requests beautifulsoup4
