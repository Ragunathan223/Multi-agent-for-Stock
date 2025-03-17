import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
from langchain_groq import ChatGroq
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from newspaper import Article
from langchain.agents import initialize_agent, Tool, AgentType
from concurrent.futures import ThreadPoolExecutor
import requests
import json
from bs4 import BeautifulSoup
import re

# Download necessary NLTK data
nltk.download('vader_lexicon')

# Manually set API Keys (⚠️ Replace with actual keys)
GROQ_API_KEY = "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"  
NEWSAPI_KEY = "pub_743520e85c70cdf17bf41e9abc19369666092"
NEWSAPI_URL = "https://newsapi.org/v2/everything"

# Initialize LLM
if not GROQ_API_KEY or "your_actual_groq_api_key" in GROQ_API_KEY:
    st.error("⚠️ Missing or invalid GROQ API Key. Please set a valid key.")
    llm = None
else:
    llm = ChatGroq(model="llama3-70b-8192", temperature=0.2, groq_api_key=GROQ_API_KEY)

# Function to fetch NSE stock data
def get_stock_data(ticker, period="1y"):
    try:
        # For NSE stocks, append ".NS" to the ticker
        nse_ticker = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
        stock = yf.Ticker(nse_ticker)
        history = stock.history(period=period)
        return history
    except Exception as e:
        return f"Error fetching stock data: {e}"

# Function to predict stock prices
def predict_stock_prices(df):
    if df is None or df.empty:
        return None, None, None

    df['Day'] = range(len(df))
    X = df[['Day']]
    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    last_day = df['Day'].iloc[-1]
    future_days = [[last_day + i] for i in range(1, 8)]
    future_predictions = model.predict(future_days)
    future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, 8)]

    return mse, future_predictions, future_dates

# Function to fetch news from multiple sources
def fetch_news_articles(ticker, company_name=None):
    all_articles = []
    
    # Clean ticker for search
    search_ticker = ticker.replace(".NS", "")
    
    # Use company name if provided, otherwise use ticker
    if company_name:
        search_term = company_name
    else:
        search_term = search_ticker
        
    # Try to get company name from Yahoo Finance if not provided
    if not company_name:
        try:
            nse_ticker = f"{search_ticker}.NS" if not search_ticker.endswith(".NS") else search_ticker
            stock = yf.Ticker(nse_ticker)
            info = stock.info
            if 'longName' in info and info['longName']:
                search_term = info['longName']
        except:
            pass
    
    # Get today's date for filtering
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Method 1: NewsAPI (with API key)
    if NEWSAPI_KEY:
        try:
            # Search for both company name and ticker for better results
            params = {
                "q": f"{search_term} OR {search_ticker} stock",
                "apiKey": NEWSAPI_KEY,
                "language": "en",
                "from": yesterday,  # Get news from yesterday
                "sortBy": "publishedAt"
            }
            response = requests.get(NEWSAPI_URL, params=params)
            news_data = response.json()
            if "articles" in news_data:
                for article in news_data["articles"][:10]:  # Get up to 10 articles
                    all_articles.append({
                        "title": article["title"],
                        "description": article.get("description", ""),
                        "url": article["url"],
                        "source": article["source"]["name"],
                        "published": article["publishedAt"]
                    })
        except Exception as e:
            st.warning(f"Error fetching news from NewsAPI: {e}")
    
    # Method 2: Search Google News (as fallback)
    if len(all_articles) < 5:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            # Construct search URL for Google News
            search_url = f"https://www.google.com/search?q={search_term}+{search_ticker}+stock+news&tbm=nws&source=lnt&tbs=qdr:d"
            response = requests.get(search_url, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                news_elements = soup.find_all('div', {'class': 'SoaBEf'})
                
                for element in news_elements[:5]:  # Limit to 5 results
                    title_element = element.find('div', {'role': 'heading'})
                    if title_element:
                        title = title_element.text
                        description = element.find('div', {'class': 'GI74Re'}).text if element.find('div', {'class': 'GI74Re'}) else ""
                        link_element = element.find('a')
                        url = link_element['href'] if link_element and 'href' in link_element.attrs else ""
                        source = element.find('div', {'class': 'CEMjEf'}).text if element.find('div', {'class': 'CEMjEf'}) else "Unknown"
                        
                        all_articles.append({
                            "title": title,
                            "description": description,
                            "url": url,
                            "source": source,
                            "published": "Today"  # Approximate date
                        })
        except Exception as e:
            st.warning(f"Error fetching news from Google: {e}")
    
    # Method 3: Financial Express (India-specific)
    if len(all_articles) < 5:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            search_url = f"https://www.financialexpress.com/market/stock-market/"
            response = requests.get(search_url, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                news_elements = soup.find_all('div', {'class': 'listitems'})
                
                for element in news_elements[:5]:  # Limit to 5 results
                    title_element = element.find('h3')
                    if title_element:
                        # Only include if it mentions our stock
                        title = title_element.text
                        if search_term.lower() in title.lower() or search_ticker.lower() in title.lower():
                            description = element.find('p').text if element.find('p') else ""
                            link_element = element.find('a')
                            url = link_element['href'] if link_element and 'href' in link_element.attrs else ""
                            
                            all_articles.append({
                                "title": title,
                                "description": description,
                                "url": url,
                                "source": "Financial Express",
                                "published": "Today"  # Approximate date
                            })
        except Exception as e:
            st.warning(f"Error fetching news from Financial Express: {e}")
    
    return all_articles

# Function to analyze sentiment
def get_news_sentiment(ticker):
    try:
        # First, get the company name for better search results
        try:
            nse_ticker = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
            stock = yf.Ticker(nse_ticker)
            info = stock.info
            company_name = info.get('longName', ticker)
        except:
            company_name = ticker
            
        # Fetch news articles
        articles = fetch_news_articles(ticker, company_name)
        
        if not articles:
            return "No news articles found.", []
        
        # Analyze sentiment for each article
        analyzer = SentimentIntensityAnalyzer()
        sentiment_results = []
        
        for article in articles:
            text = article["title"] + " " + article["description"]
            scores = analyzer.polarity_scores(text)
            sentiment = "Positive" if scores['compound'] > 0.1 else "Negative" if scores['compound'] < -0.1 else "Neutral"
            
            sentiment_results.append({
                "title": article["title"],
                "source": article["source"],
                "published": article["published"],
                "url": article["url"],
                "sentiment": sentiment,
                "score": scores['compound']
            })
        
        # Calculate average sentiment
        avg_sentiment = sum(item["score"] for item in sentiment_results) / len(sentiment_results)
        overall_sentiment = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
        
        return f"Average sentiment: {overall_sentiment} ({avg_sentiment:.2f})", sentiment_results
    
    except Exception as e:
        return f"Error in sentiment analysis: {e}", []

# LLM Analysis function with error handling
def llm_analysis(ticker, prediction_data):
    if not llm:
        return "🚨 LLM Analysis Failed: Missing or invalid API Key."

    prompt = f"Analyze {ticker} stock from the National Stock Exchange (NSE) of India, considering historical data and predicted trend for the next 7 days: {prediction_data}"
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"🚨 LLM Analysis Failed: {e}"

# Initialize Agents
stock_agent = Tool(name="Stock Data Agent", func=get_stock_data, description="Fetch NSE stock data from Yahoo Finance.")
prediction_agent = Tool(name="Prediction Agent", func=predict_stock_prices, description="Predict stock price for the next 7 days.")
sentiment_agent = Tool(name="Sentiment Agent", func=get_news_sentiment, description="Analyze stock sentiment from news articles.")
llm_agent = Tool(name="LLM Analysis Agent", func=llm_analysis, description="Analyze stock trends using AI.")

agents = [stock_agent, prediction_agent, sentiment_agent, llm_agent]
if llm:
    stock_analysis_agent = initialize_agent(agents, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
else:
    stock_analysis_agent = None

# Streamlit UI - updated for NSE
st.title("📈 NSE Stock Analysis with Multi-Agent AI")

# Default NSE stocks
nse_default_stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "BAJFINANCE", "AXISBANK", "LT", "HINDUNILVR"]
ticker = st.selectbox(
    "Select NSE Stock:",
    options=nse_default_stocks,
    index=0
)
manual_ticker = st.text_input("Or enter NSE stock ticker manually:", "")
if manual_ticker:
    ticker = manual_ticker.upper()

period = st.selectbox("Select Time Period:", ["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"])

if st.button("Analyze"):
    st.write(f"Analyzing {ticker} on NSE...")
    
    with ThreadPoolExecutor() as executor:
        future_stock = executor.submit(get_stock_data, ticker, period)
        future_sentiment = executor.submit(get_news_sentiment, ticker)

        stock_df = future_stock.result()
        sentiment_result, sentiment_articles = future_sentiment.result()

    if isinstance(stock_df, str):
        st.error(stock_df)
    else:
        st.subheader(f"📊 {ticker} Stock Data (NSE)")
        st.dataframe(stock_df.tail())

        st.subheader("📉 Candlestick Chart")
        fig = go.Figure(data=[go.Candlestick(x=stock_df.index,
                                             open=stock_df['Open'],
                                             high=stock_df['High'],
                                             low=stock_df['Low'],
                                             close=stock_df['Close'])])
        fig.update_layout(title=f"{ticker} Candlestick Chart (NSE)", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig)

        mse, future_predictions, future_dates = predict_stock_prices(stock_df)

        if future_predictions is not None:
            st.subheader("📈 7-Day Stock Price Prediction")
            st.write(f"🔹 Mean Squared Error: {mse:.2f}")

            future_df = pd.DataFrame({"Date": future_dates, "Predicted Close": future_predictions})
            st.dataframe(future_df)

            st.line_chart(future_df.set_index("Date"))

            st.subheader("🧠 LLM Analysis")
            llm_response = llm_analysis(ticker, future_predictions)
            st.write(llm_response)

        st.subheader("📰 Sentiment Analysis")
        st.write(sentiment_result)
        
        # Display detailed sentiment analysis for each article
        if sentiment_articles:
            st.subheader("📊 News Sentiment Breakdown")
            
            for i, article in enumerate(sentiment_articles):
                sentiment_color = "green" if article["sentiment"] == "Positive" else "red" if article["sentiment"] == "Negative" else "gray"
                
                with st.expander(f"{i+1}. {article['title']} [{article['sentiment']}]"):
                    st.write(f"**Source:** {article['source']}")
                    st.write(f"**Published:** {article['published']}")
                    st.write(f"**Sentiment Score:** {article['score']:.2f}")
                    if article['url']:
                        st.write(f"[Read Full Article]({article['url']})")
        else:
            st.info("No specific news articles found for sentiment analysis.")

# Display disclaimer
st.sidebar.markdown("""
## Disclaimer
This app provides NSE stock analysis for educational purposes only. The predictions are based on simple linear regression and should not be used for actual investment decisions.
""")

# Company Info Section
st.sidebar.markdown("## Company Information")
if st.sidebar.button("Get Company Info"):
    try:
        nse_ticker = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
        stock = yf.Ticker(nse_ticker)
        info = stock.info
        
        if info:
            st.sidebar.write(f"**Company:** {info.get('longName', ticker)}")
            st.sidebar.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.sidebar.write(f"**Industry:** {info.get('industry', 'N/A')}")
            st.sidebar.write(f"**Market Cap:** ₹{info.get('marketCap', 0)/10000000:.2f} Cr")
            st.sidebar.write(f"**52W High:** ₹{info.get('fiftyTwoWeekHigh', 0):.2f}")
            st.sidebar.write(f"**52W Low:** ₹{info.get('fiftyTwoWeekLow', 0):.2f}")
            st.sidebar.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
            st.sidebar.write(f"**EPS:** ₹{info.get('trailingEps', 0):.2f}")
            st.sidebar.write(f"**Dividend Yield:** {info.get('dividendYield', 0)*100:.2f}%")
    except Exception as e:
        st.sidebar.error(f"Error fetching company information: {e}")