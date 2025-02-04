from datetime import datetime, timedelta

import pandas as pd
import requests
import ta
from textblob import TextBlob


# Step 1: Fetch NEPSE Overall Details
def get_nepse_overall_details():
    nepse_index_url = "https://nepalstock.onrender.com/nepse-index"
    response = requests.get(nepse_index_url)
    nepse_index = response.json()

    market_status_url = "https://nepalstock.onrender.com/nepse-data/market-open"
    response = requests.get(market_status_url)
    market_status = response.json()

    return {"nepse_index": nepse_index, "market_status": market_status["isOpen"]}


# Step 2: Get Listed Stocks
def get_listed_stocks():
    url = "https://nepalstock.onrender.com/security?nonDelisted=true"
    response = requests.get(url)
    stocks = response.json()

    listed_stock = []

    for stock in stocks:
        if stock.get("activeStatus") != "A":
            continue
        listed_stock.append(stock)
    return listed_stock


# Step 3: Fetch News Disclosures
def get_news():
    url = "https://nepalstock.onrender.com/news/companies/disclosure"
    response = requests.get(url)
    news = response.json()
    return news


# Helper Function: Calculate Buy/Sell Pressure
def calculate_buy_sell_pressure(historical_data):
    total_buy_pressure = 0
    total_sell_pressure = 0

    for i in range(1, len(historical_data)):
        if "closePrice" not in historical_data[i] or "closePrice" not in historical_data[i - 1]:
            continue
        prev_close = historical_data[i - 1]["closePrice"]
        current_close = historical_data[i]["closePrice"]

        if current_close > prev_close:
            total_buy_pressure += historical_data[i]["totalTradedQuantity"]
        elif current_close < prev_close:
            total_sell_pressure += historical_data[i]["totalTradedQuantity"]

    return total_buy_pressure, total_sell_pressure


# Helper Function: Analyze Technical Indicators
def analyze_technical_indicators(historical_data):
    df = pd.DataFrame(historical_data)

    if "closePrice" not in df:
        return "Unavailable"
    df["RSI"] = ta.momentum.RSIIndicator(df["closePrice"]).rsi()
    df["SMA_50"] = df["closePrice"].rolling(window=50).mean()
    df["SMA_200"] = df["closePrice"].rolling(window=200).mean()

    signal = "Hold"
    if df["RSI"].iloc[-1] < 30 and df["SMA_50"].iloc[-1] > df["SMA_200"].iloc[-1]:
        signal = "Buy"
    elif df["RSI"].iloc[-1] > 70 and df["SMA_50"].iloc[-1] < df["SMA_200"].iloc[-1]:
        signal = "Sell"

    return signal


# Helper Function: Analyze News Sentiment
def analyze_news_sentiment(stock, news):
    symbol = stock["symbol"]
    stock_news = [item for item in news["companyNews"] if item.get("symbol") == symbol]
    sentiment_scores = []

    for news_item in stock_news:
        text = news_item.get("newsHeadline", "") + " " + news_item.get("remarks", "")
        analysis = TextBlob(text)
        sentiment_score = analysis.sentiment.polarity
        sentiment_scores.append(sentiment_score)

    average_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    return average_sentiment


# Step 4: Analyze Stock
def analyze_stock(stock, news):
    security_id = stock["id"]
    symbol = stock["symbol"]

    # Fetch Historical Data (1 year)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    historical_url = f"https://nepalstock.onrender.com/market/history/security/{security_id}"
    params = {"startDate": start_date, "endDate": end_date, "size": 500}
    response = requests.get(historical_url, params=params)
    historical_data = response.json().get("content", [])

    # Calculate Buy/Sell Pressure
    total_buy_pressure, total_sell_pressure = calculate_buy_sell_pressure(historical_data)

    # Analyze Technical Indicators
    technical_signal = analyze_technical_indicators(historical_data)

    # Analyze News Sentiment
    average_sentiment = analyze_news_sentiment(stock, news)

    # Combine Signals
    net_pressure = total_buy_pressure - total_sell_pressure
    signal = "Hold"
    if net_pressure > 0 and average_sentiment > 0 and technical_signal == "Buy":
        signal = "Buy"
    elif net_pressure < 0 and average_sentiment < 0 and technical_signal == "Sell":
        signal = "Sell"

    return {
        "Symbol": symbol,
        "Buy Pressure": total_buy_pressure,
        "Sell Pressure": total_sell_pressure,
        "Net Pressure": net_pressure,
        "Sentiment Score": average_sentiment,
        "Technical Signal": technical_signal,
        "Final Signal": signal,
    }


# Step 5: Send Buy/Sell Signals
def send_signals():
    nepse_details = get_nepse_overall_details()
    print(f"NEPSE Index: {nepse_details['nepse_index'][3]['currentValue']}")
    print(f"Market Status: {nepse_details['market_status']}")

    stocks = get_listed_stocks()
    news = get_news()

    results = []
    for stock in stocks:
        try:
            result = analyze_stock(stock, news)
            results.append(result)
        except Exception as e:
            print(f"Error analyzing {stock['symbol']}: {e}")

    for result in results:
        print(f"Stock: {result['Symbol']}, Final Signal: {result['Final Signal']}")

    results_df = pd.DataFrame(results)
    results_df.to_csv("buy_sell_signals.csv", index=False)
    print("Signals saved to buy_sell_signals.csv")


# Main Execution
if __name__ == "__main__":
    send_signals()
