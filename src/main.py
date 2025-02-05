import concurrent.futures
import random
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
import ta
from textblob import TextBlob


# Step 1: Fetch NEPSE Overall Details (Optimized with caching)
def get_nepse_overall_details():
    nepse_index_url = "https://nepalstock.onrender.com/nepse-index"
    market_status_url = "https://nepalstock.onrender.com/nepse-data/market-open"

    with requests.Session() as session:
        index_response = session.get(nepse_index_url)
        status_response = session.get(market_status_url)

    return {
        "nepse_index": index_response.json(),
        "market_status": status_response.json()["isOpen"],
    }


# Step 2: Get Listed Stocks (Optimized with request session)
def get_listed_stocks():
    url = "https://nepalstock.onrender.com/security?nonDelisted=true"
    with requests.Session() as session:
        response = session.get(url)
    return [stock for stock in response.json() if stock.get("activeStatus") == "A"]


# Step 3: Fetch News Disclosures (Optimized with preprocessing)
def get_news():
    url = "https://nepalstock.onrender.com/news/companies/disclosure"
    with requests.Session() as session:
        response = session.get(url)
    news = response.json()

    # Preprocess news into dictionary by symbol
    news_dict = {}
    for item in news.get("companyNews", []):
        symbol = item.get("symbol")
        if symbol:
            news_dict.setdefault(symbol, []).append(item)
    return news_dict


# Helper Function: Vectorized Buy/Sell Pressure Calculation
def calculate_buy_sell_pressure(historical_data):
    if not historical_data:
        return 0, 0

    df = pd.DataFrame(historical_data)
    if "closePrice" not in df.columns or "totalTradedQuantity" not in df.columns:
        return 0, 0

    df["price_change"] = df["closePrice"].diff()
    buy_mask = df["price_change"] > 0
    sell_mask = df["price_change"] < 0

    total_buy = df.loc[buy_mask, "totalTradedQuantity"].sum()
    total_sell = df.loc[sell_mask, "totalTradedQuantity"].sum()

    return total_buy, total_sell


# Helper Function: Optimized Technical Analysis
def analyze_technical_indicators(historical_data):
    if not historical_data:
        return "Unavailable"

    df = pd.DataFrame(historical_data)
    if "closePrice" not in df.columns:
        return "Unavailable"

    close_prices = df["closePrice"].dropna()
    if len(close_prices) < 200:
        return "Insufficient Data"

    try:
        rsi = ta.momentum.RSIIndicator(close_prices).rsi().iloc[-1]
        sma50 = close_prices.rolling(50).mean().iloc[-1]
        sma200 = close_prices.rolling(200).mean().iloc[-1]
    except Exception:
        return "Error"

    if rsi < 30 and sma50 > sma200:
        return "Buy"
    elif rsi > 70 and sma50 < sma200:
        return "Sell"
    return "Hold"


# Helper Function: Optimized News Sentiment Analysis
def analyze_news_sentiment(stock, news_dict):
    symbol = stock["symbol"]
    stock_news = news_dict.get(symbol, [])

    if not stock_news:
        return 0

    sentiment_scores = []
    try:
        for news_item in stock_news:
            text = f"{news_item.get('newsHeadline', '')} {news_item.get('remarks', '')}"
            analysis = TextBlob(text)
            sentiment_scores.append(analysis.sentiment.polarity)
    except Exception as e:
        print(e)

    if not sentiment_scores:
        return 0

    return sum(sentiment_scores) / len(sentiment_scores)


# Optimized Stock Analysis with parallel processing
def analyze_stock(stock, news_dict):
    try:
        security_id = stock["id"]
        symbol = stock["symbol"]

        # Fetch Historical Data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        url = f"https://nepalstock.onrender.com/market/history/security/{security_id}"
        historical_data = []

        max_retries = 50
        tries = 0

        while True:
            try:
                tries = tries + 1
                with requests.Session() as session:
                    response = session.get(
                        url,
                        params={
                            "startDate": start_date,
                            "endDate": end_date,
                            "size": 500,
                        },
                    )
                    historical_data = response.json().get("content", [])
                    break
            except Exception:
                if tries >= max_retries:
                    print("Max retries exceeded!")
                    break
                rand = random.randint(1, 4)
                # print(f"Sleeping for {rand} seconds")
                time.sleep(rand)

        # Parallel calculations
        total_buy, total_sell = calculate_buy_sell_pressure(historical_data)
        technical_signal = analyze_technical_indicators(historical_data)
        sentiment = analyze_news_sentiment(stock, news_dict)

        # Decision logic
        net_pressure = total_buy - total_sell
        signal = "Hold"

        if net_pressure > 100_000 and sentiment > 0.1 and technical_signal == "Buy":
            signal = "Buy"
        elif net_pressure < -100_000 and sentiment < -0.1 and technical_signal == "Sell":
            signal = "Sell"

        return {
            "Symbol": symbol,
            "Buy Pressure": total_buy,
            "Sell Pressure": total_sell,
            "Net Pressure": net_pressure,
            "Sentiment Score": round(sentiment, 2),
            "Technical Signal": technical_signal,
            "Final Signal": signal,
        }
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")
        return None


def send_signals():
    print("Starting analysis...")

    # Fetch initial data
    nepse_details = get_nepse_overall_details()
    print(f"NEPSE Index: {nepse_details['nepse_index'][3]['currentValue']}")
    print(f"Market Status: {nepse_details['market_status']}")

    stocks = get_listed_stocks()
    news_dict = get_news()

    print(f"Analyzing {len(stocks)} stocks...")

    # Parallel processing
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(analyze_stock, stock, news_dict) for stock in stocks]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
                print(f"Processed {len(results)}/{len(stocks)} {result['Symbol']}: {result['Final Signal']}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("signals.csv", index=False)
    print("Analysis complete. Results saved to signals.csv")


# Main Execution
if __name__ == "__main__":
    start_time = datetime.now()
    send_signals()
    print(f"Execution time: {datetime.now() - start_time}")
