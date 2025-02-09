import concurrent.futures
import random
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from textblob import TextBlob


# Step 1: Fetch NEPSE Overall Details
def get_nepse_overall_details():
    nepse_index_url = "https://nepalstock.onrender.com/nepse-index"
    market_status_url = "https://nepalstock.onrender.com/nepse-data/market-open"
    with requests.Session() as session:
        index_response = session.get(nepse_index_url)
        status_response = session.get(market_status_url)
    if index_response.status_code >= 400 or status_response.status_code >= 400:
        raise Exception("Some error occurred!")
    return {
        "nepse_index": index_response.json(),
        "market_status": status_response.json()["isOpen"],
    }


# Step 2: Get Listed Stocks
def get_listed_stocks():
    url = "https://nepalstock.onrender.com/security?nonDelisted=true"
    with requests.Session() as session:
        response = session.get(url)
    if response.status_code >= 400:
        raise Exception("Some error occurred")
    return [stock for stock in response.json() if stock.get("activeStatus") == "A"]


# Step 3: Fetch News Disclosures
def get_news():
    url = "https://nepalstock.onrender.com/news/companies/disclosure"
    with requests.Session() as session:
        response = session.get(url)
    if response.status_code >= 400:
        raise Exception("Some error occurred")
    news = response.json()
    news_dict = {}
    for item in news.get("companyNews", []):
        symbol = item.get("symbol")
        if symbol:
            news_dict.setdefault(symbol, []).append(item)
    return news_dict


# Helper Function: Calculate Buy/Sell Pressure
def calculate_buy_sell_pressure(historical_data):
    if not historical_data:
        return 0, 0
    df = pd.DataFrame(historical_data)
    if "closePrice" not in df.columns or "totalTradedQuantity" not in df.columns:
        return 0, 0
    # Use percentage change for better sensitivity
    df["price_change_pct"] = df["closePrice"].pct_change() * 100
    df["pressure"] = df["totalTradedQuantity"] * df["price_change_pct"].abs()
    buy_pressure = df[df["price_change_pct"] > 0]["pressure"].sum()
    sell_pressure = df[df["price_change_pct"] < 0]["pressure"].sum()
    return buy_pressure, sell_pressure


# Helper Function: Analyze News Sentiment
def analyze_news_sentiment(stock, news_dict):
    symbol = stock["symbol"]
    stock_news = news_dict.get(symbol, [])
    if not stock_news:
        return 0
    sentiment_scores = []
    try:
        for news_item in stock_news:
            # Consider recent news more important
            days_old = (datetime.now() - datetime.strptime(news_item["publishedDate"], "%Y-%m-%d")).days
            weight = max(0, 1 - days_old / 30)  # Weight decays over 30 days
            text = f"{news_item.get('newsHeadline', '')} {news_item.get('remarks', '')}"
            analysis = TextBlob(text)
            sentiment_scores.append(analysis.sentiment.polarity * weight)
    except Exception as e:
        print(f"Sentiment error for {symbol}: {str(e)}")
    if not sentiment_scores:
        return 0
    return sum(sentiment_scores) / len(sentiment_scores)


# Enhanced Technical Analysis
def calculate_moving_averages(df: pd.DataFrame, short_window: int = 5, long_window: int = 20) -> pd.DataFrame:
    """Calculate moving averages"""
    df["short_ma"] = df["closePrice"].rolling(window=short_window, min_periods=1).mean()
    df["long_ma"] = df["closePrice"].rolling(window=long_window, min_periods=1).mean()
    return df


def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    """Calculate Bollinger Bands"""
    df["middle_band"] = df["closePrice"].rolling(window=window, min_periods=1).mean()
    df["std_dev"] = df["closePrice"].rolling(window=window, min_periods=1).std()
    df["upper_band"] = df["middle_band"] + (num_std * df["std_dev"])
    df["lower_band"] = df["middle_band"] - (num_std * df["std_dev"])
    return df


def calculate_macd(
    df: pd.DataFrame, short_window: int = 12, long_window: int = 26, signal_window: int = 9
) -> pd.DataFrame:
    """Calculate MACD"""
    df["ema_short"] = df["closePrice"].ewm(span=short_window, adjust=False).mean()
    df["ema_long"] = df["closePrice"].ewm(span=long_window, adjust=False).mean()
    df["macd"] = df["ema_short"] - df["ema_long"]
    df["signal"] = df["macd"].ewm(span=signal_window, adjust=False).mean()
    df["histogram"] = df["macd"] - df["signal"]
    return df


def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calculate RSI"""
    delta = df["closePrice"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def analyze_technical_indicators(historical_data):
    if not historical_data:
        return {"Signal": "Neutral"}
    df = pd.DataFrame(historical_data)
    if "closePrice" not in df.columns or len(df) < 20:
        return {"Signal": "Neutral"}

    # Calculate technical indicators
    df = calculate_moving_averages(df)
    df = calculate_bollinger_bands(df)
    df = calculate_macd(df)
    df = calculate_rsi(df)

    # Generate signals
    signals = []
    if df["short_ma"].iloc[-1] > df["long_ma"].iloc[-1]:
        signals.append("Buy")
    elif df["short_ma"].iloc[-1] < df["long_ma"].iloc[-1]:
        signals.append("Sell")

    if df["closePrice"].iloc[-1] < df["lower_band"].iloc[-1]:
        signals.append("Buy")
    elif df["closePrice"].iloc[-1] > df["upper_band"].iloc[-1]:
        signals.append("Sell")

    if df["macd"].iloc[-1] > df["signal"].iloc[-1]:
        signals.append("Buy")
    elif df["macd"].iloc[-1] < df["signal"].iloc[-1]:
        signals.append("Sell")

    if df["rsi"].iloc[-1] < 30:
        signals.append("Buy")
    elif df["rsi"].iloc[-1] > 70:
        signals.append("Sell")

    # Determine final signal
    if not signals:
        return {"Signal": "Neutral"}
    return {"Signal": max(set(signals), key=signals.count)}


# Improved Stock Analysis with better decision logic
def analyze_stock(stock, news_dict):
    try:
        security_id = stock["id"]
        symbol = stock["symbol"]
        # Fetch Historical Data with pagination
        historical_data = []
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        max_attempt = 50
        attempt = 0
        while True:
            try:
                attempt += 1
                url = f"https://nepalstock.onrender.com/market/history/security/{security_id}"
                params = {"startDate": start_date, "endDate": end_date, "size": 500}
                with requests.Session() as session:
                    response = session.get(url, params=params)
                    if response.status_code >= 400:
                        raise Exception("Some error occurred")
                    data = response.json()
                historical_data = data.get("content", [])
                break
            except Exception as e:
                if attempt >= max_attempt or "Some error occurred" not in str(e):
                    print(f"Error fetching data for {symbol}: {str(e)}")
                    break
                time.sleep(random.randint(0, 5))

        # Calculate metrics
        buy_pressure, sell_pressure = calculate_buy_sell_pressure(historical_data)
        net_pressure = buy_pressure - sell_pressure
        technical_analysis = analyze_technical_indicators(historical_data)
        sentiment = analyze_news_sentiment(stock, news_dict)

        # Decision matrix
        signal = "Hold"
        buy_points = 0
        sell_points = 0

        # Technical signals
        if technical_analysis["Signal"] == "Buy":
            buy_points += 2
        elif technical_analysis["Signal"] == "Sell":
            sell_points += 2

        # Pressure signals
        if net_pressure > 0:
            buy_points += 1
        elif net_pressure < 0:
            sell_points += 1

        # Sentiment signals
        if sentiment > 0.05:
            buy_points += 1
        elif sentiment < -0.05:
            sell_points += 1

        # Final decision
        if buy_points >= 3 and buy_points > sell_points:
            signal = "Buy"
        elif sell_points >= 3 and sell_points > buy_points:
            signal = "Sell"

        return {
            "Symbol": symbol,
            "Buy Pressure": int(buy_pressure),
            "Sell Pressure": int(sell_pressure),
            "Net Pressure": int(net_pressure),
            "Sentiment Score": round(sentiment, 2),
            "Technical Signal": technical_analysis["Signal"],
            "Final Signal": signal,
        }
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")
        return None


def send_signals():
    print("Starting analysis...")
    nepse_details = get_nepse_overall_details()
    print(f"Market Status: {nepse_details['market_status']}")
    stocks = get_listed_stocks()
    news_dict = get_news()
    print(f"Analyzing {len(stocks)} stocks...")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(analyze_stock, stock, news_dict) for stock in stocks]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
                print(
                    f"{len(results)}/{len(stocks)} : "
                    f"{result['Symbol']}: {result['Final Signal']} "
                    f"(Technical: {result['Technical Signal']}, "
                    f"Sentiment: {result['Sentiment Score']}, "
                    f"Net Pressure: {result['Net Pressure']})"
                )
    # Categorize and sort results
    buy_signals = [r for r in results if r["Final Signal"] == "Buy"]
    sell_signals = [r for r in results if r["Final Signal"] == "Sell"]
    hold_signals = [r for r in results if r["Final Signal"] == "Hold"]

    # Sort buy signals by technical confirmation, net pressure, and sentiment
    buy_sorted = sorted(
        buy_signals,
        key=lambda x: (
            -(1 if x["Technical Signal"] == "Buy" else 0),  # 'Buy' technical first
            -x["Net Pressure"],  # Higher net pressure first
            -x["Sentiment Score"],  # Higher sentiment first
        ),
    )

    # Sort sell signals by technical confirmation, net pressure, and sentiment
    sell_sorted = sorted(
        sell_signals,
        key=lambda x: (
            -(1 if x["Technical Signal"] == "Sell" else 0),  # 'Sell' technical first
            x["Net Pressure"],  # Lower (more negative) net pressure first
            x["Sentiment Score"],  # Lower sentiment first
        ),
    )

    # Combine all categories with buys first, then sells, then holds
    sorted_results = buy_sorted + sell_sorted + hold_signals

    # Save sorted results
    results_df = pd.DataFrame(sorted_results)
    results_df.to_csv("signals.csv", index=False)
    print("Analysis complete. Results saved to signals.csv")


if __name__ == "__main__":
    start_time = datetime.now()
    send_signals()
    print(f"Execution time: {datetime.now() - start_time}")
