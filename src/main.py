import concurrent.futures
import random
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
import ta
from textblob import TextBlob


# Step 1: Fetch NEPSE Overall Details
def get_nepse_overall_details():
    nepse_index_url = "https://nepalstock.onrender.com/nepse-index"
    market_status_url = "https://nepalstock.onrender.com/nepse-data/market-open"

    with requests.Session() as session:
        index_response = session.get(nepse_index_url)
        status_response = session.get(market_status_url)

    if index_response.status_code >= 400 or status_response.status_code >= 400:
        raise Exception("Some error occured!")

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
        raise Exception("Some error occured")
    return [stock for stock in response.json() if stock.get("activeStatus") == "A"]


# Step 3: Fetch News Disclosures
def get_news():
    url = "https://nepalstock.onrender.com/news/companies/disclosure"
    with requests.Session() as session:
        response = session.get(url)

    if response.status_code >= 400:
        raise Exception("Some error occured")

    news = response.json()

    news_dict = {}
    for item in news.get("companyNews", []):
        symbol = item.get("symbol")
        if symbol:
            news_dict.setdefault(symbol, []).append(item)
    return news_dict


# Helper Function: Improved Buy/Sell Pressure Calculation
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


# Helper Function: Enhanced Technical Analysis
def analyze_technical_indicators(historical_data):
    if not historical_data:
        return "Neutral"

    df = pd.DataFrame(historical_data)
    if "closePrice" not in df.columns or len(df) < 20:
        return "Neutral"

    close_prices = df["closePrice"].dropna()

    # Use shorter windows for Nepalese market
    try:
        rsi = ta.momentum.RSIIndicator(close_prices, window=14).rsi().iloc[-1]
        sma20 = close_prices.rolling(20).mean().iloc[-1]
        sma50 = close_prices.rolling(50).mean().iloc[-1]
    except Exception as e:
        print(f"Technical analysis error: {str(e)}")
        return "Neutral"

    signals = []

    if rsi < 35:
        signals.append("Buy")
    elif rsi > 65:
        signals.append("Sell")

    if sma20 > sma50:
        signals.append("Buy")
    elif sma20 < sma50:
        signals.append("Sell")

    if not signals:
        return "Neutral"
    return max(set(signals), key=signals.count)


# Helper Function: News Sentiment Analysis with Weighting
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
                        raise Exception("Some error occured")

                    data = response.json()

                historical_data = data.get("content", [])
                break

            except Exception as e:
                if attempt >= max_attempt or "Some error occured" not in e.args[0]:
                    print(f"Error fetching data for {symbol}: {str(e)}")
                    break

                time.sleep(random.randint(0, 5))

        # Calculate metrics
        buy_pressure, sell_pressure = calculate_buy_sell_pressure(historical_data)
        net_pressure = buy_pressure - sell_pressure
        technical_signal = analyze_technical_indicators(historical_data)
        sentiment = analyze_news_sentiment(stock, news_dict)

        # Dynamic threshold based on market volume
        pressure_threshold = abs(net_pressure) * 0.3  # 30% of net pressure magnitude

        # Decision matrix
        signal = "Hold"
        buy_points = 0
        sell_points = 0

        # Technical signals
        if "Buy" in technical_signal:
            buy_points += 2
        if "Sell" in technical_signal:
            sell_points += 2

        # Pressure signals
        if net_pressure > pressure_threshold:
            buy_points += 1
        elif net_pressure < -pressure_threshold:
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
            "Technical Signal": technical_signal,
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
