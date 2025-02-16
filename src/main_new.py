import random
import time
from datetime import datetime, timedelta

from dask import delayed, compute
import pandas as pd
import requests
from textblob import TextBlob

# ---------------------------
# Global Configuration
# ---------------------------
BASE_URL = "https://nepalstock.onrender.com"

# ---------------------------
# Session Management
# ---------------------------
def get_session():
    return requests.Session()

# ---------------------------
# API Functions with Retry Logic
# ---------------------------
def get_nepse_overall_details(session, retries=5, delay=2):
    url_index = f"{BASE_URL}/nepse-index"
    url_status = f"{BASE_URL}/nepse-data/market-open"
    for attempt in range(retries):
        try:
            index_response = session.get(url_index)
            status_response = session.get(url_status)
            index_response.raise_for_status()
            status_response.raise_for_status()
            return {
                "nepse_index": index_response.json(),
                "market_status": status_response.json().get("isOpen", False),
            }
        except Exception as e:
            print(f"Attempt {attempt+1}/{retries} - Error fetching NEPSE details: {str(e)}")
            time.sleep(delay * (2 ** attempt))
    return {"nepse_index": {}, "market_status": False}

def get_listed_stocks(session, retries=5, delay=2):
    url = f"{BASE_URL}/security?nonDelisted=true"
    for attempt in range(retries):
        try:
            response = session.get(url)
            response.raise_for_status()
            # Return only active stocks
            return [stock for stock in response.json() if stock.get("activeStatus") == "A"]
        except Exception as e:
            print(f"Attempt {attempt+1}/{retries} - Error fetching listed stocks: {str(e)}")
            time.sleep(delay * (2 ** attempt))
    return []

def get_news(session, retries=5, delay=2):
    url = f"{BASE_URL}/news/companies/disclosure"
    for attempt in range(retries):
        try:
            response = session.get(url)
            response.raise_for_status()
            news = response.json()
            news_dict = {}
            for item in news.get("companyNews", []):
                symbol = item.get("symbol")
                if symbol:
                    news_dict.setdefault(symbol, []).append(item)
            return news_dict
        except Exception as e:
            print(f"Attempt {attempt+1}/{retries} - Error fetching news disclosures: {str(e)}")
            time.sleep(delay * (2 ** attempt))
    return {}

# ---------------------------
# Helper Functions for Calculations
# ---------------------------
def calculate_buy_sell_pressure(historical_data):
    try:
        if not historical_data:
            return 0, 0
        df = pd.DataFrame(historical_data)
        if "closePrice" not in df.columns or "totalTradedQuantity" not in df.columns:
            return 0, 0
        df["price_change_pct"] = df["closePrice"].pct_change() * 100
        df["pressure"] = df["totalTradedQuantity"] * df["price_change_pct"].abs()
        buy_pressure = df.loc[df["price_change_pct"] > 0, "pressure"].sum()
        sell_pressure = df.loc[df["price_change_pct"] < 0, "pressure"].sum()
        return buy_pressure, sell_pressure
    except Exception as e:
        print(f"Error calculating buy/sell pressure: {str(e)}")
        return 0, 0

def analyze_news_sentiment(stock, news_dict):
    try:
        symbol = stock["symbol"]
        stock_news = news_dict.get(symbol, [])
        if not stock_news:
            return 0
        sentiment_scores = []
        for news_item in stock_news:
            # Calculate days old and weight decay over 30 days
            days_old = (datetime.now() - datetime.strptime(news_item["publishedDate"], "%Y-%m-%d")).days
            weight = max(0, 1 - days_old / 30)
            text = f"{news_item.get('newsHeadline', '')} {news_item.get('remarks', '')}"
            analysis = TextBlob(text)
            sentiment_scores.append(analysis.sentiment.polarity * weight)
        return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    except Exception as e:
        print(f"Sentiment error for {stock.get('symbol', 'Unknown')}: {str(e)}")
        return 0

# ---------------------------
# Technical Analysis Functions
# ---------------------------
def calculate_moving_averages(df: pd.DataFrame, short_window: int = 5, long_window: int = 20) -> pd.DataFrame:
    df["short_ma"] = df["close"].rolling(window=short_window, min_periods=1).mean()
    df["long_ma"] = df["close"].rolling(window=long_window, min_periods=1).mean()
    return df

def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    required_columns = {"code", "date", "close"}
    if not required_columns.issubset(df.columns):
        print(f"Missing required columns for Bollinger Bands: {df.columns}")
        return df
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.sort_values(["code", "date"]).drop_duplicates(subset=["code", "date"], keep="last")
    grouped = df.groupby("code", group_keys=False)
    middle_band = grouped["close"].rolling(window=window).mean().reset_index(level=0, drop=True)
    std_dev = grouped["close"].rolling(window=window).std().reset_index(level=0, drop=True)
    df["middle_band"] = middle_band
    df["std_dev"] = std_dev
    df["Bollinger_Min"] = df["middle_band"] - (num_std * df["std_dev"])
    df["Bollinger_Max"] = df["middle_band"] + (num_std * df["std_dev"])
    return df

def calculate_macd(df: pd.DataFrame, short_window: int = 12, long_window: int = 26, signal_window: int = 9) -> pd.DataFrame:
    df["ema_short"] = df["close"].ewm(span=short_window, adjust=False).mean()
    df["ema_long"] = df["close"].ewm(span=long_window, adjust=False).mean()
    df["macd"] = df["ema_short"] - df["ema_long"]
    df["signal"] = df["macd"].ewm(span=signal_window, adjust=False).mean()
    df["histogram"] = df["macd"] - df["signal"]
    return df

def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta).clip(lower=0).rolling(window=window).mean()
    loss = loss.mask(loss == 0).ffill()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df

def analyze_technical_indicators(historical_data, symbol):
    try:
        if not historical_data:
            print(f"No historical data for {symbol}")
            return {"Signal": "Neutral"}
        df = pd.DataFrame(historical_data)
        df.rename(columns={"closePrice": "close", "businessDate": "date"}, inplace=True)
        df["code"] = symbol

        if not {"close", "date", "code"}.issubset(df.columns) or df.empty or len(df) < 20:
            print(f"Insufficient or missing data for {symbol}")
            return {"Signal": "Neutral"}

        df = calculate_moving_averages(df)
        df = calculate_bollinger_bands(df)
        df = calculate_macd(df)
        df = calculate_rsi(df)

        signals = []
        latest = df.iloc[-1]

        # Moving Average signal
        signals.append("Buy" if latest["short_ma"] > latest["long_ma"] else "Sell")
        # Bollinger Bands signal
        if latest["close"] < latest["Bollinger_Min"]:
            signals.append("Buy")
        elif latest["close"] > latest["Bollinger_Max"]:
            signals.append("Sell")
        # MACD signal
        signals.append("Buy" if latest["macd"] > latest["signal"] else "Sell")
        # RSI signal
        if latest["rsi"] < 30:
            signals.append("Buy")
        elif latest["rsi"] > 70:
            signals.append("Sell")

        buy_signals = signals.count("Buy")
        sell_signals = signals.count("Sell")
        final_signal = "Buy" if buy_signals > sell_signals else "Sell" if sell_signals > buy_signals else "Neutral"

        return {
            "Signal": final_signal,
            "Bollinger Min": round(latest.get("Bollinger_Min", 0), 2),
            "Bollinger Max": round(latest.get("Bollinger_Max", 0), 2),
            "Bollinger Current": round(latest.get("close", 0), 2),
            "MACD Current": round(latest.get("macd", 0), 2),
            "MACD Signal": round(latest.get("signal", 0), 2),
            "RSI Current": round(latest.get("rsi", 0), 2),
            "Short MA": round(latest.get("short_ma", 0), 2),
            "Long MA": round(latest.get("long_ma", 0), 2),
        }
    except Exception as e:
        print(f"Technical analysis error for {symbol}: {str(e)}")
        return {"Signal": "Neutral"}

# ---------------------------
# Helper for Indicator Scoring
# ---------------------------
def score_indicator(row, indicator, direction):
    """Calculate score based on indicator and direction ('Buy' or 'Sell'). 
       Returns a single float value."""
    if indicator == "Bollinger":
        result = (row["Bollinger Min"] - row["Bollinger Current"]) if direction == "Buy" else (row["Bollinger Current"] - row["Bollinger Max"])
    elif indicator == "MACD":
        result = (row["MACD Current"] - row["MACD Signal"]) if direction == "Buy" else (row["MACD Signal"] - row["MACD Current"])
    elif indicator == "RSI":
        result = (30 - row["RSI Current"]) if direction == "Buy" else (row["RSI Current"] - 70)
    elif indicator == "MA":
        result = (row["Short MA"] - row["Long MA"]) if direction == "Buy" else (row["Long MA"] - row["Short MA"])
    else:
        result = 0
    return float(result)

# ---------------------------
# Stock Analysis Function
# ---------------------------
def analyze_stock(stock, news_dict):
    symbol = stock["symbol"]
    security_id = stock["id"]
    session = get_session()
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    historical_data = []
    max_attempts = 50
    attempt = 0

    # Exponential backoff for retries on historical data
    while attempt < max_attempts:
        try:
            url = f"{BASE_URL}/market/history/security/{security_id}"
            params = {"startDate": start_date, "endDate": end_date, "size": 500}
            response = session.get(url, params=params)
            if response.status_code == 200:
                historical_data = response.json().get("content", [])
                break
            else:
                raise Exception("API Error")
        except Exception:
            attempt += 1
            time.sleep(2 ** min(attempt, 5) * 0.1 + random.random() * 0.1)
    session.close()

    if not historical_data:
        print(f"No data for {symbol}")
        return None

    buy_pressure, sell_pressure = calculate_buy_sell_pressure(historical_data)
    net_pressure = buy_pressure - sell_pressure
    technical_analysis = analyze_technical_indicators(historical_data, symbol)
    sentiment = analyze_news_sentiment(stock, news_dict)

    # Decision matrix for final signal
    buy_points, sell_points = 0, 0
    if technical_analysis["Signal"] == "Buy":
        buy_points += 2
    elif technical_analysis["Signal"] == "Sell":
        sell_points += 2
    if net_pressure > 0:
        buy_points += 1
    elif net_pressure < 0:
        sell_points += 1
    if sentiment > 0.05:
        buy_points += 1
    elif sentiment < -0.05:
        sell_points += 1

    final_signal = "Buy" if buy_points >= 3 and buy_points > sell_points else \
                   "Sell" if sell_points >= 3 and sell_points > buy_points else "Hold"
    
    try:
        return {
            "Symbol": symbol,
            "Final Signal": final_signal,
            "Sentiment Score": round(sentiment, 2),
            "Buy Pressure": int(buy_pressure),
            "Sell Pressure": int(sell_pressure),
            "Net Pressure": int(net_pressure),
            "Bollinger Min": round(technical_analysis["Bollinger Min"], 2),
            "Bollinger Max": round(technical_analysis["Bollinger Max"], 2),
            "Bollinger Current": round(technical_analysis["Bollinger Current"], 2),
            "MACD Current": round(technical_analysis["MACD Current"], 2),
            "MACD Signal": round(technical_analysis["MACD Signal"], 2),
            "RSI Current": round(technical_analysis["RSI Current"], 2),
            "Short MA": round(technical_analysis["Short MA"], 2),
            "Long MA": round(technical_analysis["Long MA"], 2),
        }
    except KeyError as e:
        return {
            "Symbol": symbol,
            "Final Signal": final_signal,
            "Sentiment Score": round(sentiment, 2),
            "Buy Pressure": int(buy_pressure),
            "Sell Pressure": int(sell_pressure),
            "Net Pressure": int(net_pressure),
        }

# ---------------------------
# Main Signal Processing using Dask
# ---------------------------
def send_signals():
    print("Starting analysis...")
    main_session = get_session()
    nepse_details = get_nepse_overall_details(main_session)
    print(f"Market Status: {nepse_details['market_status']}")
    stocks = get_listed_stocks(main_session)
    news_dict = get_news(main_session)
    main_session.close()

    if not stocks:
        print("No stocks available for analysis.")
        return

    print(f"Analyzing {len(stocks)} stocks...")

    # Create a list of delayed tasks for stock analysis
    delayed_tasks = [delayed(analyze_stock)(stock, news_dict) for stock in stocks]
    # Compute results in parallel
    results = compute(*delayed_tasks)
    # Filter out None results
    results = [res for res in results if res is not None]

    # Display progress
    for idx, result in enumerate(results, start=1):
        print(f"{idx}/{len(stocks)} : {result['Symbol']}: {result['Final Signal']} "
              f"(Sentiment: {result['Sentiment Score']}, Net Pressure: {result['Net Pressure']})")

    if not results:
        print("No valid results to process.")
        return

    results_df = pd.DataFrame(results)

    # Split results by indicator for separate Excel sheets
    indicators = ["Bollinger", "MACD", "RSI", "MA"]
    sheets = {}
    for ind in indicators:
        sheets[f"{ind}_Buy"] = results_df[results_df["Final Signal"] == "Buy"].copy()
        sheets[f"{ind}_Sell"] = results_df[results_df["Final Signal"] == "Sell"].copy()

    # Score each sheet if required columns exist
    score_requirements = {
        "Bollinger": {
            "Buy": ["Bollinger Min", "Bollinger Current"],
            "Sell": ["Bollinger Current", "Bollinger Max"]
        },
        "MACD": {
            "Buy": ["MACD Current", "MACD Signal"],
            "Sell": ["MACD Current", "MACD Signal"]
        },
        "RSI": {
            "Buy": ["RSI Current"],
            "Sell": ["RSI Current"]
        },
        "MA": {
            "Buy": ["Short MA", "Long MA"],
            "Sell": ["Short MA", "Long MA"]
        }
    }

    for ind in indicators:
        for direction in ["Buy", "Sell"]:
            sheet_name = f"{ind}_{direction}"
            df_sheet = sheets[sheet_name]
            required_cols = score_requirements[ind][direction]
            if all(col in df_sheet.columns for col in required_cols):
                score_series = df_sheet.apply(lambda row: score_indicator(row, ind, direction), axis=1)
                # Assign the score column using assign() so that the result is a Series.
                df_sheet = df_sheet.assign(Score=score_series)
                sheets[sheet_name] = df_sheet.sort_values(by="Score", ascending=False)
            else:
                print(f"Error: Missing required columns for {ind} {direction} scoring.")

    # Save results to an Excel file with separate sheets
    with pd.ExcelWriter("signals.xlsx", engine="xlsxwriter") as writer:
        results_df.to_excel(writer, sheet_name="Overall", index=False)
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

if __name__ == "__main__":
    send_signals()
