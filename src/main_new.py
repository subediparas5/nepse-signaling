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
    try:
        with requests.Session() as session:
            index_response = session.get(nepse_index_url)
            status_response = session.get(market_status_url)
            if index_response.status_code >= 400 or status_response.status_code >= 400:
                raise Exception("Error fetching NEPSE details")
            return {
                "nepse_index": index_response.json(),
                "market_status": status_response.json()["isOpen"],
            }
    except Exception as e:
        print(f"Error fetching NEPSE details: {str(e)}")
        return {"nepse_index": {}, "market_status": False}


# Step 2: Get Listed Stocks


def get_listed_stocks():
    url = "https://nepalstock.onrender.com/security?nonDelisted=true"
    try:
        with requests.Session() as session:
            response = session.get(url)
            if response.status_code >= 400:
                raise Exception("Error fetching listed stocks")
            return [stock for stock in response.json() if stock.get("activeStatus") == "A"]
    except Exception as e:
        print(f"Error fetching listed stocks: {str(e)}")
        return []


# Step 3: Fetch News Disclosures


def get_news():
    url = "https://nepalstock.onrender.com/news/companies/disclosure"
    try:
        with requests.Session() as session:
            response = session.get(url)
            if response.status_code >= 400:
                raise Exception("Error fetching news disclosures")
            news = response.json()
            news_dict = {}
            for item in news.get("companyNews", []):
                symbol = item.get("symbol")
                if symbol:
                    news_dict.setdefault(symbol, []).append(item)
            return news_dict
    except Exception as e:
        print(f"Error fetching news disclosures: {str(e)}")
        return {}


# Helper Function: Calculate Buy/Sell Pressure


def calculate_buy_sell_pressure(historical_data):
    try:
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
    except Exception as e:
        print(f"Error calculating buy/sell pressure: {str(e)}")
        return 0, 0


# Helper Function: Analyze News Sentiment


def analyze_news_sentiment(stock, news_dict):
    try:
        symbol = stock["symbol"]
        stock_news = news_dict.get(symbol, [])
        if not stock_news:
            return 0
        sentiment_scores = []
        for news_item in stock_news:
            # Consider recent news more important
            days_old = (datetime.now() - datetime.strptime(news_item["publishedDate"], "%Y-%m-%d")).days
            weight = max(0, 1 - days_old / 30)  # Weight decays over 30 days
            text = f"{news_item.get('newsHeadline', '')} {news_item.get('remarks', '')}"
            analysis = TextBlob(text)
            sentiment_scores.append(analysis.sentiment.polarity * weight)
        if not sentiment_scores:
            return 0
        return sum(sentiment_scores) / len(sentiment_scores)
    except Exception as e:
        print(f"Sentiment error for {symbol}: {str(e)}")
        return 0


# Enhanced Technical Analysis


def calculate_moving_averages(df: pd.DataFrame, short_window: int = 5, long_window: int = 20) -> pd.DataFrame:
    try:
        df["short_ma"] = df["close"].rolling(window=short_window, min_periods=1).mean()
        df["long_ma"] = df["close"].rolling(window=long_window, min_periods=1).mean()
        return df
    except Exception as e:
        print(f"Error calculating moving averages: {str(e)}")
        return df


def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    required_columns = {"code", "date", "close"}
    if not required_columns.issubset(df.columns):
        print(f"Missing required columns for Bollinger Bands: {df.columns}")
        return df  # Return unchanged if columns are missing

    # Ensure numeric types and handle duplicates
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.sort_values(["code", "date"]).drop_duplicates(subset=["code", "date"], keep="last")

    # Calculate Bollinger Bands
    grouped = df.groupby("code", group_keys=False)
    middle_band = grouped["close"].rolling(window=window).mean()
    std_dev = grouped["close"].rolling(window=window).std()

    # Align indices explicitly
    df["middle_band"] = middle_band.reset_index(level=0, drop=True)
    df["std_dev"] = std_dev.reset_index(level=0, drop=True)

    df["Bollinger_Min"] = df["middle_band"] - (num_std * df["std_dev"])
    df["Bollinger_Max"] = df["middle_band"] + (num_std * df["std_dev"])
    return df


def calculate_macd(
    df: pd.DataFrame, short_window: int = 12, long_window: int = 26, signal_window: int = 9
) -> pd.DataFrame:
    try:
        df["ema_short"] = df["close"].ewm(span=short_window, adjust=False).mean()
        df["ema_long"] = df["close"].ewm(span=long_window, adjust=False).mean()
        df["macd"] = df["ema_short"] - df["ema_long"]
        df["signal"] = df["macd"].ewm(span=signal_window, adjust=False).mean()
        df["histogram"] = df["macd"] - df["signal"]
        return df
    except Exception as e:
        print(f"Error calculating MACD: {str(e)}")
        return df


def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    try:
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        return df
    except Exception as e:
        print(f"Error calculating RSI: {str(e)}")
        return df


def analyze_technical_indicators(historical_data, symbol):
    try:
        if not historical_data:
            print(f"No historical data for {symbol}")
            return {"Signal": "Neutral"}

        df = pd.DataFrame(historical_data)
        if df.empty:
            print(f"Empty DataFrame for {symbol}")
            return {"Signal": "Neutral"}

        # Standardize column names
        df.rename(columns={"closePrice": "close", "businessDate": "date"}, inplace=True, errors="ignore")
        df["code"] = symbol  # Add symbol as code

        # Check for required columns
        required_cols = {"close", "date", "code"}
        if not required_cols.issubset(df.columns):
            print(f"Missing required columns in {symbol}: {df.columns}")
            return {"Signal": "Neutral"}

        # Ensure numeric types and handle duplicates
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.sort_values(["code", "date"]).drop_duplicates(subset=["code", "date"], keep="last")

        # Ensure sufficient data for analysis
        if len(df) < 20:
            print(f"Insufficient data for {symbol}: {len(df)} rows")
            return {"Signal": "Neutral"}

        # Calculate indicators
        df = calculate_moving_averages(df)
        df = calculate_bollinger_bands(df)
        df = calculate_macd(df)
        df = calculate_rsi(df)

        # Generate signals
        signals = []
        latest = df.iloc[-1]

        # Moving Averages
        if "short_ma" in latest and "long_ma" in latest:
            if latest["short_ma"] > latest["long_ma"]:
                signals.append("Buy")
            else:
                signals.append("Sell")

        # Bollinger Bands
        if "Bollinger_Min" in latest and "Bollinger_Max" in latest and "close" in latest:
            if latest["close"] < latest["Bollinger_Min"]:
                signals.append("Buy")
            elif latest["close"] > latest["Bollinger_Max"]:
                signals.append("Sell")

        # MACD
        if "macd" in latest and "signal" in latest:
            if latest["macd"] > latest["signal"]:
                signals.append("Buy")
            else:
                signals.append("Sell")

        # RSI
        if "rsi" in latest:
            if latest["rsi"] < 30:
                signals.append("Buy")
            elif latest["rsi"] > 70:
                signals.append("Sell")

        # Determine final signal
        buy_signals = signals.count("Buy")
        sell_signals = signals.count("Sell")
        final_signal = "Neutral"
        if buy_signals > sell_signals:
            final_signal = "Buy"
        elif sell_signals > buy_signals:
            final_signal = "Sell"

        return {
            "Signal": final_signal,
            "Bollinger_Min": latest.get("Bollinger_Min", 0),
            "Bollinger_Max": latest.get("Bollinger_Max", 0),
            "Bollinger_Current": latest.get("close", 0),
            "MACD_Current": latest.get("macd", 0),
            "MACD_Signal": latest.get("signal", 0),
            "RSI_Current": latest.get("rsi", 0),
            "Short_MA": latest.get("short_ma", 0),
            "Long_MA": latest.get("long_ma", 0),
        }
    except Exception as e:
        print(f"Technical analysis error for {symbol}: {str(e)}")
        return {"Signal": "Neutral"}


# Improved Stock Analysis with better decision logic


def analyze_stock(stock, news_dict):
    try:
        security_id = stock["id"]
        symbol = stock["symbol"]

        # Fetch historical data
        historical_data = []
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        max_attempt = 50
        attempt = 0

        while True:
            if attempt >= max_attempt:
                break
            try:
                url = f"https://nepalstock.onrender.com/market/history/security/{security_id}"
                params = {"startDate": start_date, "endDate": end_date, "size": 500}
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    historical_data = response.json().get("content", [])
                    break
                else:
                    raise Exception("API Error")
            except Exception:
                attempt += 1
                time.sleep(random.randint(1, 3))

        if not historical_data:
            print(f"No data for {symbol}")
            return None

        # Calculate metrics
        buy_pressure, sell_pressure = calculate_buy_sell_pressure(historical_data)
        net_pressure = buy_pressure - sell_pressure
        technical_analysis = analyze_technical_indicators(historical_data, symbol)
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
            "Final Signal": signal,
            "Sentiment Score": round(sentiment, 2),
            "Buy Pressure": int(buy_pressure),
            "Sell Pressure": int(sell_pressure),
            "Net Pressure": int(net_pressure),
            "Bollinger Min": round(technical_analysis["Bollinger_Min"], 2),
            "Bollinger Max": round(technical_analysis["Bollinger_Max"], 2),
            "Bollinger Current": round(technical_analysis["Bollinger_Current"], 2),
            "MACD Current": round(technical_analysis["MACD_Current"], 2),
            "MACD Signal": round(technical_analysis["MACD_Signal"], 2),
            "RSI Current": round(technical_analysis["RSI_Current"], 2),
            "Short MA": round(technical_analysis["Short_MA"], 2),
            "Long MA": round(technical_analysis["Long_MA"], 2),
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
            if not result:
                continue
            if all(key in result for key in ["Symbol", "Final Signal"]):
                results.append(result)
                print(
                    f"{len(results)}/{len(stocks)} : "
                    f"{result['Symbol']}: {result['Final Signal']} "
                    f"(Sentiment: {result['Sentiment Score']}, "
                    f"Net Pressure: {result['Net Pressure']})"
                )
            else:
                print(f"Skipping invalid result for stock: {result.get('Symbol', 'Unknown')}")

    # Convert results to DataFrame
    if not results:
        return
    results_df = pd.DataFrame(results)

    # Create separate sheets for each indicator
    bollinger_buy = results_df[results_df["Final Signal"] == "Buy"].copy()
    bollinger_sell = results_df[results_df["Final Signal"] == "Sell"].copy()
    macd_buy = results_df[results_df["Final Signal"] == "Buy"].copy()
    macd_sell = results_df[results_df["Final Signal"] == "Sell"].copy()
    rsi_buy = results_df[results_df["Final Signal"] == "Buy"].copy()
    rsi_sell = results_df[results_df["Final Signal"] == "Sell"].copy()
    ma_buy = results_df[results_df["Final Signal"] == "Buy"].copy()
    ma_sell = results_df[results_df["Final Signal"] == "Sell"].copy()

    # Check for required columns before scoring
    if "Bollinger Min" in bollinger_buy.columns and "Bollinger Current" in bollinger_buy.columns:
        bollinger_buy["Score"] = bollinger_buy["Bollinger Min"] - bollinger_buy["Bollinger Current"]
    else:
        print("Error: Missing required columns for Bollinger Buy scoring.")

    if "Bollinger Current" in bollinger_sell.columns and "Bollinger Max" in bollinger_sell.columns:
        bollinger_sell["Score"] = bollinger_sell["Bollinger Current"] - bollinger_sell["Bollinger Max"]
    else:
        print("Error: Missing required columns for Bollinger Sell scoring.")

    if "MACD Current" in macd_buy.columns and "MACD Signal" in macd_buy.columns:
        macd_buy["Score"] = macd_buy["MACD Current"] - macd_buy["MACD Signal"]
    else:
        print("Error: Missing required columns for MACD Buy scoring.")

    if "MACD Current" in macd_sell.columns and "MACD Signal" in macd_sell.columns:
        macd_sell["Score"] = macd_sell["MACD Signal"] - macd_sell["MACD Current"]
    else:
        print("Error: Missing required columns for MACD Sell scoring.")

    if "RSI Current" in rsi_buy.columns:
        rsi_buy["Score"] = 30 - rsi_buy["RSI Current"]
    else:
        print("Error: Missing required columns for RSI Buy scoring.")

    if "RSI Current" in rsi_sell.columns:
        rsi_sell["Score"] = rsi_sell["RSI Current"] - 70
    else:
        print("Error: Missing required columns for RSI Sell scoring.")

    if "Short MA" in ma_buy.columns and "Long MA" in ma_buy.columns:
        ma_buy["Score"] = ma_buy["Short MA"] - ma_buy["Long MA"]
    else:
        print("Error: Missing required columns for Moving Average Buy scoring.")

    if "Short MA" in ma_sell.columns and "Long MA" in ma_sell.columns:
        ma_sell["Score"] = ma_sell["Long MA"] - ma_sell["Short MA"]
    else:
        print("Error: Missing required columns for Moving Average Sell scoring.")

    bollinger_buy = bollinger_buy.sort_values(by="Score", ascending=False)
    bollinger_sell = bollinger_sell.sort_values(by="Score", ascending=False)
    macd_buy = macd_buy.sort_values(by="Score", ascending=False)
    macd_sell = macd_sell.sort_values(by="Score", ascending=False)
    rsi_buy = rsi_buy.sort_values(by="Score", ascending=False)
    rsi_sell = rsi_sell.sort_values(by="Score", ascending=False)
    ma_buy = ma_buy.sort_values(by="Score", ascending=False)
    ma_sell = ma_sell.sort_values(by="Score", ascending=False)

    # Save to Excel
    with pd.ExcelWriter("signals.xlsx", engine="xlsxwriter") as writer:
        results_df.to_excel(writer, sheet_name="Overall", index=False)
        bollinger_buy.to_excel(writer, sheet_name="Bollinger_Buy", index=False)
        bollinger_sell.to_excel(writer, sheet_name="Bollinger_Sell", index=False)
        macd_buy.to_excel(writer, sheet_name="MACD_Buy", index=False)
        macd_sell.to_excel(writer, sheet_name="MACD_Sell", index=False)
        rsi_buy.to_excel(writer, sheet_name="RSI_Buy", index=False)
        rsi_sell.to_excel(writer, sheet_name="RSI_Sell", index=False)
        ma_buy.to_excel(writer, sheet_name="MA_Buy", index=False)
        ma_sell.to_excel(writer, sheet_name="MA_Sell", index=False)


if __name__ == "__main__":
    send_signals()
