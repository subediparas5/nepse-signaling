import os
import logging
from datetime import datetime
import requests
import pandas as pd
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
if not OPEN_AI_API_KEY:
    raise ValueError("OPEN_AI_API_KEY environment variable is not set.")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:135.0) Gecko/20100101 Firefox/135.0",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Origin": "https://www.nepsetrading.com",
    "Connection": "keep-alive",
    "Referer": "https://www.nepsetrading.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "Priority": "u=0",
    "TE": "trailers",
}

NEPSE_TRADING_SESSION = requests.Session()
NEPSE_TRADING_SESSION.headers.update(HEADERS)

STOCK_DETAILS_URL = "https://api.nepsetrading.com/sidebar?code={stock_symbol}"
SECTOR_DETAILS_BASE_URL = "https://api.nepsetrading.com/stocks-listed"
FUNDAMENTAL_DETAILS_URL = "https://api.nepsetrading.com/recent-report?"

unnecessary_metrics = [
    "share_float",  # Not directly relevant for fundamental/technical analysis
    "latesttransactionprice",  # Redundant with LTP
    # "volume",  # Already covered by avg_volume_3_days
    "beta_yearly",  # Not a priority for low-cap stock selection
    "support_zone_lower",  # Technical analysis can be derived from other indicators
    "support_zone_upper",  # Technical analysis can be derived from other indicators
    "resistance_zone_lower",  # Technical analysis can be derived from other indicators
    "resistance_zone_upper",  # Technical analysis can be derived from other indicators
    "macd_signal",  # Covered by technical_rating
    "rsi_signal",  # Covered by technical_rating
    "bb_signal",  # Covered by technical_rating
    "fib_signal",  # Covered by technical_rating
    "fib_range",  # Covered by technical_rating
    "supertrend_signal",  # Covered by technical_rating
    "ema_signal",  # Covered by technical_rating
    "sar_signal",  # Covered by technical_rating
    "market_trend",  # Covered by market_sentiment
    "trade_signal",  # Covered by technical_rating
    "market_sentiment",  # Redundant with other sentiment indicators
    "ma_signal",  # Covered by technical_rating
    "trend_confirmation",  # Covered by technical_rating
    "obv_price_divergence",  # Covered by technical_rating
    "obv_breakout",  # Covered by technical_rating
    "daily_volatility_rs",  # Not a priority for low-cap stock selection
    "weekly_volatility_rs",  # Not a priority for low-cap stock selection
    "monthly_volatility_rs",  # Not a priority for low-cap stock selection
    "avg_volume_3_days",  # Redundant with volume
    "week_52_high",  # Not directly relevant for analysis
    "week_52_low",  # Not directly relevant for analysis
    "divident_yeild",  # Typo, and not a priority for low-cap stocks
    "eps",  # Redundant with eps_ttm
    # "bv",  # Redundant with other metrics
    "dpps",  # Not directly relevant for analysis
    # "npl",  # Not directly relevant for analysis
    "ltp",  # Redundant with latesttransactionprice
]

def shorten_value(value: str) -> str:
    if "Below" in value:
        value = value.replace("Below", "<")
    elif "Above" in value:
        value = value.replace("Above", ">")
    elif "Neutral" in value:
        value = value.replace("Neutral", "-")
    elif "Strong" in value:
        value = value.replace("Strong", "S")
    elif "No Breakout" == value:
        value = value.replace("No Breakout", "NO")
    elif "Confirmed" == value:
        value = value.replace("Confirmed", "CONF")
    if "Bearish" in value:
        value = value.replace("Bearish", "BEAR")
    elif "Bullish" in value:
        value = value.replace("Bullish", "BULL")
    if "Sell" in value:
        value = value.replace("Sell", "SL")
    elif "Buy" in value:
        value = value.replace("Buy", "BY")
    return value

# Functions
def get_listed_stocks_with_sector() -> list:
    for attempt in range(5):
        try:
            response = NEPSE_TRADING_SESSION.get(SECTOR_DETAILS_BASE_URL)
            response.raise_for_status()
            return response.json().get("data", [])
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} - Error fetching stock info: {str(e)}")
            logger.error(f"Response: {response.text if 'response' in locals() else 'No response'}")
    return []

def get_fundamental_details() -> list:
    for attempt in range(5):
        try:
            response = NEPSE_TRADING_SESSION.get(FUNDAMENTAL_DETAILS_URL)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} - Error fetching fundamental data: {str(e)}")
            logger.error(f"Response: {response.text if 'response' in locals() else 'No response'}")
    return []

def get_stock_info(stock_symbol: str) -> dict:
    for attempt in range(5):
        try:
            response = NEPSE_TRADING_SESSION.get(STOCK_DETAILS_URL.format(stock_symbol=stock_symbol))
            response.raise_for_status()
            response_json = response.json()

            new_dict = {}
            for key, value in response_json.items():
                if key in unnecessary_metrics:
                    continue
                if isinstance(value, float):
                    new_dict[key] = round(value, 2)
                elif isinstance(value, str):
                    try:
                        new_dict[key] = round(float(value), 2)
                    except:
                        new_dict[key] = shorten_value(value)
                else:
                    new_dict[key] = value
            return new_dict
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} - Error fetching stock info, {stock_symbol}: {str(e)}")
            if attempt == 4:
                return {}

def join_fundamental_and_technical_data():
    fundamental_data = get_fundamental_details()
    combined_data = []

    for stock in fundamental_data:
        stock_symbol = stock.get("symbol")
        if not stock_symbol:
            continue

        stock_details = get_stock_info(stock_symbol)
        if not stock_details:
            continue

        # Merge fundamental and technical data
        for key, value in stock.items():
            if key not in stock_details:
                stock_details[key] = value

        combined_data.append(stock_details)

    return combined_data

def get_sector_wise_stocks():
    sector_wise_stocks = {}
    listed_stocks = get_listed_stocks_with_sector()
    combined_data = join_fundamental_and_technical_data()

    for stock in listed_stocks:
        sector = stock.get("sector")
        if not sector:
            logger.info(f"Skipping {stock['code']}")
            continue

        symbol_details = get_stock_info(stock["symbol"])
        if not symbol_details:
            continue

        # Merge with fundamental data
        for combined_stock in combined_data:
            if combined_stock.get("symbol") == stock["symbol"]:
                for key, value in combined_stock.items():
                    if key not in symbol_details:
                        symbol_details[key] = value

        if sector not in sector_wise_stocks:
            sector_wise_stocks[sector] = []
        sector_wise_stocks[sector].append(symbol_details)

    return sector_wise_stocks

def analyze_sector_wise_stocks(stocks):
    return "Analysis not implemented yet."
    client = OpenAI(api_key=OPEN_AI_API_KEY, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-reasoning",
        messages=[
            {
                "role": "system",
                "content": """
                    You are an expert stock analyst specializing in identifying undervalued, low market capitalization stocks with strong growth potential. Your task is to analyze the provided list of stocks and recommend the best stocks to buy based on the following criteria:

                    **1. Fundamental Analysis**:
                    - **Promoter Holding**: Prefer stocks with high promoter holding (>50%), indicating strong insider confidence.
                    - **Institutional Interest**: Look for significant FII/DII holding (>3.5%), signaling institutional trust.
                    - **Profitability**: Prioritize stocks with positive Return on Equity (ROE) and Return on Assets (ROA).
                    - **Valuation**: Favor stocks with a reasonable Price-to-Earnings (P/E) ratio relative to their sector.

                    **2. Technical Analysis**:
                    - **Trend Indicators**: Use Supertrend, EMA, MACD, RSI, and Bollinger Bands to identify bullish trends.
                    - **Support/Resistance**: Focus on stocks trading near support zones or breaking out of resistance zones.
                    - **Avoid Sell Signals**: Exclude stocks with strong sell signals unless there is a compelling contrarian opportunity.

                    **3. Market Sentiment**:
                    - **Recent Performance**: Favor stocks with positive one-month and three-month returns.
                    - **Market Sentiment**: Consider overall market conditions (bullish/bearish) and their impact on the stock.

                    **4. Low Market Cap**:
                    - **Market Capitalization**: Prioritize stocks with a market capitalization below NRs 200,000,000.

                    **Output Format**:
                    For each recommended stock, provide the following details in JSON format:
                    - "symbol": Stock symbol.
                    - "name": Company name.
                    - "market_cap": Market capitalization.
                    - "fii_dii_holding": Percentage of institutional holding.
                    - "pe_ratio": Price-to-Earnings ratio.
                    - "roe": Return on Equity (TTM).
                    - "technical_rating": Overall technical rating (e.g., Buy, Neutral, Sell).
                    - "buy_reason": A concise explanation of why this stock is recommended.

                    Return the results as a JSON array of recommended stocks.
                """,
            },
            {"role": "user", "content": f"{stocks}"},
        ],
        stream=False,
    )
    return response.choices[0].message.content

def convert_to_csv(overall_file_name, is_first, sector, stocks):
    sector_file_path = os.path.join("data", f"{sector}.csv")
    pd.DataFrame(stocks).to_csv(sector_file_path, index=False)

    mode = 'w' if is_first else 'a'
    header = is_first
    pd.DataFrame(stocks).to_csv(overall_file_name, mode=mode, header=header, index=False)

if __name__ == "__main__":
    stocks = get_sector_wise_stocks()
    output_dir = os.path.join("data", datetime.now().isoformat())
    os.makedirs(output_dir, exist_ok=True)
    overall_file_name = os.path.join(output_dir, "overall.csv")

    is_first = True
    for sector, sector_stocks in stocks.items():
        convert_to_csv(overall_file_name, is_first, sector, sector_stocks)
        is_first = False

    # Analyze stocks using OpenAI
    analysis_result = analyze_sector_wise_stocks(stocks)
    logger.info(f"Analysis Result: {analysis_result}")
