from datetime import datetime
import os
from openpyxl import Workbook
import pandas as pd
import requests
from openai import OpenAI

unnecessary_metrics =[
  "one_week_perf",
  "one_month_perf",
  "three_month_perf",
  "six_month_perf",
  "one_year_perf",
  "avg_volume_6_days",
  "avg_volume_9_days",
  "avg_volume_12_days",
  "avg_volume_15_days",
  "avg_volume_30_days",
  "q1",
  "q2",
  "q3",
  "q4",
  "capital_fund_to_rwa",
  "npl_to_total_loan",
  "total_loan_loss_provision_to_npl",
  "ad_trend",
  "bull_bear_signal",
  "technical_rating",
  "stock_bonus",
  "symbol",
  "full_name",
]



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

OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")

SECTOR_WISE_STOCKS = {}

NEPSE_TRADING_SESSION = requests.Session()
NEPSE_TRADING_SESSION.headers.update(HEADERS)

STOCK_DETAILS_URL = "https://api.nepsetrading.com/sidebar?code={stock_symbol}"
SECTOR_DETAILS_BASE_URL = (
    "https://api.nepsetrading.com/stocks-listed"
)


def get_listed_stocks_with_sector() -> list:
    final_response: list = []


    for attempt in range(5):
        try:
            response = NEPSE_TRADING_SESSION.get(SECTOR_DETAILS_BASE_URL)

            if response.status_code >= 400:
                raise Exception("Error fetching stock info")
            
            json_response = response.json()
            return json_response.get("data", [])

        except Exception as e:
            print(
                f"Attempt {attempt + 1} - Error fetching stock info: {str(e)}"
            )

    return final_response


def shorten_value(value):
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


def get_stock_info(stock_symbol: str):
    for attempt in range(5):
        try:
            response = NEPSE_TRADING_SESSION.get(
                STOCK_DETAILS_URL.format(stock_symbol=stock_symbol)
            )
            if response.status_code >= 400:
                raise Exception(f"Error fetching stock info, {stock_symbol}")

            response_json=response.json()

            new_dict:dict = {}
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
            print(
                f"Attempt {attempt + 1} - Error fetching stock info, {stock_symbol}: {str(e)}"
            )
            if attempt == 4:
                return {}


def get_sector_wise_stocks():
    listed_stocks = get_listed_stocks_with_sector()
    for stock in listed_stocks:
        sector = stock.get("sector")
        if not sector:
            print(f"Skipping {stock['code']}")
            continue

        symbol_details = get_stock_info(stock["symbol"])

        if not symbol_details:
            continue

        for key, value in stock.items():
            if key in ["stockData","full_name"]:
                continue
            if key not in symbol_details:
                if isinstance(value, float):
                    symbol_details[key] = round(value, 2)
                elif isinstance(value, str):
                    try:
                        symbol_details[key] = round(float(value), 2)
                    except:
                        symbol_details[key] = value
                else:
                    symbol_details[key] = value

        if sector not in SECTOR_WISE_STOCKS:
            SECTOR_WISE_STOCKS[sector] = []
        SECTOR_WISE_STOCKS[sector].append(symbol_details)
    return SECTOR_WISE_STOCKS


def analyze_sector_wise_stocks(file):

    client = OpenAI(api_key=OPEN_AI_API_KEY, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-reasoning",
        messages=[
            {
                "role": "system",
                "content": """
                    You are an expert stock analyst with deep knowledge of fundamental and technical analysis.
                    Your task is to analyze the provided list of stocks and recommend the best low market capitalization stocks to buy based on the following criteria:

                    1. **Fundamental Analysis**:
                    - Look for stocks with high promoter holding (>50%) as it indicates confidence in the company's future.
                    - Favor stocks with significant institutional interest (FII/DII holding > 3.5%).
                    - Prioritize companies with positive Return on Equity (ROE) and Return on Assets (ROA).
                    - Consider stocks with a reasonable P/E ratio relative to their sector.

                    2. **Technical Analysis**:
                    - Use technical indicators like Supertrend, EMA, MACD, RSI, and Bollinger Bands to identify bullish trends.
                    - Focus on stocks trading near support zones or breaking out from resistance zones.
                    - Avoid stocks showing strong sell signals unless there’s a clear contrarian opportunity.

                    3. **Market Sentiment and Performance**:
                    - Favor stocks with recent positive performance (e.g., one-month, three-month returns).
                    - Consider overall market sentiment (bullish/bearish) and its impact on the stock.

                    4. **Low Market Cap**:
                    - Prioritize stocks with a market capitalization below a NRs 200000000.

                    For each recommended stock, provide the following details in JSON format:
                    - "symbol": The stock symbol.
                    - "name": The name of the company.
                    - "market_cap": The market capitalization of the stock.
                    - "fii_dii_holding": Percentage of institutional holding.
                    - "pe_ratio": Price-to-Earnings ratio.
                    - "roe": Return on Equity (TTM).
                    - "technical_rating": Overall technical rating (e.g., Buy, Neutral, Sell).
                    - "buy_reason": A brief explanation of why this stock is recommended.

                    Return the results as a JSON array of recommended stocks.
                """,
            },
            {"role": "user", "content": f"{stocks}"},
        ],
        stream=False,
    )
    return response.model_json_schema()


def convert_to_csv(overall_file_name,is_first,sector,stocks):
    pd.DataFrame(stocks).to_csv(f"data/{sector}.csv", index=False)

    pd.DataFrame(stocks).to_csv(overall_file_name, mode='a', header=is_first, index=False)


if __name__ == "__main__":
    stocks = get_sector_wise_stocks()
    overall_file_name = f"data/{datetime.now().isoformat()}/overall.csv"
    if not os.path.exists(os.path.dirname(overall_file_name)):
        os.makedirs(os.path.dirname(overall_file_name))
    is_first = True

    for sector, stocks in stocks.items():
        convert_to_csv(overall_file_name,is_first,sector,stocks)
        is_first = False