import pandas as pd
import numpy as np
import yfinance as yf

AAPL_PATH = "backtester/data/aapl.csv"
MSFT_PATH = "backtester/data/msft.csv"

START_DATE = "2000-01-01"
END_DATE = "2010-01-01"


def fetch_stock_data(symbol: str, start_date, end_date) -> pd.DataFrame:
    """Fetch simple stock data using yfinance"""
    # Download df
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
    # Delete unused columns
    del df["Dividends"], df["Stock Splits"], df["High"], df["Low"], df["Open"]
    # Calculate returns
    df["Return"] = df["Close"] / df["Close"].shift(1) - 1
    df["Log"] = np.log(df["Close"] / df["Close"].shift(1))
    # Change format of date column
    df = df.reset_index()
    df["Date"] = df["Date"].dt.date
    df = df.set_index("Date")
    return df


aapl = fetch_stock_data("AAPL", START_DATE, END_DATE)
msft = fetch_stock_data("MSFT", START_DATE, END_DATE)

with open(AAPL_PATH, "w", newline="") as f:
    f.write(aapl.to_csv())

with open(MSFT_PATH, "w", newline="") as f:
    f.write(msft.to_csv())
