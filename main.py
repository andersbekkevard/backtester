import numpy as np
import matplotlib as plt
import pandas as pd
import yfinance as yf


def calculate_correlation(stock_a: pd.DataFrame, stock_b: pd.DataFrame) -> float:
    """Calculate correlation between two stocks. Assumes both datasets are complete"""
    ret_a = stock_a["Return"]
    ret_b = stock_b["Return"]
    return ret_a.corr(ret_b)


def manual_correlation(stock_a: pd.DataFrame, stock_b: pd.DataFrame) -> float:
    """
    Manually calculates correlation between two stocks. Assumes both datasets are complete
    (Just had to prove that i know my statistics)
    """
    ret_a = stock_a["Return"].to_numpy()[1:]
    ret_b = stock_b["Return"].to_numpy()[1:]
    std_a = np.sqrt(((ret_a - ret_a.mean()) ** 2).mean())
    std_b = np.sqrt(((ret_b - ret_b.mean()) ** 2).mean())
    cov = ((ret_a - ret_a.mean()) * (ret_b - ret_b.mean())).mean()
    return cov / (std_a * std_b)


def manual_regression(predicted: pd.DataFrame, predictor: pd.DataFrame) -> tuple:
    """Calculates regression coefficients by using the day ahead returns of comparable stock"""
    y = predicted["Return"].to_numpy()[1:-1]
    x = predictor["Return"].shift(1).to_numpy()[2:]
    sxy = ((x - x.mean()) * y).sum()
    sxx = ((x - x.mean()) ** 2).sum()
    betaHat = sxy / sxx
    alphaHat = y.mean() - betaHat * x.mean()
    return alphaHat, betaHat


AAPL_PATH = "backtester/data/aapl.csv"
MSFT_PATH = "backtester/data/msft.csv"

aapl = pd.read_csv("backtester/data/aapl.csv")
msft = pd.read_csv("backtester/data/msft.csv")

alpha, beta = manual_regression(aapl, msft)
print(beta)

total_return = aapl["Log"].sum()
momentum_return = aapl[msft["Return"].shift(1) > 0]["Log"].sum()
print(total_return)
print(momentum_return)
