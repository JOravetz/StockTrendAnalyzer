#!/usr/bin/python3

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal

from scipy import signal
from matplotlib import style, gridspec
from alpaca_trade_api.rest import REST, TimeFrame
from config import API_KEY_ID, SECRET_KEY_ID, BASE_URL

from helper_classes import (
    StockHelper,
    AvoAttributesCalculator,
    CircleCalculator,
    PriceCalculator,
    Plotter,
)

# Magic number replaced with a constant
WINDOW_RATIO = 22.90

style.use("dark_background")

# Adjust display options
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

API_KEY_ID    = os.getenv("APCA_API_KEY_ID")
SECRET_KEY_ID = os.getenv("APCA_API_SECRET_KEY")
BASE_URL      = os.getenv("APCA_API_BASE_URL")

rest_api = REST(API_KEY_ID, SECRET_KEY_ID, BASE_URL)

# Create instances of the classes
stock_helper = StockHelper(rest_api)
avo_attributes_calculator = AvoAttributesCalculator()
circle_calculator = CircleCalculator()
price_calculator = PriceCalculator()
plotter = Plotter()

# Set up argparse for command line arguments
parser = argparse.ArgumentParser(
    description="Process stock market data from Alpaca Markets"
)
parser.add_argument("--command", type=str, help="The command to run")
parser.add_argument(
    "--symbol", type=str, required=True, help="The stock symbol"
)
parser.add_argument(
    "--tail",
    type=int,
    default=252,
    nargs="?",
    help="The tail value (default: 252)",
)
parser.add_argument(
    "--window", type=int, default=None, nargs="?", help="The window value"
)
parser.add_argument(
    "--factor",
    type=float,
    default=0.200,
    nargs="?",
    help="The factor value (default: 0.20)",
)
parser.add_argument(
    "--plot",
    type=int,
    default=1,
    nargs="?",
    help="The plot value (default: 1)",
)

args = parser.parse_args()

symbol = args.symbol.upper()
tail = args.tail
window = args.window if args.window else round(tail // WINDOW_RATIO)
factor = args.factor
plot_switch = args.plot

if symbol is None:
    raise ValueError("Must enter symbol eg: TQQQ")

# Find company name for the given stock symbol
co_name = stock_helper.find_co_name(symbol)

# Generate DataFrame with NYSE trading days for the specified tail length
df, start_date, end_date, ndays = stock_helper.nyse_trading_days_dataframe(tail)

# Fetch historical data for the given symbol from Alpaca API
historical_data = stock_helper.fetch_alpaca_data(
    symbol, start_date, end_date, tail
)

# Combine the df and bars DataFrames, keeping only rows with values in the close column
df = df.merge(historical_data, left_index=True, right_index=True, how="inner")

# Remove any blank rows from the combined DataFrame
df.dropna(axis=0, inplace=True)

# Fetch the current price
current_price = rest_api.get_latest_trade(symbol).price
ns = df.shape[0]
df.loc[end_date] = current_price
df = df.tail(ns)

# Get the last row's index and close value
# last_index = df.index[-1]
# last_close = df.loc[last_index, "close"]

# Update the DataFrame with current price, calculate returns and convert to cumulative percent change
# df = price_calculator.update_current_price(df, current_price, last_close, last_index)

df = price_calculator.calculate_returns(df)
df = price_calculator.convert_to_cumulative_percent_change(df)

# Set the first daily return to zero
daily_returns = df.daily_returns.values
daily_returns[0] = 0.0

# Get the percent change for the current day
percent_change_today = df["daily_returns"].iloc[-1]

# Create a temporary DataFrame to store AVO attributes
df_temp = pd.DataFrame(
    index=[symbol],
    columns=[
        "nsamps",
        "filter",
        "mean",
        "std_dev",
        "last_vel",
        "detect_value",
        "factor",
        "trend_diff",
        "action",
        "action_price",
        "price",
        "isamp_ago",
        "percent",
    ],
)

# Compute AVO attributes, including filter_close, final_min, final_max, velocity, and df_temp
(
    df["filter_close"],
    final_min,
    final_max,
    df["velocity"],
    df_temp,
) = circle_calculator.compute_circles(symbol, df, window, factor, current_price, df_temp)

# Print the last 10 rows of the DataFrame and the df_temp DataFrame
print(df.tail(10))
print(" ")
print(df_temp)

# Add a column of zeros to the DataFrame
df["zero"] = 0.0

# Plot the graph based on the computed data
plotter.plot_graph(
    df,
    final_min,
    final_max,
    symbol,
    co_name,
    percent_change_today,
    current_price,
    plot_switch,
)


if __name__ == "__main__":
    parser.parse_args()
