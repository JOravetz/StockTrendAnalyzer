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

from helper_functions import find_co_name, nyse_trading_days_dataframe, fetch_alpaca_data, compute_avo_attributes, group_consecutives, compute_trend_and_filter, calculate_velocity, calculate_min_max, calculate_action_points, compute_circles, update_current_price, calculate_returns, convert_to_cumulative_percent_change, plot_graph

rest_api = REST(API_KEY_ID, SECRET_KEY_ID, BASE_URL)

style.use("dark_background")

# Adjust display options
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

parser = argparse.ArgumentParser(
    description="Process stock market data from Alpaca Markets"
)
parser.add_argument("--command", type=str, help="The command to run")
parser.add_argument("--symbol", type=str, help="The stock symbol")
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
    default=0.20,
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
window = args.window if args.window else round(tail // 22.90)
factor = args.factor
plot_switch = args.plot

if symbol is None:
    print("Must enter symbol eg: TQQQ -> exiting")
    sys.exit(0)

co_name = find_co_name(symbol)

df, start_date, end_date, ndays = nyse_trading_days_dataframe(tail)
historical_data = fetch_alpaca_data(
    symbol, start_date, end_date, tail, rest_api
)

# Combine the df and bars DataFrames, keeping only rows with values in the close column
df = df.merge(historical_data, left_index=True, right_index=True, how="inner")

# Remove any blank rows from the combined DataFrame
df.dropna(axis=0, inplace=True)

# Fetch the current price
current_price = rest_api.get_latest_trade(symbol).price

# Get the last row's index and close value
last_index = df.index[-1]
last_close = df.loc[last_index, "close"]

df = update_current_price(df, current_price, last_close, last_index)
df = calculate_returns(df)
df = convert_to_cumulative_percent_change(df)

daily_returns = df.daily_returns.values
daily_returns[0] = 0.0

percent_change_today = df["daily_returns"].iloc[-1]

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

(
    df["filter_close"],
    final_min,
    final_max,
    df["velocity"],
    df_temp,
) = compute_circles(symbol, df, window, factor, current_price, df_temp)

df["zero"] = 0.0

plot_graph(df, final_min, final_max, symbol, co_name, percent_change_today, current_price, plot_switch)

df.drop(["zero"], axis=1, inplace=True)
print(df.tail(10))
print(" ")
print(df_temp)
