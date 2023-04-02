#!/usr/bin/python3

import os
import re
import sys
import argparse
import subprocess
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal

from scipy import signal
from matplotlib import style, gridspec
from alpaca_trade_api.rest import REST, TimeFrame
from config import API_KEY_ID, SECRET_KEY_ID, BASE_URL

rest_api = REST(API_KEY_ID, SECRET_KEY_ID, BASE_URL)

style.use("dark_background")

# Adjust display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

parser = argparse.ArgumentParser(description='Process stock market data from Alpaca Markets')
parser.add_argument('--command', type=str, help='The command to run')
parser.add_argument('--symbol', type=str, help='The stock symbol')
parser.add_argument('--tail', type=int, default=252, nargs='?', help='The tail value (default: 252)')
parser.add_argument('--window', type=int, default=None, nargs='?', help='The window value')
parser.add_argument('--factor', type=float, default=0.20, nargs='?', help='The factor value (default: 0.20)')
parser.add_argument('--plot', type=int, default=1, nargs='?', help='The plot value (default: 1)')

args = parser.parse_args()

symbol = args.symbol.upper()
tail = args.tail
window = args.window if args.window else round(tail // 22.90)
factor = args.factor
plot_switch = args.plot

if symbol is None:
    print("Must enter symbol eg: TQQQ -> exiting")
    sys.exit(0)

if args.command == "help":
    parser.print_help()
    sys.exit(0)

def find_co_name(symbol, file_path='tickers.txt'):
    with open(file_path, 'r') as file:
        pattern = re.compile(r'^\b' + re.escape(symbol) + r'\b\|(.*)$')
        for line in file:
            match = pattern.match(line.strip())
            if match:
                return match.group(1)
    return "N/A"

try:
    co_name = find_co_name(symbol)
except BaseException:
    co_name = "N/A"

def nyse_trading_days_dataframe(tail: int) -> pd.DataFrame:
    # Get the NYSE trading calendar
    nyse = mcal.get_calendar('NYSE')

    # Get the current timestamp in the America/New_York timezone
    end_date = pd.Timestamp.now(tz='America/New_York').strftime("%Y-%m-%d")

    # Get the trading schedule up to the current date
    schedule = nyse.schedule(start_date='1900-01-01', end_date=end_date)

    # Calculate the start date by selecting the last 'ndays' trading days
    ndays = int(1.5 * tail)
    start_date = schedule.iloc[-ndays].name.strftime("%Y-%m-%d")

    # Get the trading schedule for the specified date range
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)

    # Create a date range with only the trading days and format it as "year-month-day"
    trading_days = mcal.date_range(schedule, frequency='1D').strftime('%Y-%m-%d')

    # Create a DataFrame with the trading days as the index
    df = pd.DataFrame(index=trading_days)

    return df, start_date, end_date, ndays

df, start_date, end_date, ndays = nyse_trading_days_dataframe(tail)

def fetch_alpaca_data(symbol: str, start_date: str, end_date: str, tail: int, rest_api: REST) -> pd.DataFrame:
    # Fetch historical data from Alpaca
    bars = (
        rest_api.get_bars(
            symbol, TimeFrame.Day, start_date, end_date, adjustment="split"
        )
        .df[["close"]]
        .tail(tail)
    )

    # Format the index as "year-month-day"
    bars.index = bars.index.strftime('%Y-%m-%d')

    return bars

df, start_date, end_date, ndays = nyse_trading_days_dataframe(tail)
historical_data = fetch_alpaca_data(symbol, start_date, end_date, tail, rest_api)

# Combine the df and bars DataFrames, keeping only rows with values in the close column
df = df.merge(historical_data, left_index=True, right_index=True, how='inner')

# Remove any blank rows from the combined DataFrame
df.dropna(axis=0, inplace=True)

# Fetch the current price
current_price = rest_api.get_latest_trade(symbol).price

# Get the last row's index and close value
last_index = df.index[-1]
last_close = df.loc[last_index, 'close']

# Check if the current price is different from the last close value
if current_price != last_close:
    # Update the last row's close value with the current price
    df.loc[last_index, 'close'] = current_price

hundred = 100.0
# Calculate the percent change and convert it to cumulative percent change
df['cumulative_pct_change'] = ((df['close'].pct_change() + 1).cumprod() - 1) * hundred

# Calculate daily returns in percent change
df['daily_returns'] = df['close'].pct_change() * hundred

# Calculate the z-score (normalized standard deviation) for the daily returns
df['daily_returns_z_score'] = (df['daily_returns'] - df['daily_returns'].mean()) / df['daily_returns'].std()

def compute_avo_attributes(data):
    num_samples = len(data)
    z_array = np.linspace(1, num_samples, num_samples)
    trans_matrix = np.vstack([z_array, np.ones(num_samples)]).T
    gradient, intercept = np.linalg.lstsq(trans_matrix, data, rcond=None)[0]
    return (gradient, intercept)


def group_consecutives(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def compute_circles(symbol, df, window, df_temp):
    try:
        num_samples = len(df)
        data_percent = df.cumulative_pct_change.values
        data_close = df.close.values
        data_percent[0] = 0.0

        gradient = (data_percent[-1] - data_percent[0]) / (num_samples - 1)
        intercept = data_percent[0]

        remove_trend = data_filter = np.zeros(num_samples)
        for i in range(0, num_samples):
            remove_trend[i] = data_percent[i] - ((gradient * i) + intercept)

        computed_filter = signal.windows.hann(window)
        filter_result = signal.convolve(
            remove_trend, computed_filter, mode="same"
        ) / sum(computed_filter)

        for i in range(0, num_samples):
            data_filter[i] = filter_result[i] + ((gradient * i) + intercept)

        mean_value = np.mean(data_percent)
        std_dev = np.std(data_percent)
        test_factor = (data_percent[-1] - mean_value) / std_dev

        first_derivative = signal.savgol_filter(
            data_filter, delta=1, window_length=3, polyorder=2, deriv=1
        )

        mean = np.mean(first_derivative)
        std = np.std(first_derivative)
        a = (first_derivative - mean) / std
        first_derivative = a / np.max(a)
        velocity = first_derivative

        (min_,) = np.where(first_derivative < -factor)
        (max_,) = np.where(first_derivative > factor)

        gap_min = group_consecutives(min_)
        gap_max = group_consecutives(max_)

        gap_array = []
        for i in range(0, len(gap_min)):
            if len(gap_min[i]) > 0:
                gap_array = np.append(gap_array, np.min(gap_min[i]))
                gap_array = np.append(gap_array, np.max(gap_min[i]))
        for i in range(0, len(gap_max)):
            if len(gap_max[i]) > 0:
                gap_array = np.append(gap_array, np.min(gap_max[i]))
                gap_array = np.append(gap_array, np.max(gap_max[i]))

        gap_array = np.append(gap_array, 1)
        gap_array = np.append(gap_array, num_samples)
        gap_array = np.unique(np.sort(gap_array).astype(int))
        diff_array = np.diff(np.sign(first_derivative))
        (mina_,) = np.where(diff_array < 0)
        (maxa_,) = np.where(diff_array > 0)
        final_min = final_max = []
        last = "None"

        for i in range(0, len(gap_array) - 1):
            val1 = gap_array[max(min(i, num_samples), 0)]
            val2 = gap_array[max(min(i + 1, num_samples), 0)]
            index_min = 0
            for j in range(0, len(mina_)):
                if mina_[j] >= val1 and mina_[j] <= val2:
                    if mina_[j] > index_min:
                        index_min = mina_[j]
            index_max = 0
            for j in range(0, len(maxa_)):
                if maxa_[j] >= val1 and maxa_[j] <= val2:
                    if maxa_[j] > index_max:
                        index_max = maxa_[j]
            if index_min > 0 or index_max > 0:
                if index_min > index_max:
                    if last != "min":
                        final_min = np.append(final_min, index_min).astype(
                            int
                        )
                        last = "min"
                else:
                    if last != "max":
                        final_max = np.append(final_max, index_max).astype(
                            int
                        )
                        last = "max"

        if len(final_min) > 0 and len(final_max) > 0:
            final_min = final_min.astype(int)
            final_max = final_max.astype(int)

        isamp = 0
        action_price = 0.0
        if len(final_min) > 0 and len(final_max) > 0:
            if final_min[-1:] > final_max[-1:]:
                isamp = int(final_min[-1:])
                action = "Sell"
            else:
                isamp = int(final_max[-1:])
                action = "Buy"
        elif len(final_min) == 0 and len(final_max) > 0:
            isamp = int(final_max[-1:])
            action = "Buy"
        elif len(final_min) > 0 and len(final_max) == 0:
            isamp = int(final_min[-1:])
            action = "Sell"
        else:
            isamp = int(len(data_percent))
            action = "None"

        isamp_ago = int(num_samples - isamp)
        action_price = float(data_close[isamp])

        gradient, intercept = compute_avo_attributes(data_percent)
        trend_diff = (
            (data_percent[isamp] - ((gradient * isamp) + intercept))
            / data_percent[isamp]
        ) * 100.0

        mean_value = np.mean(data_percent)
        std_dev = np.std(data_percent)
        detect_value = (data_percent[-1] - mean_value) / abs(std_dev)

        percent = ((current_price - action_price) / action_price) * hundred

        df_temp.loc[symbol] = [
            num_samples,
            window,
            mean_value,
            std_dev,
            velocity[-1],
            detect_value,
            test_factor,
            trend_diff,
            action,
            action_price,
            current_price,
            isamp_ago,
            percent,
        ]
    except Exception as e:
        print(e)
        sys.exit(0)
    return data_filter, final_min, final_max, first_derivative, df_temp

percent_change_today = (current_price/df["close"].iloc[-2] - 1.0) * hundred

df_temp = pd.DataFrame(
    index=[symbol],
    columns=[
        "nsamps",
        "filter",
        "mean",
        "std_dev",
        "last_vel",
        "detect_value",
        "test_factor",
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
) = compute_circles(symbol, df, window, df_temp)

df["zero"] = 0.0

plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(3, 1, height_ratios=[3, 2, 1])

ax0 = plt.subplot(gs[0])
df["cumulative_pct_change"].plot(ax=ax0)

data_percent = df.cumulative_pct_change.values
data_close = df.close.values
df["filter_close"].plot(ax=ax0, linewidth=7, alpha=0.6, color="yellow")
df["filter_close"].plot(
    ax=ax0, color="black", linewidth=2, alpha=1.0, linestyle="--"
)

if len(final_min) > 0:
    ax0.scatter(
        final_min, data_percent[final_min], c="r", s=400, edgecolor="white"
    )
if len(final_max) > 0:
    ax0.scatter(
        final_max, data_percent[final_max], c="g", s=400, edgecolor="white"
    )

ax0.grid(color="white", linestyle="--", linewidth=0.40)
ax0.legend(["percent", "filter_close"], loc="best")
ax0.set_title(
    "Cumulative daily percent price-change for: %s / %s / %.2f Percent Change Today"
    % (symbol, co_name, percent_change_today)
)

ax1 = plt.subplot(gs[1])
df["close"].plot(ax=ax1, label="close_price")

if len(final_min) > 0:
    ax1.scatter(
        final_min,
        data_close[final_min],
        c="r",
        s=150,
        edgecolor="white",
        label="sell",
    )
if len(final_max) > 0:
    ax1.scatter(
        final_max,
        data_close[final_max],
        c="g",
        s=150,
        edgecolor="white",
        label="buy",
    )

ax1.grid(color="white", linestyle="--", linewidth=0.40)
ax1.legend(["close_price", "sell", "buy"], loc="best")
ax1.set_title(f"Close Prices / Current Price: {current_price}")

ax2 = plt.subplot(gs[2])
df["velocity"].plot(ax=ax2)
df["zero"].plot(ax=ax2, linewidth=4, alpha=0.6, color="yellow")
df["zero"].plot(ax=ax2, linewidth=1, alpha=1.0, color="black", linestyle="--")
ax2.grid(color="white", linestyle="--", linewidth=0.40)
ax2.legend(["velocity"], loc="best")
ax2.set_title("Filtered Velocity")

plt.tight_layout()

df.drop(["zero"], axis=1, inplace=True)
print(df.tail(10))
print(" ")
print(df_temp)

if plot_switch == 0:
    plt.savefig(
        "./%s_daily.png" % (symbol),
        dpi=90,
        bbox_inches="tight",
    )
    plt.close()
else:
    plt.show()
