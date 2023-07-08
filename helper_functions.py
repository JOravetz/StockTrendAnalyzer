import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal

from scipy import signal
from matplotlib import gridspec
from alpaca_trade_api.rest import REST, TimeFrame


def find_co_name(target):
    with open('tickers.txt', 'r') as file:
        reader = csv.reader(file, delimiter='|')
        for row in reader:
            if row[0] == target:
                return row[1]
    return "N/A"

def nyse_trading_days_dataframe(tail: int) -> pd.DataFrame:
    nyse = mcal.get_calendar("NYSE")
    end_date = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
    schedule = nyse.schedule(start_date="1900-01-01", end_date=end_date)
    ndays = int(1.5 * tail)
    start_date = schedule.iloc[-ndays].name.strftime("%Y-%m-%d")
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    trading_days = mcal.date_range(schedule, frequency="1D").strftime(
        "%Y-%m-%d"
    )
    df = pd.DataFrame(index=trading_days)
    return df, start_date, end_date, ndays


def fetch_alpaca_data(
    symbol: str, start_date: str, end_date: str, tail: int, rest_api: REST
) -> pd.DataFrame:
    # Fetch historical data from Alpaca
    bars = (
        rest_api.get_bars(
            symbol, TimeFrame.Day, start_date, end_date, adjustment="split"
        )
        .df[["close"]]
        .tail(tail)
    )

    # Format the index as "year-month-day"
    bars.index = bars.index.strftime("%Y-%m-%d")

    return bars


def compute_avo_attributes(data):
    num_samples = len(data)
    z_array = np.linspace(1, num_samples, num_samples)
    trans_matrix = np.vstack([z_array, np.ones(num_samples)]).T
    gradient, intercept = np.linalg.lstsq(trans_matrix, data, rcond=None)[0]

    return (gradient, intercept)


def group_consecutives(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def compute_trend_and_filter(num_samples, data_percent, window):

    gradient = (data_percent[-1] - data_percent[0]) / (num_samples - 1)
    intercept = data_percent[0]

    remove_trend = np.zeros(num_samples)
    for i in range(num_samples):
        remove_trend[i] = data_percent[i] - ((gradient * i) + intercept)

    computed_filter = signal.windows.hann(window)
    filter_result = signal.convolve(
        remove_trend, computed_filter, mode="same"
    ) / sum(computed_filter)

    data_filter = np.zeros(num_samples)
    for i in range(num_samples):
        data_filter[i] = filter_result[i] + ((gradient * i) + intercept)

    return data_filter, gradient, intercept


def calculate_velocity(data_filter):

    first_derivative = signal.savgol_filter(
        data_filter, delta=1, window_length=3, polyorder=2, deriv=1
    )

    mean = np.mean(first_derivative)
    std = np.std(first_derivative)
    a = (first_derivative - mean) / std
    first_derivative = a / np.max(a)
    velocity = first_derivative

    return first_derivative, velocity


def calculate_min_max(gap_array, mina_, maxa_):
    final_min = final_max = []
    last = "None"
    num_samples = len(gap_array) - 1
    for i in range(0, num_samples):
        val1 = gap_array[max(min(i, num_samples), 0)]
        val2 = gap_array[max(min(i + 1, num_samples), 0)]
        index_min = max(
            [
                mina_[j]
                for j in range(0, len(mina_))
                if mina_[j] >= val1 and mina_[j] <= val2
            ],
            default=0,
        )
        index_max = max(
            [
                maxa_[j]
                for j in range(0, len(maxa_))
                if maxa_[j] >= val1 and maxa_[j] <= val2
            ],
            default=0,
        )
        if index_min > 0 or index_max > 0:
            if index_min > index_max:
                if last != "min":
                    final_min = np.append(final_min, index_min).astype(int)
                    last = "min"
            else:
                if last != "max":
                    final_max = np.append(final_max, index_max).astype(int)
                    last = "max"
    return final_min, final_max


def calculate_action_points(num_samples, factor, first_derivative):
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

    final_min, final_max = calculate_min_max(gap_array, mina_, maxa_)

    isamp = 0
    action = "None"
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
        isamp = int(len(first_derivative))
        action = "None"

    return final_min, final_max, isamp, action


def compute_circles(symbol, df, window, factor, current_price, df_temp):
    try:
        num_samples = len(df)
        data_percent = df.cumulative_pct_change.values
        data_close = df.close.values
        data_percent[0] = 0.0

        # using defined compute_avo_attributes
        gradient, intercept = compute_avo_attributes(data_percent)

        # using defined compute_trend_and_filter
        data_filter, gradient, intercept = compute_trend_and_filter(
            num_samples, data_percent, window
        )

        # using defined calculate_velocity
        velocity, first_derivative = calculate_velocity(data_filter)

        # using defined calculate_action_points
        final_min, final_max, isamp, action = calculate_action_points(
            num_samples, factor, first_derivative
        )

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

        percent = ((current_price - action_price) / action_price) * 100.0

        df_temp.loc[symbol] = [
            num_samples,
            window,
            mean_value,
            std_dev,
            velocity[-1],
            detect_value,
            factor,
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

def update_current_price(df, current_price, last_close, last_index):
    if current_price != last_close:
        # Update the last row's close value with the current price
        df.loc[last_index, "close"] = current_price
    return df

def calculate_returns(df):
    # Calculate the cumulative percentage change
    df["cumulative_pct_change"] = (df["close"].pct_change() + 1).cumprod().fillna(
        1
    ) * 100.0 - 100.0

    # Calculate the daily returns
    df["daily_returns"] = df["close"].pct_change() * 100.0
    df["daily_returns"].iloc[0] = 0.0
    daily_returns = df["daily_returns"].values
    std = np.std(daily_returns)
    std_dev = daily_returns / std
    df["std-dev"] = std_dev

    return df

def convert_to_cumulative_percent_change(df):
    data = df.close.values
    ns = len(data)

    data_perc = np.zeros(ns)
    data_perc[0] = 0.0
    for i in range(1, ns):
        last = data[i - 1]
        data_perc[i] = data_perc[i - 1] + ((data[i] - last) / last) * 100.0
    df["cumulative_pct_change"] = data_perc

    return df

def plot_graph(df, final_min, final_max, symbol, co_name, percent_change_today, current_price, plot_switch):
    """
    Function to plot graphs for cumulative daily percent price-change, close prices and filtered velocity.

    :param df: DataFrame containing data to plot
    :param final_min: final minimum values
    :param final_max: final maximum values
    :param symbol: stock symbol
    :param co_name: company name
    :param percent_change_today: percent change today
    :param current_price: current price
    :param plot_switch: switch to control plot saving or showing
    :return: None
    """

    plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 2, 1])

    ax0 = plt.subplot(gs[0])
    df["cumulative_pct_change"].plot(ax=ax0)

    data_percent = df.cumulative_pct_change.values
    data_close = df.close.values
    df["filter_close"].plot(ax=ax0, linewidth=7, alpha=0.6, color="yellow")
    df["filter_close"].plot(ax=ax0, color="black", linewidth=2, alpha=1.0, linestyle="--")

    if len(final_min) > 0:
        ax0.scatter(final_min, data_percent[final_min], c="r", s=400, edgecolor="white")
    if len(final_max) > 0:
        ax0.scatter(final_max, data_percent[final_max], c="g", s=400, edgecolor="white")

    ax0.grid(color="white", linestyle="--", linewidth=0.40)
    ax0.legend(["percent", "filter_close"], loc="best")
    ax0.set_title("Cumulative daily percent price-change for: %s / %s / %.2f Percent Change Today"
                  % (symbol, co_name, percent_change_today))

    ax1 = plt.subplot(gs[1])
    df["close"].plot(ax=ax1, label="close_price")

    if len(final_min) > 0:
        ax1.scatter(final_min, data_close[final_min], c="r", s=150, edgecolor="white", label="sell")
    if len(final_max) > 0:
        ax1.scatter(final_max, data_close[final_max], c="g", s=150, edgecolor="white", label="buy")

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

    if plot_switch == 0:
        plt.savefig("./%s_daily.png" % (symbol), dpi=90, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
