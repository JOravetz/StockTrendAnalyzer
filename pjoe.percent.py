#!/usr/bin/python3

import os
import argparse
import concurrent
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from scipy import signal
from alpaca_trade_api.rest import REST, TimeFrame

API_KEY_ID = os.environ["APCA_API_KEY_ID"]
SECRET_KEY_ID = os.environ["APCA_API_SECRET_KEY"]
BASE_URL = os.environ["APCA_API_BASE_URL"]

rest_api = REST(API_KEY_ID, SECRET_KEY_ID, BASE_URL)

pd.options.mode.chained_assignment = None
pd.set_option("display.width", 1000)
pd.set_option("display.max_rows", None, "display.max_columns", None)

hundred = 100.0


def compute_avo_attributes(data):
    num_samples = len(data)
    z_array = np.linspace(1, num_samples, num_samples)
    trans_matrix = np.vstack([z_array, np.ones(num_samples)]).T
    gradient, intercept = np.linalg.lstsq(trans_matrix, data, rcond=None)[0]
    return (gradient, intercept)


def group_consecutives(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def _get_alpaca_prices(
    symbols, df_index, TIMEFRAME, _from, to_end, tail, workers=10
):
    """Get the map of DataFrame price data from alpaca, in parallel."""

    def historic_get_bars(symbol):
        try:
            bars = (
                rest_api.get_bars(
                    symbol, TIMEFRAME, _from, to_end, adjustment="split"
                ).df[["close"]]
                # .tail(tail)
            )

            bars.reset_index(inplace=True)
            bars["datetime"] = pd.to_datetime(bars["timestamp"]).dt.strftime(
                "%Y-%m-%d"
            )

            bars.set_index("datetime", inplace=True)
            df = df_index.merge(bars, left_index=True, right_index=True)
            df.drop(["timestamp"], axis=1, inplace=True)
        except Exception:
            return
        return df

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers
    ) as executor:
        results = {}
        future_to_symbol = {
            executor.submit(historic_get_bars, symbol): symbol
            for symbol in symbols
        }
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                results[symbol] = future.result()
            except Exception as exc:
                print("{} generated an exception: {}".format(symbol, exc))
        return results


def _process_data(
    symbols,
    price_map,
    filter_window,
    detect_factor,
    reference,
    tail,
    df_temp,
    workers=10,
):
    """Get the map of DataFrame price data from alpaca, in parallel."""

    def compute_circles(symbol):
        try:
            df = price_map[symbol]
            current_price = rest_api.get_latest_trade(symbol).price

            to_end = pd.Timestamp.now(tz="America/New_York").strftime(
                "%Y-%m-%d"
            )

            if df['close'].iloc[-1] != current_price:
                df.loc[to_end] = current_price

            if reference == 1:
                df.drop(df.index[-1], inplace=True)
                current_price = df["close"].iloc[-1]

            df = df.tail(tail)
            num_samples = len(df)

            df["daily_return"] = df["close"].pct_change().values * hundred
            daily_return = df.daily_return.values
            daily_return[0] = 0.0
            std = np.std(daily_return)
            df["std-dev"] = daily_return / std

            data_close = df.close.values

            data_percent = np.zeros(num_samples)
            data_percent[0] = 0.0
            for i in range(1, num_samples):
                last = data_close[i - 1]
                data_percent[i] = (
                    data_percent[i - 1]
                    + ((data_close[i] - last) / last) * hundred
                )
            df["percent"] = data_percent

            df = df[["close", "percent", "daily_return", "std-dev"]]

            num_samples = len(data_percent)
            if num_samples < filter_window:
                return

            gradient = (data_percent[-1] - data_percent[0]) / (
                num_samples - 1
            )
            intercept = data_percent[0]
            remove_trend = data_filter = np.zeros(num_samples)
            for i in range(0, num_samples):
                remove_trend[i] = data_percent[i] - (
                    (gradient * i) + intercept
                )
            computed_filter = signal.windows.hann(filter_window)
            filter_result = signal.convolve(
                remove_trend, computed_filter, mode="same"
            ) / sum(computed_filter)
            for i in range(0, num_samples):
                data_filter[i] = filter_result[i] + (
                    (gradient * i) + intercept
                )

            # mean_value = np.mean(data_percent)
            # std_dev = np.std(data_percent)
            # test_factor = (data_perc - mean_value) / std_dev

            first_derivative = signal.savgol_filter(
                data_filter, delta=1, window_length=3, polyorder=2, deriv=1
            )

            mean = np.mean(first_derivative)
            std = np.std(first_derivative)
            a = (first_derivative - mean) / std
            # velocity = first_derivative
            first_derivative = a / np.max(a)

            (min_,) = np.where(first_derivative < -detect_factor)
            (max_,) = np.where(first_derivative > detect_factor)

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
                            final_min = np.append(
                                final_min, index_min
                            ).astype(int)
                            last = "min"
                    else:
                        if last != "max":
                            final_max = np.append(
                                final_max, index_max
                            ).astype(int)
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

            if data_percent[isamp] != 0.0:
                trend_diff = (
                    (data_percent[isamp] - ((gradient * isamp) + intercept))
                    / data_percent[isamp]
                ) * 100.0
            else:
                trend_diff = 0.0

            # mean_value = np.mean(data_percent)
            # std_dev = np.std(data_percent)
            # detect_value = (data_percent[-1] - mean_value) / abs(std_dev)

            percent = (
                (current_price - action_price) / action_price
            ) * hundred

            df_temp.loc[symbol] = [
                num_samples,
                filter_window,
                first_derivative[-1],
                trend_diff,
                action,
                action_price,
                current_price,
                isamp_ago,
                percent,
            ]
        except Exception as e:
            print(e)
        return

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers
    ) as executor:
        results = {}
        future_to_symbol = {
            executor.submit(compute_circles, symbol): symbol
            for symbol in symbols
        }
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                results[symbol] = future.result()
            except Exception as exc:
                print("{} generated an exception: {}".format(symbol, exc))
        return results


def run(args):

    SAMPLE = str(args.sample)
    tail = int(args.tail)
    ndays = int(args.ndays)
    detect_factor = float(args.factor)
    FILTER = int(args.window)
    LIST = str(args.list)
    SKEY = str(args.skey)
    DIRECTION = int(args.dir)
    reference = int(args.ref)

    with open(LIST + ".lis", "r") as f:
        universe = [row.split()[0] for row in f]
    f.close()
    universe[:] = [item.upper() for item in universe if item != ""]

    TIMEFRAME = TimeFrame.Minute
    if SAMPLE == "Day":
        TIMEFRAME = TimeFrame.Day

    if ndays == 0:
        ndays = int(1.5 * tail)

    if FILTER == 0:
        filter_window = round(tail // 22.90)
    else:
        filter_window = FILTER
    if (filter_window % 2) == 0:
        filter_window += 1

    nyse = mcal.get_calendar("NYSE")

    end_dt = pd.Timestamp.now(tz="America/New_York")
    start_dt = end_dt - pd.Timedelta("%5d days" % ndays)
    _from = start_dt.strftime("%Y-%m-%d")
    to_end = end_dt.strftime("%Y-%m-%d")

    schedule = nyse.schedule(start_date=_from, end_date=to_end)

    mcal_index = mcal.date_range(schedule, frequency="1D").strftime(
        "%Y-%m-%d"
    )
    df_index = pd.DataFrame(index=mcal_index)

    price_map = _get_alpaca_prices(
        universe, df_index, TIMEFRAME, _from, to_end, tail
    )

    df_temp = pd.DataFrame(
        index=universe,
        columns=[
            "nsamps",
            "filter",
            "velocity",
            "trend_diff",
            "action",
            "action_price",
            "price",
            "isamp_ago",
            "percent",
        ],
    )

    _process_data(
        universe,
        price_map,
        filter_window,
        detect_factor,
        reference,
        tail,
        df_temp,
    )

    df_results = df_temp.dropna().sort_values(
        by=[SKEY], ascending=[DIRECTION]
    )

    print(df_results)

    if LIST:
        tfile = open(LIST + ".pjoe.percent.txt", "w")
    else:
        tfile = open("pjoe.percent.txt", "w")

    tfile.write(df_results.to_csv(sep=" "))
    tfile.close()


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "--list",
        type=str,
        default="momentum",
        help="Symbol list (default=momentum)",
    )
    PARSER.add_argument(
        "--ndays",
        type=int,
        default=0,
        help="Number of trading days to fetch historical data (default=0)",
    )
    PARSER.add_argument(
        "--factor",
        type=float,
        default=0.20,
        help="Threshold factor for extrema detection (default=0.20)",
    )
    PARSER.add_argument(
        "--window",
        type=int,
        default=11,
        help="Number of samples in Hanning filter window (default=11, off)",
    )
    PARSER.add_argument(
        "--sample",
        type=str,
        default="Day",
        help="Timeframe between samples (default=Day)",
    )
    PARSER.add_argument(
        "--tail",
        type=int,
        default=252,
        help="Use the last number of samples (default=252, off)",
    )
    PARSER.add_argument(
        "--skey",
        type=str,
        default="percent",
        help="Sort key column name for output result " '(default="percent")',
    )
    PARSER.add_argument(
        "--dir",
        type=int,
        default=0,
        help="Sort direction (ascending=1, default=0)",
    )
    PARSER.add_argument(
        "--ref",
        type=int,
        default=0,
        help="Switch to compute reference from last trading "
        "day (default is 0=off, 1=on)",
    )

    ARGUMENTS = PARSER.parse_args()
    run(ARGUMENTS)
