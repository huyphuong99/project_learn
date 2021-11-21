import pandas as pd
import seaborn as sns
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import matplotlib as mpl
from statsmodels.tsa.seasonal import seasonal_decompose
# import pandas_profiling

def visuallize_data(df):
    df.Date = pd.to_datetime(df.Date)
    df.set_index("Date", inplace=True)
    df.plot()
    plt.xlabel('Date')
    plt.show()
    return df


def norm_data(df):
    diff_mean = df["Mean"][1:].values - df["Mean"][:-1].values
    plt.figure(figsize=(10, 6))
    plt.boxplot(diff_mean)
    plt.ylabel("Mean", fontsize=12)
    plt.title("Boxplot of data", fontsize=18)
    plt.show()
    return diff_mean


def detect_outliers(series):
    """
      series: 1-D numpy array input
    """
    Q1 = np.quantile(series, 0.25)
    Q3 = np.quantile(series, 0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    lower_compare = series <= lower_bound
    upper_compare = series >= upper_bound
    outlier_idxs = np.where(lower_compare | upper_compare)[0]
    print("Outlier indices: ", outlier_idxs)
    print("Outlier years: ", df.index[outlier_idxs + 1].values)
    print("Outlier values: ", series[outlier_idxs])
    return outlier_idxs


def cyclical(df):
    mpl.rc("figure", figsize=(10, 6))
    plot_acf(df["Mean"], lags=12)
    plt.show()

def sma(df):
    # Single moving average in pandas
    df_sma = df.rolling(window=4).mean()
    plt.figure(figsize=(10, 6))
    plt.plot(df["Mean"], label="x")
    plt.plot(df_sma["Mean"], label="sma(4)")
    plt.xticks(rotation=45)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Mean", fontsize=12)
    plt.legend(fontsize=12)
    plt.title("Mean in years", fontsize=18)
    plt.show()

def ema(df):
    df_ewm = df.ewm(span=12, adjust=False).mean()
    plt.figure(figsize=(10, 6))
    plt.plot(df["Mean"], label="x")
    plt.plot(df_ewm["Mean"], label="ewm(12)")
    plt.xticks(rotation=45)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Mean", fontsize=12)
    plt.legend(fontsize=12)
    plt.title("Mean of years", fontsize=18)
    plt.show()

def temp(df):
    result = seasonal_decompose(df["Mean"], model='additive')
    result.plot()
    plt.show()


if __name__ == "__main__":
    path = "/home/huyphuong99/Downloads/global-temp_zip/archive/monthly.csv"
    # Read data
    df = pd.read_csv(path)
    df = df[:500]
    # print(df)
    # df.info()

    # Visualize data
    df = visuallize_data(df)

    # Norm data
    take_data = norm_data(df)
    outline = detect_outliers(take_data)
    #
    #Cyclical
    cyclical(df)

    ##Proccess sma
    sma(df)
    ##Proccess ema
    ema(df)

    # temp(df)

    # df.profile_report()
