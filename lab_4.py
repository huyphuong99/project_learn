import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy


def read_data():
    return pd.read_csv("./data/Banks.txt", sep='\t', encoding="cp1251")


def distribution(df, ax):
    a = sns.distplot(df, kde=True, ax=ax)
    return a


def bar_plot(df, ax):
    a = sns.boxplot(df, ax=ax)
    return a


def statistic(df):
    return df.describe()


def test_mean(mean, median, std, ax):
    x = [mean, median]
    y = [1, 3]
    error = [std ** (1 / 2), std ** (1 / 2)]
    plt.errorbar(x, y, xerr=error, fmt="o", color="b")
    plt.yticks((0, 1, 3, 5), ("", "median", "mean", ""))
    # plt.title("95% confidence intervals")
    return ax


def aderson_statistic(df, st):
    s = (st['std'] / (st['count'] - 1)) ** (1 / 2)
    skewness = (st['std'] * (3 / 2)) / ((st['count'] - 1) * (s ** 3))
    kurtosis = (st['std'] ** 2) / ((st['count'] - 1) * (s ** 4))
    anderson_statistic = scipy.stats.anderson(df, dist='norm')
    print(f"Test Anderson: {anderson_statistic.statistic}")
    print('-------------------------------')
    print(f"Mean: {st['mean']}")
    print(f"StDev: {st['std'] ** (1 / 2)}")
    print(f"Variance: {st['std']}")
    print(f"Skewness: {skewness}")
    print(f"Kurtosis: {kurtosis}")
    print(f"N: {int(st['count'])}")
    print('-------------------------------')
    print(f"Min: {st['min']}")
    print(f"1st Quartile: {st['25%']}")
    print(f"Median: {st['50%']}")
    print(f"3st Quartile: {st['75%']}")
    print(f"Maximum: {st['max']}")
    print('-------------------------------')
    print("95% confidence intervals")
    print(f"Mean     : {st['mean'] - st['std'] ** (1 / 2)}\t\t{st['mean'] + st['std'] ** (1 / 2)}")
    print(f"Median   : {st['50%'] - st['std'] ** (1 / 2)}\t\t{st['50%'] + st['std'] ** (1 / 2)}")
    print(f"std      : {st['std'] - st['std'] ** (1 / 2)}\t\t{st['std'] + st['std'] ** (1 / 2)}")


def visualize_data(df):
    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1 = distribution(df, ax1)
    ax2 = bar_plot(df, ax2)
    ax3 = test_mean(st["mean"], df.median(), st["std"], ax3)
    plt.show()


if __name__ == "__main__":
    df = read_data()
    print(df.columns)
    # data = df['Рег. Номер']
    data = df['Количество рабочих']
    st = statistic(data)
    visualize_data(data)
    aderson_statistic(data, st)
