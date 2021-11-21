import matplotlib.pyplot as plt
import numpy as np


def data_1():
    x = np.linspace(1, 2.9, 100)
    y1 = np.exp(1 / np.sin(x))
    y2 = - np.sqrt(2 * x - 1)
    return x, y1, y2

def handle_1(ax):
    x, y1, y2 = data_1()
    ax.plot(x, y1, color="g")
    ax.plot(x, y2, color="r")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Ví dụ 1")
    ax.legend(['y1', 'y2'])
    return ax


def handle_2(ax):
    x = np.linspace(0, 100, 100)
    y = np.linspace(-50, 50, 100)
    z = np.sin(2 * x) + np.cos(y)
    ax.scatter(x, y, z, color="g")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Ví dụ 2")
    return ax


def main():
    figure = plt.figure(figsize=(10, 10))
    ax1 = figure.add_subplot(321)
    ax2 = figure.add_subplot(322, projection="3d")
    ax3 = figure.add_subplot(323)
    ax4 = figure.add_subplot(313)
    ax1 = handle_1(ax1)
    ax2 = handle_2(ax2)
    figure.show()


if __name__=="__main__":
    main()