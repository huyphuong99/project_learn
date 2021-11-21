import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def data_1():
    x = np.linspace(1, 2.9, 100)
    y1 = np.exp(1 / (np.sin(x) + 1e-4))
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


def data_3():
    x = ["Toán", "Anh", "Nga", "Robot", "Web", "Dự án"]
    y = [90, 92, 95, 85, 95, 80]
    return x, y


def handle_3(ax):
    x, y = data_3()
    #https://matplotlib.org/stable/gallery/color/named_colors.html
    ax.bar(x, y, color=["r", "g", "b", "y", "lime", "m"])
    #https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html
    for (x, y) in zip(x, y):
        ax.text(x, y, y,  va='bottom')
    ax.set_xlabel("Môn học")
    ax.set_ylabel("Điểm")
    ax.set_title("Ví dụ 3")
    return ax


def handle_4(ax):
    scale = 5 / 100
    x, y = data_3()
    y = [scale * element for element in y]
    # print([i for i in zip(x, y)])
    for (a, b) in zip(x, y):
        ax.text(a, b, round(b, 2),  va='bottom')
    ax.bar(x, y, color=["r", "g", "b", "y", "lime", "m"])
    ax.set_xlabel("Môn học")
    ax.set_ylabel("Điểm")
    ax.set_title("Ví dụ 4")
    return ax


def main():
    fig = plt.figure(figsize=(20, 20))
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232, projection="3d")
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(212)

    ax1 = handle_1(ax1)
    ax2 = handle_2(ax2)
    ax3 = handle_3(ax3)
    ax4 = handle_4(ax4)
    plt.title("Plot visualize")
    plt.show()
    # fig.savefig("./img/plot.svg")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
