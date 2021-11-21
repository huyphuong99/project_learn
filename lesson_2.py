import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


def visualize_data():
    fig = plt.figure(figsize=[10, 10])
    ax = fig.add_subplot(111)
    x, y, z, m, n = "Iris-setosa", "Iris-virginica", "Iris-versicolor", "Flower 1", "Flower 2"
    colors = {x: "red", y: "green", z: "blue"}
    ax.scatter(df['sepal length'], df['petal length'], c=df['label'].map(colors))
    #Thêm 2 điểm dữ liệu ngẫu nhiên vào biểu đồ,
    # ax.scatter(6.2, 5.3, c="m")
    # ax.scatter(5.4, 1.9, c="aqua")

    pop_a = mpatches.Patch(color='r', label=x)
    pop_b = mpatches.Patch(color='g', label=y)
    pop_c = mpatches.Patch(color='b', label=z)
    # pop_d = mpatches.Patch(color="m", label=m)
    # pop_e = mpatches.Patch(color="aqua", label=n)
    #Legend: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
    plt.legend(handles=[pop_a, pop_b, pop_c], loc=0)
    plt.grid(True)
    fig.suptitle("Flower Classification Problem")
    plt.show()


def model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    classifier = SVC(kernel='rbf', degree=8)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    columns = ["sepal length", "sepal width", "petal length", "petal width", "label"]
    df = pd.read_csv("./data/iris.data", names=columns)
    X = df.drop(columns="label")
    y = df["label"]
    # model(X, y)
    visualize_data()

