import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN
import pandas as pd
import numpy as np


def create_data(n_sample, n_feature, centers, cluster_std, shuffle, random_state):
    X, y = make_blobs(
        n_samples=n_sample,
        n_features=n_feature,
        centers=centers,
        cluster_std=cluster_std,
        shuffle=shuffle,
        random_state=random_state
    )
    print(X)
    return X, y


def visualize(x, y, color="white", marker="o", edgecolors="white", s=50, label=None):
    plt.scatter(
        x,
        y,
        c=color,
        marker=marker,
        edgecolors=edgecolors,
        s=s,
        label=label
    )


def k_mean(X, n_clusters=3):
    k_mean_algorithm = KMeans(
        n_clusters=n_clusters, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    y_km = k_mean_algorithm.fit_predict(X)
    return y_km, k_mean_algorithm


def dbscan(X):
    dbscan = DBSCAN(eps=.3, min_samples=10).fit(X)
    label = dbscan.labels_
    return label, dbscan


def read_data():
    columns = ["sepal length", "sepal width", "petal length", "petal width", "label"]
    df = pd.read_csv("./data/iris.data", names=columns)
    x1 = np.array(df["sepal length"])
    x1 = x1.reshape(x1.shape[0], 1)
    x2 = np.array(df["petal length"])
    x2 = x2.reshape(x2.shape[0], 1)
    data = np.concatenate((x1, x2), axis=1)
    visualize(data[:, 0], data[:, 1], color="white", marker="o", edgecolors="black", s=50)
    return data

def test_algorithm(data, type_algorithm="k-mean"):
    if type_algorithm=="k-mean":
        y_km, algorithm = k_mean(data)
        # y_db, algorithm_db = dbscan(data)
        visualize(data[y_km == 0, 0], data[y_km == 0, 1], color="lime", marker="s", edgecolors="w", s=50, label="cluster 1")
        visualize(data[y_km == 1, 0], data[y_km == 1, 1], color="sandybrown", marker="x", edgecolors="w", s=50,
                  label="cluster 2")
        visualize(data[y_km == 2, 0], data[y_km == 2, 1], color="blueviolet", marker="v", edgecolors="w", s=50,
                  label="cluster 3")
        visualize(algorithm.cluster_centers_[:, 0], algorithm.cluster_centers_[:, 1], color="red", marker="*",
                  edgecolors="black", s=200, label="center")

    elif type_algorithm =="dbscan":
        y_km, algorithm = dbscan(data)
        print(y_km)
        # y_db, algorithm_db = dbscan(data)
        visualize(data[y_km == 0, 0], data[y_km == 0, 1], color="lime", marker="s", edgecolors="w", s=50,
                  label="cluster 1")
        visualize(data[y_km == 1, 0], data[y_km == 1, 1], color="sandybrown", marker="x", edgecolors="w", s=50,
                  label="cluster 2")
        visualize(data[y_km == 2, 0], data[y_km == 2, 1], color="blueviolet", marker="v", edgecolors="w", s=50,
                  label="cluster 3")
        visualize(data[y_km == -1, 0], data[y_km == -1, 1], color="white", marker="o", edgecolors="w", s=50,
                  label="cluster 4")
    plt.legend(scatterpoints=1)
    plt.grid(True)
    plt.title("Algorithm Clustering for Data")
    plt.show()


if __name__ == "__main__":
    n_sample = 150
    n_feature = 2
    centers = 4
    cluster_std = .5
    shuffle = True
    random_state = 0
    #Tạo dữ liệu
    X, y = create_data(n_sample, n_feature, centers, cluster_std, shuffle, random_state)
    # print(X, X.shape)
    # visualize(X[:,0], X[:, 1])
    # plt.show()
    data = read_data()
    print(data)
    test_algorithm(data, "dbscan")
