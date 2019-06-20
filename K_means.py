"""
The module of work with the k-means clustering algorithm
"""
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import KMeans
import tkinter
import matplotlib.pyplot as plt
import mglearn


def build_data(kmeans, points, cnt_clusters):
    """
    The rendering function is a clustering by the k-means algorithm
    :param kmeans: object class KMeans
    :param points: coordinates of clustering points
    :param cnt_clusters: number of clusters
    :return: None
    """
    mglearn.discrete_scatter(points[:, 0], points[:, 1], kmeans.labels_, markers='o')
    mglearn.discrete_scatter(
        kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [x for x in range(cnt_clusters)],
        markers='^', markeredgewidth=2)
    plt.show()


def main():
    data = [make_moons(n_samples=200, noise=0.05, random_state=0),
            make_circles(n_samples=200, noise=0.05, random_state=0, factor=0.4),
            make_blobs(n_samples=200, random_state=0, cluster_std=0.5),
            make_blobs(n_samples=200, random_state=170, cluster_std=[1.0, 1.5, 0.5])]   # datasets
    for points, right_clustering in data:
        cnt_clusters = max(right_clustering) + 1    # number of clusters
        kmeans = KMeans(n_clusters=cnt_clusters)    # creation of cluster objects
        kmeans.fit(points)      # clustering
        build_data(kmeans, points, cnt_clusters)


if __name__ == '__main__':
    main()