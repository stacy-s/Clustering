import Cluster
import Clustering
import Graph
import Metrics
import numpy as np
from sklearn.datasets import make_moons, make_blobs, make_biclusters, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt


def make_plot(points, title, name_validation_metric, xticks: np.ndarray, yticks: np.ndarray, figsize=(7, 7)):
    fig = plt.figure(figsize=figsize)
    # fig.set_ylim(bottom=0, top=1.1)
    plt.ylabel(name_validation_metric)
    plt.xlabel('eps')
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.plot(points[:, 0], points[:, 1], 'o', color='blue', linestyle='solid',
             markersize=4)
    plt.title(title)
    plt.grid()
    plt.show()


def find_position_of_max_value_of_metric(value_of_metric):
    mx = max(value_of_metric)
    return [x for x in range(len(value_of_metric)) if value_of_metric[x] == mx][0]


def build_point(value_of_metric, objects):
    objects.cluster.view(title=str(objects) + " ARI = {0:.2f}".format(value_of_metric))


def build_plot_metrics_eps(k, eps: np.ndarray, algorithm_clustering, cluster: Cluster.Cluster, metric: Metrics.ARI,
                           step = 0.5):
    value_of_metric = []
    objects = []
    for cur_eps in eps:
        c = algorithm_clustering(k=k, eps=cur_eps, cluster=cluster)
        c()
        metric = Metrics.ARI(c.cluster)
        value_of_metric.append(metric())
        objects.append(c)
    pos = find_position_of_max_value_of_metric(value_of_metric)
    build_point(value_of_metric[pos], objects[pos])
    points = np.array([[eps[i], value_of_metric[i]] for i in range(len(value_of_metric))])
    title = str(objects[pos])
    make_plot(points, title, "ARI", np.arange(min(eps), max(eps) + step, step), np.arange(0, 1.1, 0.1))


def main():
    points, correct_clustering = make_blobs(n_samples=200,
                                            random_state=0,
                                            cluster_std=0.5)

    d = Cluster.Cluster(points, correct_clustering)
    build_plot_metrics_eps(10, np.arange(0.1, 4.1, 0.1), Clustering.K_MXTGauss, d, Metrics.ARI)


if __name__ == '__main__':
    main()