import Cluster
import Clustering
import Metrics
import numpy as np
import BuildData
import ResultsAnalysis
from sklearn.datasets import make_moons, make_blobs, make_circles
import time
import functools


def build_plot_metrics_eps(k, eps: np.ndarray, algorithm_clustering, metric, cluster: Cluster.Cluster, step=0.5,
                           is_show_time=False):
    """
    The function starts the clustering algorithm for all specified values of eps parameters for some value of k.
    The function illustrates the best clustering result for a given k
    The function builds a graph of the ARI metric versus eps values.
    :param k: k value
    :param eps: numpy array of eps values
    :param algorithm_clustering: clustering algorithm
    :param metric: algorithm for assessing the quality of clustering from the Metrics module
    :param cluster: cluster class object
    :param step: grid step on the graph of dependence of the ARI metric on eps values
    :param is_show_time: do I need to print the running time of the clustering algorithm
    :return: None
    """
    def make_title(obj):
        """
        Getting the name of the algorithm
        :param obj:
        :return: algorithm name
        """
        title = str(obj)
        title = title[:title.find(',')]
        return title

    value_of_metric = []
    objects = []
    for cur_eps in eps:
        c = algorithm_clustering(k=k, eps=cur_eps, cluster=cluster)
        start = time.time()
        c()
        end = time.time()
        score = metric(c)
        value_of_metric.append(score())
        objects.append(c)
        if is_show_time:
            print(end - start)
    pos = BuildData.find_position_of_max_value_of_metric(value_of_metric)
    BuildData.build_point(value_of_metric[pos], objects[pos], metric.__name__)
    points = np.array([[eps[i], value_of_metric[i]] for i in range(len(value_of_metric))])
    title = make_title(objects[pos])
    BuildData.make_plot(points, title, metric.__name__, np.arange(0.0, max(eps) + step, step), np.arange(0.0, 1.1, 0.1))


def run_experimental(k, eps):
    """
    The function performs an experiment for given values of the parameter k and eps.
    :param k: the value of k
    :param eps: the value of eps
    :return: None
    """

    data = [functools.partial(make_blobs, n_samples=200, cluster_std=0.5),
            functools.partial(make_blobs, n_samples=200, cluster_std=[1.0, 1.5, 0.5]),
            functools.partial(make_circles, n_samples=200, noise=0.05, factor=0.4),
            functools.partial(make_moons, n_samples=200, noise=0.05)]  # наборы данных
    for i in range(1):
        print(data[i].__str__())
        experimental_results = ResultsAnalysis.static_experiment(Metrics.ARI, 1000,
                                                                 k, eps, data[i])
        ResultsAnalysis.save_results(f"{data[i].func.__name__} {data[i].__str__()[data[i].__str__().find(',') + 2:-1]} k={k} eps={eps}",
                                     experimental_results)


def run_clustering_2d(k, eps):
    """
    The function starts the k-MXT and k-MXT-Gauss algorithm for all specified k and eps values for each data set.
    The function builds a graph of the ARI metric versus eps values.
    The function builds a graph of the ARI metric versus eps values.
    :param k: the value of k
    :param eps: the value of eps
    :return: None
    """
    data = [make_moons(n_samples=200, noise=0.05, random_state=0),
            make_circles(n_samples=200, noise=0.05, random_state=0, factor=0.4),
            make_blobs(n_samples=200, random_state=0, cluster_std=0.5),
            make_blobs(n_samples=200, random_state=170, cluster_std=[1.0, 1.5, 0.5])]  # datasets
    for points, ccorrect_clustering in data:
        d = Cluster.Cluster(points, ccorrect_clustering)
        for kk in k:
            build_plot_metrics_eps(kk, eps, Clustering.K_MXT, Metrics.ARI, d)
            build_plot_metrics_eps(kk, eps, Clustering.K_MXTGauss, Metrics.ARI, d)


def run_clustering_city(filepath, filename, k, eps, latitude, longitude):
    """
    The function clusters data for a given city and draws the result obtained on the map.
    :param filepath: path of file .csv
    :param filename: name of file .csv
    :param k: the value of k
    :param eps: the value of eps
    :param latitude: latitude of city
    :param longitude: longitude of city
    :return: None
    """
    d = Cluster.ClusterGreatCircles(filepath, filename)
    for k in [7]:
        for eps in [50]:
            c = Clustering.K_MXTGreatCircle(eps, k, d)
            c()
            m = Metrics.Modularity(c)
            print(f'k-MXT k={k} eps={eps} Modularity={m()}')
            c.cluster.view_at_map(latitude=latitude, longitude=longitude,
                                  filename_of_map=f'{k}-MXT-eps{eps}')
            c = Clustering.K_MXTGaussGreatCircle(eps, k, d)
            c()
            c.cluster.view_at_map(latitude=latitude, longitude=longitude,
                                  filename_of_map=f'{k}-MXTGauss-eps{eps}')
            m = Metrics.Modularity(c)
            print(f'k-MXT-Gauss k={k} eps={eps} Modularity={m()}')


def main():
    # run_experimental(k=np.arange(1, 13), eps=np.arange(0.1, 4.1, 0.1))
    run_clustering_2d(k=np.arange(1, 13), eps=np.arange(0.1, 4.1, 0.1))
    run_clustering_city(filepath='./datasets/', filename='geoflickr_spb_drop_duplicates.csv',
                        k=[7, 8, 10, 12], eps=[50, 70], latitude=59.93863, longitude=30.31413)


if __name__ == '__main__':
    main()
