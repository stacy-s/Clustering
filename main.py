import Cluster
import Clustering
import Metrics
import numpy as np
import BuildData
from sklearn.datasets import make_moons, make_blobs, make_circles
import time


def build_plot_metrics_eps(k, eps: np.ndarray, algorithm_clustering, cluster: Cluster.Cluster, step=0.5):
    """
    Запуск алгоритма кластеризации для всех указанных значений параметра eps для некоторого значения k.
    Иллюстрация самого лучшего разбиения на кластеры для заданного k
    Построение графика зависимости метрики ARI от значений eps
    :param k: значение k
    :param eps: numpy массив значений eps
    :param algorithm_clustering: алгоритм кластеризации
    :param cluster: объект класса cluster
    :param step: шаг сетки на графике зависимости метрики ARI от значений eps
    :return: None
    """
    def make_title(obj):
        """

        :param obj:
        :return:
        """
        title = str(obj)
        title = title[:title.find(',')]
        return title

    value_of_metric = []
    objects = []
    working_time = []
    for cur_eps in eps:
        c = algorithm_clustering(k=k, eps=cur_eps, cluster=cluster)
        start = time.time()
        c()
        working_time.append(time.time() - start)
        metric = Metrics.ARI(c.cluster)
        value_of_metric.append(metric())
        objects.append(c)
    pos = BuildData.find_position_of_max_value_of_metric(value_of_metric)
    BuildData.build_point(value_of_metric[pos], objects[pos])
    points = np.array([[eps[i], value_of_metric[i]] for i in range(len(value_of_metric))])
    title = make_title(objects[pos])
    BuildData.make_plot(points, title, "ARI", np.arange(0.0, max(eps) + step, step), np.arange(0.0, 1.1, 0.1))


def main():
    data = [make_moons(n_samples=200, noise=0.05, random_state=0),
            make_circles(n_samples=200, noise=0.05, random_state=0, factor=0.4),
            make_blobs(n_samples=200, random_state=0, cluster_std=0.5),
            make_blobs(n_samples=200, random_state=170, cluster_std=[1.0, 1.5, 0.5])]  # наборы данных

    # Запуск алгоритма k-MXT для всех указанных значений k и eps для каждого набора данных.
    # Построение графика зависимости метрики ARI от значений eps для каждого k.
    # Отрисовка лучшего разбиения на кластеры для каждого k.
    for points, correct_clustering in data:
        d = Cluster.Cluster(points, correct_clustering)
        for k in range(1, 13):
            build_plot_metrics_eps(k, np.arange(0.1, 4.1, 0.1), Clustering.K_MXT, d)

    # Кластеризация данных для города Санкт-Петербурга при заданных значениях параметров k и eps.
    d = Cluster.ClusterGreatCircles('~/documents/diplom/', 'geoflickr_spt.csv')
    for k in [7, 8, 10, 12]:
        for eps in [25, 50, 70]:
            c = Clustering.K_MXTGreatCircle(eps, k, d)
            c()
            c.cluster.view_at_map(latitude=59.93863, longitude=30.31413,
                                  filename_of_map="{0}-MXT-eps{1}".format(k, eps))


if __name__ == '__main__':
    main()
