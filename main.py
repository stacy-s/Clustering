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
    Запуск алгоритма кластеризации для всех указанных значений параметра eps для некоторого значения k.
    Иллюстрация самого лучшего разбиения на кластеры для заданного k
    Построение графика зависимости метрики ARI от значений eps
    :param k: значение k
    :param eps: numpy массив значений eps
    :param algorithm_clustering: алгоритм кластеризации
    :param metric: алгоритм оценки качества кластеризации из модуля Metrics
    :param cluster: объект класса cluster
    :param step: шаг сетки на графике зависимости метрики ARI от значений eps
    :param is_show_time: нужно ли печать время работы алгоритма кластеризации
    :return: None
    """
    def make_title(obj):
        """
        Получение названия алгоритма
        :param obj:
        :return: название алгоритма
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


def main():
    data = [functools.partial(make_blobs, n_samples=200, cluster_std=0.5),
            functools.partial(make_blobs, n_samples=200, cluster_std=[1.0, 1.5, 0.5]),
            functools.partial(make_circles, n_samples=200, noise=0.05, factor=0.4),
            functools.partial(make_moons, n_samples=200, noise=0.05)]  # наборы данных
    for i in range(len(data)):
        print(data[i].__str__())
        experimental_results = ResultsAnalysis.static_experiment(Metrics.ARI, 1000,
                                                                 np.arange(1, 13), np.arange(0.1, 4.1, 0.1), data[i])
        ResultsAnalysis.save_results(f"{data[i].func.__name__} {data[i].__str__()[data[i].__str__().find(',') + 2:-1]}",
                                     experimental_results)

    # data = [make_moons(n_samples=200, noise=0.05, random_state=0),
    #         make_circles(n_samples=200, noise=0.05, random_state=0, factor=0.4),
    #         make_blobs(n_samples=200, random_state=0, cluster_std=0.5),
    #         make_blobs(n_samples=200, random_state=170, cluster_std=[1.0, 1.5, 0.5])]  # наборы данных
    #
    # # Запуск алгоритма k-MXT для всех указанных значений k и eps для каждого набора данных.
    # # Построение графика зависимости метрики ARI от значений eps для каждого k.
    # # Отрисовка лучшего разбиения на кластеры для каждого k.
    # # for points, correct_clustering in data:
    # for points, ccorrect_clustering in data:
    #     d = Cluster.Cluster(points, ccorrect_clustering)
    #     for k in range(1, 13):
    #         build_plot_metrics_eps(k, np.arange(0.1, 4.1, 0.1), Clustering.K_MXT, Metrics.ARI, d)
    #         build_plot_metrics_eps(k, np.arange(0.1, 4.1, 0.1), Clustering.K_MXTGauss, Metrics.ARI, d)

    # # Кластеризация данных для города Санкт-Петербурга при заданных значениях параметров k и eps.
    # d = Cluster.ClusterGreatCircles('~/documents/diplom/', 'geoflickr_spt.csv')
    # for k in [7]:
    #     for eps in [50]:
    #         c = Clustering.K_MXTGreatCircle(eps, k, d)
    #         c()
    #         m = Metrics.Modularity(c)
    #         print('k-MXT ', m(), k, eps)
    #         c.cluster.view_at_map(latitude=59.93863, longitude=30.31413,
    #                               filename_of_map=f'{k}-MXT-eps{eps}')
    #         c = Clustering.K_MXTGaussGreatCircle(eps, k, d)
    #         c()
    #         c.cluster.view_at_map(latitude=59.93863, longitude=30.31413,
    #                               filename_of_map=f'{k}-MXTGauss-eps{eps}')
    #         m = Metrics.Modularity(c)
    #         print('Gauss ', m(), k, eps)



if __name__ == '__main__':
    main()
