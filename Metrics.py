"""
Модуль метрик оценки качества кластеризации
"""
import Cluster
import Clustering
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np


class ARI:
    """
    Подсчет метрики ARI
    """
    def __init__(self, clustering: Clustering.Clustering):
        """
        Конструктор
        :param clustering: объект класса алгоритма кластеризации
        """
        self.cluster = clustering.cluster

    def __call__(self):
        """
        Подсчет метрики ARI
        :return: значение метрики ARI
        """
        return adjusted_rand_score(self.cluster.right_clustering, self.cluster.resulting_clustering)


class Modularity:
    """
    Метрика оценки modularity.
    """
    def __init__(self, clustering: Clustering.Clustering):
        """
        Конструктор
        :param clustering: объект класса алгоритма кластеризации
        """
        self.__cluster = clustering.cluster     # объект класс Cluster.Cluster
        self.__number_of_clusters = max(clustering.cluster.resulting_clustering) + 1    # количество кластеров
        self.__e = np.zeros(self.__number_of_clusters)  # количество дуг внутри кластера для каждого кластера
        self.__a = np.zeros(self.__number_of_clusters)  # количество дуг исходящих из кластера для каждого кластера
        self.__elementary_graph = clustering.elementary_graph   # список смежности графа
        self.__total_number_of_edges = 0    # общее количество ребер

    def __calculate_cluster_edge(self):
        """
        Подсчет общего кол-ва ребер, self.__e, self.__a для каждого кластера
        """
        for i in range(len(self.__elementary_graph)):
            for j in self.__elementary_graph[i]:
                self.__total_number_of_edges += 1
                if self.__cluster.resulting_clustering[i] == -1:
                    continue
                self.__a[self.__cluster.resulting_clustering[i]] += 1
                if self.__cluster.resulting_clustering[i] == self.__cluster.resulting_clustering[j]:
                    self.__e[self.__cluster.resulting_clustering[i]] += 1

    def __call__(self):
        self.__calculate_cluster_edge()
        score = 0
        for i in range(self.__number_of_clusters):
            score += self.__e[i] / self.__total_number_of_edges - (self.__a[i] / self.__total_number_of_edges) ** 2
        return score