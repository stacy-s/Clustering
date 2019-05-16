import Cluster
import Clustering
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np


class ARI():
    def __init__(self, cluster: Cluster.Cluster):
        self.cluster = cluster

    def __call__(self):
        return adjusted_rand_score(self.cluster.right_clustering, self.cluster.resulting_clustering)


class Modularity():
    """
    Метрика оценки modularity.
    self.all_edge - кол-во всех ребер графа
    self.num_of_clusters - кол-во кластеров
    self.e = массив e_kk
    self.a = массив a_k
    """

    def __init__(self, clustering: Clustering.Clustering):
        self.__cluster = clustering.cluster
        self.__number_of_clusters = max(clustering.cluster.resulting_clustering) + 1
        self.__e = np.zeros(self.__number_of_clusters)
        self.__a = np.zeros(self.__number_of_clusters)
        self.__elementary_graph = clustering.elementary_graph
        self.__total_number_of_edges = 0

    def __calculate_cluster_edge(self):
        """
        Подсчет общего кол-ва ребер, e, a для каждого кластера
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