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
        self.cluster = clustering.cluster
        self.number_of_clusters = max(clustering.cluster.resulting_clustering) + 1
        self.e = np.zeros(self.number_of_clusters)
        self.a = np.zeros(self.number_of_clusters)
        self.elementary_graph = clustering.elementary_graph
        self.total_number_of_edges = 0

    def calculate_cluster_edge(self):
        """
        Подсчет общего кол-ва ребер, e, a для каждого кластера
        """
        for i in range(len(self.elementary_graph)):
            for j in self.elementary_graph[i]:
                self.total_number_of_edges += 1
                if self.cluster.resulting_clustering[i] == -1:
                    continue
                self.a[self.cluster.resulting_clustering[i]] += 1
                if self.cluster.resulting_clustering[i] == self.cluster.resulting_clustering[j]:
                    self.e[self.cluster.resulting_clustering[i]] += 1

    def __call__(self):
        self.calculate_cluster_edge()
        score = 0
        for i in range(self.number_of_clusters):
            score += self.e[i] / self.total_number_of_edges - (self.a[i] / self.total_number_of_edges) ** 2
        return score