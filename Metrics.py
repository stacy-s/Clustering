"""
Clustering Quality Assessment Metrics Module
"""
import Cluster
import Clustering
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np


class ARI:
    """
    ARI evaluation metric
    """
    def __init__(self, clustering: Clustering.Clustering):
        """
        Constructor
        :param clustering: is an object of the clustering algorithm class
        """
        self.cluster = clustering.cluster

    def __call__(self):
        """
        ARI metric counting
        :return: the value of the ARI metric
        """
        return adjusted_rand_score(self.cluster.right_clustering, self.cluster.resulting_clustering)


class Modularity:
    """
    Modularity evaluation metric
    """
    def __init__(self, clustering: Clustering.Clustering):
        """
        Constructor
        :param clustering: is an object of the clustering algorithm class
        """
        self.__cluster = clustering.cluster     # object of class Cluster.Cluster
        self.__number_of_clusters = max(clustering.cluster.resulting_clustering) + 1    # number of clusters
        self.__e = np.zeros(self.__number_of_clusters)  # number of arcs inside the cluster for each cluster
        self.__a = np.zeros(self.__number_of_clusters)  # number of arcs outgoing from the cluster for each cluster
        self.__elementary_graph = clustering.elementary_graph   # graph adjacency list
        self.__total_number_of_edges = 0    # total number of edges

    def __calculate_cluster_edge(self):
        """
        Counting the total number of edges, self .__ e, self .__ a for each cluster
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
        """
        Modularity metric counting
        :return: the value of the Modularity metric
        """
        self.__calculate_cluster_edge()
        score = 0
        for i in range(self.__number_of_clusters):
            score += self.__e[i] / self.__total_number_of_edges - (self.__a[i] / self.__total_number_of_edges) ** 2
        return score