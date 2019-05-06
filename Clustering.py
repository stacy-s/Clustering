import numpy as np
import math
import Cluster
import Graph
import copy
from geopy import distance


EPS = 1e-9


def distance2d(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def distance_great_circle(point1, point2):
    return distance.great_circle((point1[0], point1[1]), (point2[0], point2[1])).km * 1000


class Clustering():
    def __init__(self, eps, cluster: Cluster):
        self.eps = eps
        self.cluster = copy.deepcopy(cluster)
        self.number_of_vertices = self.cluster.coord_of_points.shape[0]
        self._default_cluster_number = 0
        self.cluster.resulting_clustering = np.full(self.number_of_vertices, self._default_cluster_number)
        self.distance = distance2d
        self.elementary_graph = None

    def distance_between_1st_coord(self, v, u):
        return abs(self.cluster.coord_of_points[v][0] - self.cluster.coord_of_points[u][0])

    def left_binary_search(self, left, right, x):
        """
        Бинарный поиск для поиска самого левого подходящего значения
        """
        lf = left - 1
        rg = right
        while rg - lf > 1:
            mid = lf + (rg - lf) // 2
            if self.distance_between_1st_coord(mid, x) > self.eps:
                lf = mid
            else:
                rg = mid
        return rg

    def right_binary_search(self, left, right, x):
        """
        Бинарный поиск для поиска самого правого подходящего значения
        """
        lf = left
        rg = right + 1
        while rg - lf > 1:
            mid = lf + (rg - lf) // 2
            if self.distance_between_1st_coord(mid, x) <= self.eps:
                lf = mid
            else:
                rg = mid
        return lf

    def _vertex_neighbor_list(self, vertex):
        neighbor = []
        left = self.left_binary_search(0, vertex, vertex)
        right = self.right_binary_search(vertex, self.number_of_vertices - 1, vertex)
        for to in range(left, right + 1):
            if to != vertex and self.distance(self.cluster.coord_of_points[vertex],
                                              self.cluster.coord_of_points[to]) + EPS <= self.eps:
                neighbor.append(to)
        return np.array(neighbor)

    def _make_elementary_graph(self):
        graph = [np.array([]) for _ in range(self.number_of_vertices)]
        for v in range(self.number_of_vertices):
            graph[v] = self._vertex_neighbor_list(v)
        return graph


class DBSCAN(Clustering):
    def __init__(self, eps, minPts, cluster: Cluster):
        super().__init__(eps, cluster)
        self.minPts = minPts
        self._used = np.full(self.number_of_vertices, False)

    def cluster_expansion(self, v, root_vertices, number_of_cluster):
        self.cluster.resulting_clustering[v] = number_of_cluster
        root_vertices = list(root_vertices)
        while len(root_vertices) != 0:
            to = root_vertices[0]
            root_vertices.pop(0)
            print("*",to)
            if not self._used[to]:
                self._used[to] = True
                if self.elementary_graph[to].shape[0] >= self.minPts:
                    root_vertices += list(self.elementary_graph[to])
            if self.cluster.resulting_clustering[to] == self._default_cluster_number:
                self.cluster.resulting_clustering[to] = number_of_cluster

    def clustering(self):
        number_of_next_cluster = self._default_cluster_number + 1
        for v in range(self.number_of_vertices):
            if self._used[v]:
                continue
            print(v)
            self._used[v] = True
            root_vertices = self.elementary_graph[v]
            if root_vertices.shape[0] < self.minPts:
                continue
            self.cluster_expansion(v, root_vertices, number_of_next_cluster)
            number_of_next_cluster += 1

    def __call__(self):
        self.elementary_graph = self._make_elementary_graph()
        self.clustering()


class DBSCANGreatCircle(DBSCAN):
    def __init__(self, eps, minPts, cluster: Cluster):
        DBSCAN.__init__(self, eps, minPts, cluster)
        self.distance = distance_great_circle


class K_MXT(Clustering):
    def __init__(self, eps, k, cluster: Cluster):
        np.random.seed(4000)
        super().__init__(eps, cluster)
        self.k = k
        self.graph_consists_of_k_arcs = None
        self.binary_representation_of_elementary_graph = None

    def build_binary_elementary_graph(self):
        self.binary_representation_of_elementary_graph = np.zeros((self.number_of_vertices, self.number_of_vertices), dtype=int)
        for v in range(self.number_of_vertices):
            for to in self.elementary_graph[v]:
                self.binary_representation_of_elementary_graph[v][to] = 1

    def weight(self, v, u):
        return np.sum(self.binary_representation_of_elementary_graph[v] *
                      self.binary_representation_of_elementary_graph[u])

    def counting_weights_of_neighbors(self, v):
        cnt_neighbor = self.elementary_graph[v].shape[0]
        cnt_information = 2
        weights = np.full((cnt_neighbor, cnt_information), 0)
        for i, to in enumerate(self.elementary_graph[v]):
            weights[i, 0] = self.weight(v, to)
            weights[i, 1] = to
            # print(weights)
        return weights

    def choose_k_arcs(self, v):
        def sorting_weight_in_descending_order(weights):
            index = weights[:, 0].argsort()
            weights = weights[index]
            weights = np.flip(weights, axis=0)
            return weights

        def get_weights_more_than_k_position(weights, k):
            if k == 0:
                return np.array([])
            weight_k = weights[k - 1][0]
            weights_more_that_k_position = weights[:, 0] > weight_k
            weight_better = weights[weights_more_that_k_position]
            return weight_better[:, 1]

        def get_weights_equal_k_position(weights, need2add, k):
            if k == 0:
                return np.array([])
            is_equal_weights = weights[:, 0] == weights[k - 1, 0]
            equal_weights = weights[is_equal_weights]
            np.random.shuffle(equal_weights)
            return equal_weights[:need2add, 1]

        weights = self.counting_weights_of_neighbors(v)
        weights = sorting_weight_in_descending_order(weights)
        k = min(self.k, len(self.elementary_graph[v]))
        weights_more_that_k_position = get_weights_more_than_k_position(weights, k)
        number_of_arcs_available = weights_more_that_k_position.shape[0]
        need2add = k - number_of_arcs_available
        result_vertices = list(np.concatenate((weights_more_that_k_position,
                                               get_weights_equal_k_position(weights, need2add, k))))
        return [int(x) for x in result_vertices]

    def build_graph_consists_of_k_arcs(self):
        self.graph_consists_of_k_arcs = [[] for _ in range(self.number_of_vertices)]
        for v in range(self.number_of_vertices):
            self.graph_consists_of_k_arcs[v] = self.choose_k_arcs(v)

    def __call__(self):
        self.elementary_graph = self._make_elementary_graph()
        self.build_binary_elementary_graph()
        self.build_graph_consists_of_k_arcs()
        g = Graph.StronglyConnectedComponent(self.number_of_vertices, self.graph_consists_of_k_arcs,
                                             self._default_cluster_number)
        self.cluster.resulting_clustering = g()


class K_MXTGreatCircle(K_MXT):
    def __init__(self, eps, k, cluster: Cluster):
        K_MXT.__init__(self, eps, k, cluster)
        self.distance = distance_great_circle


class K_MXTGauss(K_MXT):
    def __init__(self, eps, k, cluster: Cluster):
        super().__init__(eps, k, cluster)
        self.sigma = eps / 3

    def gauss(self, x):
        return 1 / (self.sigma * np.sqrt(np.pi * 2)) * np.exp(-x ** 2 / (2 * self.sigma ** 2))

    def weight(self, v, u):
        dist = self.distance(self.cluster.coord_of_points[v], self.cluster.coord_of_points[u])
        return self.gauss(dist) * super().weight(v, u)

class K_MXTGaussGreatCircle(K_MXTGauss):
    def __init__(self, eps, k, cluster: Cluster):
        K_MXTGauss.__init__(self, eps, k, cluster)
        self.distance = distance_great_circle