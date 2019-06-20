"""
Module clustering algorithms
"""
import numpy as np
import math
import Cluster
import Graph
import copy
from geopy import distance

EPS = 1e-9


def distance2d(point1, point2):
    """
    The function of calculating the distance between two points in the Cartesian coordinate system.
    :param point1: x, y coordinates of the first point
    :param point2: x, y coordinates of the second point
    :return: distance between two points
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def distance_between_1st_coord2d(point1, point2):
    """
    The function of calculating the distance between two points in the Cartesian coordinate system,
    taking into account only the first coordinate
    :param point1: x, y coordinates of the first point
    :param point2: x, y coordinates of the second point
    :return: distance between two vertices on the first coordinate
    """
    return abs(point1[0] - point2[0])


def distance_great_circle(point1, point2):
    """
    The function of calculating the distance between two points on the surface of the Earth
    :param point1: longitude, latitude of the first point
    :param point2: longitude, latitude of the second point
    :return: distance between two points
    """
    return distance.great_circle((point1[1], point1[0]), (point2[1], point2[0])).km * 1000


def distance_between_1st_coord_great_circle(point1, point2):
    """
    The function of calculating the distance between two points on the surface of the Earth,
    taking into account only the first coordinate.
    :param point1: longitude, latitude of the first point
    :param point2: longitude, latitude of the second point
    :return: distance between two vertices on the first coordinate
    """
    return distance.great_circle((0, point1[0]), (0, point2[0])).km * 1000


class Clustering():
    """
    The base class of the cluster clustering algorithm
    """

    def __init__(self, eps, cluster: Cluster.Cluster):
        """
        Constructor
        :param eps: if the points are located at a distance not exceeding eps,
            then the points will be connected by an edge
        :param cluster: object of class Cluster.Cluster, with information about points
        """
        self.eps = eps
        self.cluster = copy.deepcopy(cluster)
        self.number_of_vertices = self.cluster.coord_of_points.shape[0]  # number of vertices of the graph
        self._default_cluster_number = 0  # the initial value of the cluster number
        self.cluster.resulting_clustering = np.full(self.number_of_vertices, self._default_cluster_number)
        # clustering result
        self._distance = distance2d  # distance calculation function (for Cartesian coordinate system)
        self.elementary_graph = None  # initial graph
        self._distance_between_1st_coord = \
            distance_between_1st_coord2d  # the function of counting the distance along the x coordinate
        # (Cartesian coordinate system)

    def _left_binary_search(self, left, right, x):
        """
        The binary search returns the leftmost (first) point for
        which the distance in the first coordinate to the given point (x) does not correspond to eps.
        :param left: left limit of the start of the search
        :param right: right edge of the start of the search
        :param x: the number of the point for which you want to find the left border
        :return: the number of the very first (left) point,
            for which the distance along the first coordinate in x does not exceed eps
        """
        lf = left - 1
        rg = right
        while rg - lf > 1:
            mid = lf + (rg - lf) // 2
            if self._distance_between_1st_coord(self.cluster.coord_of_points[mid],
                                                self.cluster.coord_of_points[x]) > self.eps:
                lf = mid
            else:
                rg = mid
        return rg

    def _right_binary_search(self, left, right, x):
        """
        The binary search returns the rightmost (last) point for
        which the distance in the first coordinate to the given point (x) does not correspond to eps.
        :param left: left limit of the start of the search
        :param right: right edge of the start of the search
        :param x: the number of the point for which you want to find the right border
        :return: the number of the most recent (right) point
            for which the distance along the first coordinate in x does not exceed eps
        """
        lf = left
        rg = right + 1
        while rg - lf > 1:
            mid = lf + (rg - lf) // 2
            if self._distance_between_1st_coord(self.cluster.coord_of_points[mid],
                                                self.cluster.coord_of_points[x]) <= self.eps:
                lf = mid
            else:
                rg = mid
        return lf

    def _vertex_neighbor_list(self, vertex):
        """
        The function finds all neighbors of a given vertex.
        :param vertex: the vertex for which you need to find all the neighbors
        :return: np.array all neighbors vertex
        """
        neighbor = []
        left = self._left_binary_search(0, vertex, vertex)
        right = self._right_binary_search(vertex, self.number_of_vertices - 1, vertex)
        for to in range(left, right + 1):
            if to != vertex and self._distance(self.cluster.coord_of_points[vertex],
                                               self.cluster.coord_of_points[to]) + EPS <= self.eps:
                neighbor.append(to)
        return np.array(neighbor)

    def _make_elementary_graph(self):
        """
        The function builds a graph in which the distance between two vertices does not exceed eps.
        :return: adjacency list (list of numpy arrays) obtained graph
        """
        graph = [np.array([]) for _ in range(self.number_of_vertices)]
        for v in range(self.number_of_vertices):
            graph[v] = self._vertex_neighbor_list(v)
        return graph

    def counting_number_of_elements_in_list(self, lst):
        """
        The function counts the number of items in a two-dimensional list.
        :param lst: two-dimensional list
        :return: number of items in the list
        """
        number_of_elements = 0
        for i in range(len(lst)):
            number_of_elements += len(lst[i])
        return number_of_elements

    def counting_number_of_arcs_in_elementary_graph(self):
        """
        The function counts the number of arcs in the initial graph.
        :return: the number of arcs in the initial graph
        """
        return self.counting_number_of_elements_in_list(self.elementary_graph)


class K_MXT(Clustering):
    """
    The k-MXT clustering algorithm for points in the Cartesian coordinate system.
    """

    def __init__(self, eps, k, cluster: Cluster.Cluster):
        """
        Constructor
        :param eps: if the points are located at a distance not exceeding eps,
            then the points will be connected by an edge
        :param k: the value of the parameter k. No more than k outgoing arcs will remain from each vertex.
        :param cluster: object of class Cluster.Cluster, with information about points
        """
        np.random.seed(4000)
        super().__init__(eps, cluster)
        self.k = k
        self.graph_consists_of_k_arcs = None  # graph in no more than k outgoing arcs from one vertex
        self.binary_representation_of_elementary_graph = None  # binary adjacency matrix of the initial graph

    def _build_binary_elementary_graph(self):
        """
        The function builds a binary adjacency matrix of the original graph.
        :return: None
        """
        self.binary_representation_of_elementary_graph = np.zeros((self.number_of_vertices, self.number_of_vertices),
                                                                  dtype=int)
        for v in range(self.number_of_vertices):
            for to in self.elementary_graph[v]:
                self.binary_representation_of_elementary_graph[v][to] = 1

    def _weight(self, v, u):
        """
        The function calculates the weight of the arc from v to u.
        :param v: first vertex
        :param u: second vertex
        :return: arc weight (v, u)
        """
        return np.sum(self.binary_representation_of_elementary_graph[v] *
                      self.binary_representation_of_elementary_graph[u])

    def _counting_weights_of_neighbors(self, v):
        """
        The function calculates the weights of the arcs from the vertex v to all the neighbors of this vertex.
        :param v: the vertex for which the weights of all outgoing arcs are considered
        :return: the weight of all arcs emanating from the vertex v
        """
        cnt_neighbor = self.elementary_graph[v].shape[0]
        cnt_information = 2
        weights = np.full((cnt_neighbor, cnt_information), 0)
        for i, to in enumerate(self.elementary_graph[v]):
            weights[i, 0] = self._weight(v, to)
            weights[i, 1] = to
        return weights

    def _choose_k_arcs(self, v):
        """
        The function selects at most k outgoing arcs from the vertex v.
        The function selects the arc with the greatest weight.
        The function selects the weight of the arcs with the same weights randomly.
        :param v: vertex for which arcs are selected
        :return: list of outgoing vertices
        """

        def sorting_weight_in_descending_order(weights):
            """
            The function sorts vertices in ascending order of weights.
            :param weights: numpy array of vertex weights and vertex numbers
            :return: sorted array of vertices
            """
            index = weights[:, 0].argsort()
            weights = weights[index]
            weights = np.flip(weights, axis=0)
            return weights

        def get_weights_more_than_k_position(weights, k):
            """
            The function selects arcs with weights greater than the weight of the arc in position k - 1
            :param weights: numpy array of vertex weights and vertex numbers
            :param k: number of vertices to take. k does not exceed the size of the weights array
            :return: numbers of selected vertices
            """
            if k == 0:
                return np.array([])
            weight_k = weights[k - 1][0]
            weights_more_that_k_position = weights[:, 0] > weight_k
            weight_better = weights[weights_more_that_k_position]
            return weight_better[:, 1]

        def get_weights_equal_k_position(weights, need2add, k):
            """
            The function selects vertices with the same weights.
            :param weights: numpy array of vertex weights and vertex numbers
            :param need2add: the number of vertices you need to get
            :param k: total number of vertices to take
            :return: the required number of vertex numbers with the same weights
            """
            if k == 0:
                return np.array([])
            is_equal_weights = weights[:, 0] == weights[k - 1, 0]
            equal_weights = weights[is_equal_weights]
            np.random.shuffle(equal_weights)
            return equal_weights[:need2add, 1]

        weights = self._counting_weights_of_neighbors(v)
        weights = sorting_weight_in_descending_order(weights)
        k = min(self.k, len(self.elementary_graph[v]))
        weights_more_that_k_position = get_weights_more_than_k_position(weights, k)
        number_of_arcs_available = weights_more_that_k_position.shape[0]
        need2add = k - number_of_arcs_available
        result_vertices = list(np.concatenate((weights_more_that_k_position,
                                               get_weights_equal_k_position(weights, need2add, k))))
        return [int(x) for x in result_vertices]

    def _build_graph_consists_of_k_arcs(self):
        """
        The function builds a graph in which no more than k outgoing arcs
        :return: None
        """
        self.graph_consists_of_k_arcs = [[] for _ in range(self.number_of_vertices)]
        for v in range(self.number_of_vertices):
            self.graph_consists_of_k_arcs[v] = self._choose_k_arcs(v)

    def counting_number_of_arcs_in_graph_consists_of_k_arcs(self):
        """
        The function calculates the number of arcs in the graph in which there are no more than k outgoing arcs.
        :return: the number of arcs in the graph in which no more than k outgoing arcs
        """
        return self.counting_number_of_elements_in_list(self.graph_consists_of_k_arcs)

    def __call__(self):
        """
        The function builds the source graph.
        The function builds a graph with not more than k outgoing arcs from each vertex.
        The function selects the strong connected components of the resulting graph with a maximum of k outgoing arcs.
            (select of clusters)
        :return:  None
        """
        self.elementary_graph = self._make_elementary_graph()
        self._build_binary_elementary_graph()
        self._build_graph_consists_of_k_arcs()
        g = Graph.StronglyConnectedComponent(self.number_of_vertices, self.graph_consists_of_k_arcs,
                                             self._default_cluster_number)
        self.cluster.resulting_clustering = g()

    def __str__(self):
        """The function provides information about the object."""
        return "{0}-MXT, eps = {1:.1f}".format(self.k, self.eps)


class K_MXTGreatCircle(K_MXT):
    """
    Algorithm class k-MXT for points on the surface of the Earth
    """

    def __init__(self, eps, k, cluster: Cluster.Cluster):
        """
        Constructor
        :param eps: if the points are located at a distance not exceeding eps,
            then the points will be connected by an edge
        :param k: the value of the parameter k. No more than k outgoing arcs from each vertex
        :param cluster: object of class Cluster.Cluster, with information about points
        """
        K_MXT.__init__(self, eps, k, cluster)
        self._distance = distance_great_circle  # the function of calculating the distance between two points
        # on the surface of the Earth.
        self._distance_between_1st_coord = \
            distance_between_1st_coord_great_circle  # longitude distance counting function (on the surface of the
        # Earth)


class K_MXTGauss(K_MXT):
    """
    The class of the k-MXT-Gauss algorithm for points in the Cartesian coordinate system
    """
    def __init__(self, eps, k, cluster: Cluster.Cluster):
        """
        Constructor
        :param eps: if the points are located at a distance not exceeding eps,
            then the points will be connected by an edge
        :param k: the value of the parameter k. No more than k outgoing arcs from each vertex
        :param cluster: object of class Cluster.Cluster, with information about points
        """
        super().__init__(eps, k, cluster)
        self.sigma = eps / 3  # Gaussian distribution density standard deviation

    def gauss(self, x):
        """
        The function of calculating the density of the Gaussian distribution at a given x
        (the distance between two points)
        :param x: distance between two points
        :return: the value of the function for the given x
        """
        return 1 / (self.sigma * np.sqrt(np.pi * 2)) * np.exp(-x ** 2 / (2 * self.sigma ** 2))

    def _weight(self, v, u):
        """
        The function of counting the weights of the arc (v, u)
        :param v: first peak
        :param u: second peak
        :return: arc weight (v, u)
        """
        dist = self._distance(self.cluster.coord_of_points[v], self.cluster.coord_of_points[u])
        return self.gauss(dist) * super()._weight(v, u)

    def __str__(self):
        """The function provides information about the object."""
        return "{0}-MXT-Gauss, eps = {1:.1f}".format(self.k, self.eps)


class K_MXTGaussGreatCircle(K_MXTGauss):
    """
    Algorithm class k-MXT-Gauss for points on the surface of the Earth
    """
    def __init__(self, eps, k, cluster: Cluster.Cluster):
        """
        Constructor
        :param eps: if the points are located at a distance not exceeding eps,
            then the points will be connected by an edge
        :param k: the value of the parameter k. No more than k outgoing arcs from each vertex
        :param cluster: object of class Cluster.Cluster, with information about points
        """
        K_MXTGauss.__init__(self, eps, k, cluster)
        self._distance = distance_great_circle  # the function of calculating the distance between
        # two points on the surface of the ball
        self._distance_between_1st_coord = \
            distance_between_1st_coord_great_circle  # the function of calculating the distance in longitude
        # (on the surface of the Earth)