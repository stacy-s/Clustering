"""
Модуль алгоритмов кластеризации
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
    Функция подсчета расстояния между двумя точками в декартовой системе координат.
    :param point1: координаты по x, y первой точки
    :param point2: координаты по x, y второй точки
    :return: расстояние между двумя точками
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def distance_between_1st_coord2d(point1, point2):
    """
    Функция подсчета расстояния между двумя точками в декартовой системе координат с учетом только первой координаты.
    :param point1: координаты по x, y первой точки
    :param point2: координаты по x, y второй точки
    :return: расстояние между двумя вершинами по первой координате
    """
    return abs(point1[0] - point2[0])


def distance_great_circle(point1, point2):
    """
    Функция подсчета расстояния между двумя точками на поверхности Земли.
    :param point1: долгота, широта первой точки
    :param point2: долгота, широта второй точки
    :return: расстояние между двумя точками
    """
    return distance.great_circle((point1[0], point1[1]), (point2[0], point2[1])).km * 1000


def distance_between_1st_coord_great_circle(point1, point2):
    """
    Функция подсчета расстояния между двумя точками на поверхности Земли с учетом только первой координаты.
    :param point1: долгота, широта первой точки
    :param point2: долгота, широта второй точки
    :return: расстояние между двумя вершинами по первой координате
    """
    return distance.great_circle((point1[0], 0), (point2[0], 0)).km * 1000


class Clustering():
    """
    Базовый класс алгоритма кластеризации графа
    """

    def __init__(self, eps, cluster: Cluster.Cluster):
        """
        Конструктор
        :param eps: расстояние. Точки находящиеся на расстояние не превосходящим eps будут соединяться ребром
        :param cluster: объект класса Cluster.Cluster, с информацией о точках
        """
        self.eps = eps
        self.cluster = copy.deepcopy(cluster)
        self.number_of_vertices = self.cluster.coord_of_points.shape[0]  # кол-во вершин графа
        self._default_cluster_number = 0  # начальное значение номера кластера
        self.cluster.resulting_clustering = np.full(self.number_of_vertices, self._default_cluster_number)
        # результат кластеризации
        self._distance = distance2d  # функция подсчета расстояния (для декартовой системы координат)
        self.elementary_graph = None  # начальный граф
        self._distance_between_1st_coord = \
            distance_between_1st_coord2d  # функция подсчета расстояния по координате x (декартовая системы координат)

    def _left_binary_search(self, left, right, x):
        """
        Бинарный поиск для поиска самой левой (первой) подходящей точки,
        для которой расстояние по первой координате до заданной точки (x) не привосходит eps
        :param left: левая граница начала поиска
        :param right: правая граница наччала поиска
        :param x: номер точки, для которой нужно найти левую границу
        :return: номер самой первой (левой) точки,
            для которой расстояние по первой координате по x не превосходит eps
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
        Бинарный поиск для поиска самой правой (последней) подходящей точки,
        для которой расстояние по первой координате до заданной точки (x) не привосходит eps
        :param left: левая граница начала поиска
        :param right: правая граница наччала поиска
        :param x: номер точки, для которой нужно найти правую границу
        :return: номер самой последней (правой) точки,
            для которой расстояние по первой координате по x не превосходит eps
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
        Находит всех соседей заданной вершины
        :param vertex: вершина, для которй нужно найти всех соседей
        :return: np.array всех соседей верины vertex
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
        Построение графа, в которой расстояние между двумя вершина не перевосходит eps.
        :return: список смежности (список массивов numpy), полученного графа.
        """
        graph = [np.array([]) for _ in range(self.number_of_vertices)]
        for v in range(self.number_of_vertices):
            graph[v] = self._vertex_neighbor_list(v)
        return graph

    def counting_number_of_elements_in_list(self, lst):
        """
        Подсчет количества элементов в двумерном списке
        :param lst: список двумерный
        :return: количество элементов в списке
        """
        number_of_elements = 0
        for i in range(len(lst)):
            number_of_elements += len(lst[i])
        return number_of_elements

    def counting_number_of_arcs_in_elementary_graph(self):
        """
        Подсчет количества дуг в начальном графе
        :return: количество дуг в начальном графе
        """
        return self.counting_number_of_elements_in_list(self.elementary_graph)

class K_MXT(Clustering):
    """
    Алгоритм кластеризации k-MXT для точек в декартовой системе координат.
    """

    def __init__(self, eps, k, cluster: Cluster.Cluster):
        """
        Конструктор
        :param eps: расстояние. Точки находящиеся на расстояние не превосходящим eps будут соединяться ребром
        :param k: значение параметра k. Не более k исходящих дуг останется из каждой вершины
        :param cluster: объект класса Cluster.Cluster, с информацией о точках
        """
        np.random.seed(4000)
        super().__init__(eps, cluster)
        self.k = k
        self.graph_consists_of_k_arcs = None  # граф в  не более, чем k исходящими дугами из одной вершины
        self.binary_representation_of_elementary_graph = None  # бинарная матрица смежности начального графа

    def _build_binary_elementary_graph(self):
        """
        Построение бинарной матрицы смежности начального графа
        :return: None
        """
        self.binary_representation_of_elementary_graph = np.zeros((self.number_of_vertices, self.number_of_vertices),
                                                                  dtype=int)
        for v in range(self.number_of_vertices):
            for to in self.elementary_graph[v]:
                self.binary_representation_of_elementary_graph[v][to] = 1

    def _weight(self, v, u):
        """
        Подсчет веса дуги из v в u
        :param v: первая вершина
        :param u: вторая вершина
        :return: вес дуги (v, u)
        """
        return np.sum(self.binary_representation_of_elementary_graph[v] *
                      self.binary_representation_of_elementary_graph[u])

    def _counting_weights_of_neighbors(self, v):
        """
        Подсчет весов дуг из вершины v до всех соседей этой вершины
        :param v: вершина, для которой считается веса всех исходящих дуг
        :return:
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
        Выбор не более k исходящих дуг из вершины v.
        Дуги выбираются по наибольшему весу.
        При одинаковом весе дуги выбираются случайно.
        :param v: вершина, для которой выбираются дуги
        :return: список исходящих вершин
        """

        def sorting_weight_in_descending_order(weights):
            """
            Сортировка вершин в невозрастающем порядке весов
            :param weights:  numpy массив из весов вершин и номеров вершин
            :return: отсортированный массив врешин
            """
            index = weights[:, 0].argsort()
            weights = weights[index]
            weights = np.flip(weights, axis=0)
            return weights

        def get_weights_more_than_k_position(weights, k):
            """
            Выбор вершин с весами большими, чем с весом на позиции k - 1 в массиве weights
            :param weights: numpy массив из весов вершин и номеров вершин
            :param k: кол-во вершин, которое нужно взять. k не привосходит размера массива weights
            :return: номера выбранных вершин
            """
            if k == 0:
                return np.array([])
            weight_k = weights[k - 1][0]
            weights_more_that_k_position = weights[:, 0] > weight_k
            weight_better = weights[weights_more_that_k_position]
            return weight_better[:, 1]

        def get_weights_equal_k_position(weights, need2add, k):
            """
            Выбор вершин с одинаковыми весами.
            :param weights: numpy массив из весов вершин и номеров вершин
            :param need2add: количество вершин, которые нужно получить,
            :param k: общее кол-во вершин, которое нужно взять
            :return: нужное количество номеров вершин с одинаковыми весами
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
        Построение графа, в котором не более k исходящих дуг.
        :return: None
        """
        self.graph_consists_of_k_arcs = [[] for _ in range(self.number_of_vertices)]
        for v in range(self.number_of_vertices):
            self.graph_consists_of_k_arcs[v] = self._choose_k_arcs(v)

    def counting_number_of_arcs_in_graph_consists_of_k_arcs(self):
        """
        Подсчет количества дуг в графе, в котором не более k исходящих дуг.
        :return: количество дуг в графе, в котором не более k исходящих дуг
        """
        return self.counting_number_of_elements_in_list(self.graph_consists_of_k_arcs)

    def __call__(self):
        """
        Построение начального графа.
        Построение графа с количеством исходящих дуг не более k
        Выделение компонет сильной связанности полученного графа с не более k исходящими дугами (выделение кластеров)
        :return:  None
        """
        self.elementary_graph = self._make_elementary_graph()
        self._build_binary_elementary_graph()
        self._build_graph_consists_of_k_arcs()
        g = Graph.StronglyConnectedComponent(self.number_of_vertices, self.graph_consists_of_k_arcs,
                                             self._default_cluster_number)
        self.cluster.resulting_clustering = g()

    def __str__(self):
        """Получение строковой информации об объекте."""
        return "{0}-MXT, eps = {1:.1f}".format(self.k, self.eps)


class K_MXTGreatCircle(K_MXT):
    """
    Класс алгоритма k-MXT для точек на поверхности шара
    """

    def __init__(self, eps, k, cluster: Cluster.Cluster):
        """
        Конструктор
        :param eps: расстояние. Точки находящиеся на расстояние не превосходящим eps будут соединяться ребром
        :param k: значение параметра k. Не более k исходящих дуг останется из каждой вершины
        :param cluster: объект класса Cluster.Cluster, с информацией о точках
        """
        K_MXT.__init__(self, eps, k, cluster)
        self._distance = distance_great_circle  # функция подсчета расстояния между двумя точками на поверхности шара
        self._distance_between_1st_coord = \
            distance_between_1st_coord_great_circle  # функция подсчета расстояния по долготе (на повехности Земли)


class K_MXTGauss(K_MXT):
    """
    Класс алгоритма k-MXT-Gauss для точек в декартовой системе координат
    """

    def __init__(self, eps, k, cluster: Cluster.Cluster):
        """
        Конструктор
        :param eps: расстояние. Точки находящиеся на расстояние не превосходящим eps будут соединяться ребром
        :param k: значение параметра k. Не более k исходящих дуг останется из каждой вершины
        :param cluster: объект класса Cluster.Cluster, с информацией о точках
        """
        super().__init__(eps, k, cluster)
        self.sigma = eps / 3  # стандартное отклонение плотности распределения Гаусса

    def gauss(self, x):
        """
        Функция подсчета плотности распределения Гаусса при заданном x (расстояние между двумя точками)
        :param x: расстояние между двумя точками
        :return: значение функции для заданного x
        """

        return 1 / (self.sigma * np.sqrt(np.pi * 2)) * np.exp(-x ** 2 / (2 * self.sigma ** 2))

    def _weight(self, v, u):
        """
        Функция посчета весов дуги (v, u)
        :param v: первая вершина
        :param u: вторая вершина
        :return: вес дуги (v, u)
        """
        dist = self._distance(self.cluster.coord_of_points[v], self.cluster.coord_of_points[u])
        return self.gauss(dist) * super()._weight(v, u)

    def __str__(self):
        """Получение строковой информации об объекте."""
        return "{0}-MXT-Gauss, eps = {1:.1f}".format(self.k, self.eps)


class K_MXTGaussGreatCircle(K_MXTGauss):
    """
        Класс алгоритма k-MXT-Gauss для точек на поверхности Земли
        """

    def __init__(self, eps, k, cluster: Cluster.Cluster):
        """
        Конструктор
        :param eps: расстояние. Точки находящиеся на расстояние не превосходящим eps будут соединяться ребром
        :param k: значение параметра k. Не более k исходящих дуг останется из каждой вершины
        :param cluster: объект класса Cluster.Cluster, с информацией о точках
        """
        K_MXTGauss.__init__(self, eps, k, cluster)
        self._distance = distance_great_circle  # функция подсчета расстояния между двумя точками на поверхности шара
        self._distance_between_1st_coord = \
            distance_between_1st_coord_great_circle  # функция подсчета расстояния по долготе (на повехности Земли)
