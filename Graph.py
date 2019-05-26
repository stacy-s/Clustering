import numpy as np


class FirstSearch():
    """Базовый класс обходов графа"""
    def __init__(self, number_of_vertices, list_of_adjacent_edges_of_graph):
        """
        Конструктор
        :param number_of_vertices:  количество вершин в графе
        :param list_of_adjacent_edges_of_graph: список смежности графа
        """
        self.number_of_vertices = number_of_vertices
        self.list_of_adjacent_edges_of_graph = list_of_adjacent_edges_of_graph
        self._used = \
            [False for _ in range(self.number_of_vertices)]    # массив, показывающий была ли вершина обработана


class TopologicalSort(FirstSearch):
    """
    Класс топологической сортировки
    """
    def __init__(self, number_of_vertices, list_of_adjacent_edges_of_graph):
        super().__init__(number_of_vertices, list_of_adjacent_edges_of_graph)
        self.order_topological_sort = []    # список вершин в порядке топологической сортировки

    def _topological_sort_from_start_vertex(self, v):
        """
        Построение топологической сортировки (обход в глубину с сохранением времени окончания обработки вершин)
        :param v: вершина, из которой начинается обход на данном этапе
        :return: None
        """
        self._used[v] = True
        for to in self.list_of_adjacent_edges_of_graph[v]:
            if not self._used[to]:
                self._topological_sort_from_start_vertex(to)
        self.order_topological_sort.insert(0, v)

    def _run(self):
        """
        Запуск топологической сортировки из непосещенных вершин
        :return: None
        """
        for v in range(self.number_of_vertices):
            if not self._used[v]:
                self._topological_sort_from_start_vertex(v)



    def __call__(self):
        """Построение топологической сортировки из всех вершин графа"""
        self._run()


class StronglyConnectedComponent(TopologicalSort):
    """
    Класс поиска компонет сильной связности ориентированного графа.
    """
    def __init__(self, number_of_vertices, list_of_adjacent_edges_of_graph, default_cluster_number):
        """
        Конструктор
        :param number_of_vertices: количество вершин графа
        :param list_of_adjacent_edges_of_graph: список смежности графа
        :param default_cluster_number: номер кластера по умолчанию
        """
        super().__init__(number_of_vertices, list_of_adjacent_edges_of_graph)
        self.list_of_adjacent_edges_of_transposed_graph = None  # список смежности транспонированного графа
        self.default_cluster = default_cluster_number
        self.numbers_of_connected_component = \
            [self.default_cluster for _ in range(self.number_of_vertices)]  # номера компонет сильной связности графа

    def __build_transposed_graph(self):
        """
        Построение транспонированного графа
        :return: список смежности транспонированного графа
        """
        self.list_of_adjacent_edges_of_transposed_graph = [[] for i in range(self.number_of_vertices)]
        for v in range(self.number_of_vertices):
            for to in self.list_of_adjacent_edges_of_graph[v]:
                self.list_of_adjacent_edges_of_transposed_graph[to].append(v)
        return self.list_of_adjacent_edges_of_transposed_graph

    def dfs(self, v, number_of_connected_component):
        """
        Обход в глубину из вершины v
        :param v: вершина, из которой запускается обход в глубину на данном этапе
        :param number_of_connected_component: номер компонетны сильной связности текущей вершины
        :return: None
        """
        self.numbers_of_connected_component[v] = number_of_connected_component
        for to in self.list_of_adjacent_edges_of_transposed_graph[v]:
            if self.numbers_of_connected_component[to] == self.default_cluster:
                self.dfs(to, number_of_connected_component)

    def find_connected_component(self):
        """
        Поиск компонет сильной связности для каждой вершины графа
        :return: None
        """
        number_of_connected_component = self.default_cluster + 1
        for v in self.order_topological_sort:
            if self.numbers_of_connected_component[v] == self.default_cluster:
                self.dfs(v, number_of_connected_component)
                number_of_connected_component += 1

    def __call__(self):
        """
        Постороение топологической сортировки графа
        Поиск компонет сильной связности графа
        :return:
        """
        super()._run()
        self.__build_transposed_graph()
        self.find_connected_component()
        return np.array(self.numbers_of_connected_component)
