import numpy as np


class FirstSearch():
    def __init__(self, number_of_vertices, list_of_adjacent_edges_of_graph):
        self.number_of_vertices = number_of_vertices
        self.list_of_adjacent_edges_of_graph = list_of_adjacent_edges_of_graph
        self._used = [False for _ in range(self.number_of_vertices)]


class TopologicalSort(FirstSearch):
    def __init__(self, number_of_vertices, list_of_adjacent_edges_of_graph):
        super().__init__(number_of_vertices, list_of_adjacent_edges_of_graph)
        self.order_topological_sort = []

    def _topological_sort_from_start_vertex(self, v):
        self._used[v] = True
        for to in self.list_of_adjacent_edges_of_graph[v]:
            if not self._used[to]:
                self._topological_sort_from_start_vertex(to)
        self.order_topological_sort.insert(0, v)

    def _run(self):
        for v in range(self.number_of_vertices):
            if not self._used[v]:
                self._topological_sort_from_start_vertex(v)



    def __call__(self):
        self._run()


class StronglyConnectedComponent(TopologicalSort):
    def __init__(self, number_of_vertices, list_of_adjacent_edges_of_graph, default_cluster_number):
        super().__init__(number_of_vertices, list_of_adjacent_edges_of_graph)
        self.list_of_adjacent_edges_of_transposed_graph = None
        self.default_cluster = default_cluster_number
        self.numbers_of_connected_component = [self.default_cluster for _ in range(self.number_of_vertices)]

    def __build_transposed_graph(self):
        self.list_of_adjacent_edges_of_transposed_graph = [[] for i in range(self.number_of_vertices)]
        for v in range(self.number_of_vertices):
            for to in self.list_of_adjacent_edges_of_graph[v]:
                self.list_of_adjacent_edges_of_transposed_graph[to].append(v)
        return self.list_of_adjacent_edges_of_transposed_graph

    def dfs(self, v, number_of_connected_component):
        self.numbers_of_connected_component[v] = number_of_connected_component
        for to in self.list_of_adjacent_edges_of_transposed_graph[v]:
            if self.numbers_of_connected_component[to] == self.default_cluster:
                self.dfs(to, number_of_connected_component)

    def find_connected_component(self):
        number_of_connected_component = self.default_cluster + 1
        for v in self.order_topological_sort:
            if self.numbers_of_connected_component[v] == self.default_cluster:
                self.dfs(v, number_of_connected_component)
                number_of_connected_component += 1

    def __call__(self):
        super()._run()
        self.__build_transposed_graph()
        self.find_connected_component()
        return np.array(self.numbers_of_connected_component)
