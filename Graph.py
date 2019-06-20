"""
The module of algorithms from graph theory
"""
import numpy as np


class FirstSearch():
    """The base class of traversal graph"""
    def __init__(self, number_of_vertices, list_of_adjacent_edges_of_graph):
        """
        Constructor
        :param number_of_vertices:  number of vertices of the graph
        :param list_of_adjacent_edges_of_graph: graph adjacency list
        """
        self.number_of_vertices = number_of_vertices
        self.list_of_adjacent_edges_of_graph = list_of_adjacent_edges_of_graph
        self._used = \
            [False for _ in range(self.number_of_vertices)]    # an array indicating whether the vertex was processed


class TopologicalSort(FirstSearch):
    """
    Topological sorting class
    """
    def __init__(self, number_of_vertices, list_of_adjacent_edges_of_graph):
        super().__init__(number_of_vertices, list_of_adjacent_edges_of_graph)
        self.order_topological_sort = []    # list of vertices in topological sort order

    def _topological_sort_from_start_vertex(self, v):
        """
        The function builds topological sorting from the vertex v.
        :param v: the vertex from which the traversal starts at this stage
        :return: None
        """
        self._used[v] = True
        for to in self.list_of_adjacent_edges_of_graph[v]:
            if not self._used[to]:
                self._topological_sort_from_start_vertex(to)
        self.order_topological_sort.insert(0, v)

    def _run(self):
        """
        The function starts topological sorting from unvisited vertices of the graph.
        :return: None
        """
        for v in range(self.number_of_vertices):
            if not self._used[v]:
                self._topological_sort_from_start_vertex(v)


    def __call__(self):
        """The function builds topological sorting on all vertices of the graph."""
        self._run()


class StronglyConnectedComponent(TopologicalSort):
    """
    Class for searching strongly connected graph of a graph.
    """
    def __init__(self, number_of_vertices, list_of_adjacent_edges_of_graph, default_cluster_number):
        """
        Constructor
        :param number_of_vertices: number of vertices of the graph
        :param list_of_adjacent_edges_of_graph: graph adjacency list
        :param default_cluster_number: default cluster number
        """
        super().__init__(number_of_vertices, list_of_adjacent_edges_of_graph)
        self.list_of_adjacent_edges_of_transposed_graph = None  # adjacency list of transposed graph
        self.default_cluster = default_cluster_number
        self.numbers_of_connected_component = \
            [self.default_cluster for _ in range(self.number_of_vertices)]  # numbers of components of
                                                                            # strongly connected graph

    def __build_transposed_graph(self):
        """
        The function builds a transposed graph.
        :return: adjacency list of transposed graph
        """
        self.list_of_adjacent_edges_of_transposed_graph = [[] for i in range(self.number_of_vertices)]
        for v in range(self.number_of_vertices):
            for to in self.list_of_adjacent_edges_of_graph[v]:
                self.list_of_adjacent_edges_of_transposed_graph[to].append(v)
        return self.list_of_adjacent_edges_of_transposed_graph

    def dfs(self, v, number_of_connected_component):
        """
        The function starts DFS from the vertex v.
        :param v: the vertex from which the depth to depth is started at this stage
        :param number_of_connected_component: the number of components is strongly connected with the current vertex
        :return: None
        """
        self.numbers_of_connected_component[v] = number_of_connected_component
        for to in self.list_of_adjacent_edges_of_transposed_graph[v]:
            if self.numbers_of_connected_component[to] == self.default_cluster:
                self.dfs(to, number_of_connected_component)

    def find_connected_component(self):
        """
        The function searches for strongly connected components for each vertex of the graph.
        :return: None
        """
        number_of_connected_component = self.default_cluster + 1
        for v in self.order_topological_sort:
            if self.numbers_of_connected_component[v] == self.default_cluster:
                self.dfs(v, number_of_connected_component)
                number_of_connected_component += 1

    def __call__(self):
        """
        The function builds topological sorting.
        The function searches for strongly connected components of the graph.
        :return: numbers of components strongly connected of the graph
        """
        super()._run()
        self.__build_transposed_graph()
        self.find_connected_component()
        return np.array(self.numbers_of_connected_component)
