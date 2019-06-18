"""
Модуль кластеров.
Классы содержат координаты точек, правильную кластеризацию, если она известна, полученную кластеризацию.
Выполняется отрисовка кластеров.
"""
import numpy as np
import pandas as pd
import random
import tkinter
import folium
import matplotlib.pyplot as plt


class Cluster:
    """
    Класс объекта кластер для данных в декартовой системе координат.
    """
    def __init__(self, coords_of_points: np.ndarray, right_clustering: np.ndarray, resulting_clustering=None):
        """
        Конструктор
        :param coords_of_points: координаты точек для кластеризации в двумерной системе координат.
            В функции будет отсортированы с помощью функции __sort
        :param right_clustering: правильное разбиение точек на кластеры
        :param resulting_clustering: полученная кластеризация, по умолчанипю None
        """
        if resulting_clustering is None:
            resulting_clustering = []
        self.coord_of_points, self.right_clustering = self.__sort(coords_of_points, right_clustering)
        self.resulting_clustering = resulting_clustering
        self.__number_of_vertices = self.coord_of_points.shape[0]

    def __sort(self, coords_of_points, right_clustering):
        """
        Сортирует точки (массивы coords_of_points, right_clustering)  в порядке увеличения первой координаты,
        при равенстве первой координаты  в порядке увеличения второй коодинаты точки
        :param coords_of_points: координаты точек для кластеризации в двумерной системе координат
        :param right_clustering: правильное разбиение точек на кластеры
        :return: отсортированные массивы coords_of_points, right_clustering
        """
        index = coords_of_points[:, 0].argsort()
        coords_of_points = coords_of_points[index]
        if right_clustering is not None:
            right_clustering = right_clustering[index]
        return coords_of_points, right_clustering

    def _make_colors_of_clusters(self):
        """
        Определение цвета для каждого класстера
        """
        def rand_color():
            r = random.randint(0, 165)
            r_hex = hex(r)
            r_hex = r_hex[2:]
            if len(r_hex) < 2:
                return '0' + r_hex
            return r_hex
        random.seed(10)
        cnt_clusters = max(self.resulting_clustering) + 1
        colors_of_clusters = np.full(cnt_clusters, '0000000')
        for i in range(cnt_clusters):
            colors_of_clusters[i] = '#' + rand_color() + rand_color() + rand_color()
        return colors_of_clusters

    def view(self, title='', figsize=(7, 7)):
        """
        Функция рисования точек на координатной прямой. Одинаковым цветом помечаются точки, относящиеся к одному кластеру
        :param title: подпись к графику
        :param figsize: размер графика
        :return: None
        """
        colors_of_clusterts = self._make_colors_of_clusters()
        colors_of_points = []
        for i in range(len(self.resulting_clustering)):
            colors_of_points.append(colors_of_clusterts[self.resulting_clustering[i]])

        fig = plt.figure(figsize=figsize)
        plot = fig.add_subplot()
        plt.scatter(self.coord_of_points[:, 0], self.coord_of_points[:, 1], c=colors_of_points, s=10)
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def __str__(self):
        """Получение строковой информации об объекте."""
        return "coord_of_points: {0}\n" \
               "right_clustering: {1}\n" \
               "resulting_clustering:{2}".format(self.coord_of_points, self.right_clustering, self.resulting_clustering)


class ClusterGreatCircles(Cluster):
    """
        Класс объекта кластер для данных на поверхности шара.
    """
    def __init__(self, filepath, filename,  resulting_clustering=None):
        """
        Конструктор
        :param filepath: путь до файла с долготами и широтами точек на поверхности Земли.
        :param filename: имя файла с расшинением .csv. Файл досжен содержать столбцы
            longitude (догота), latitude (широта), owner (владелец)
        :param resulting_clustering: полученная кластеризация, по умолчанипю None
        """

        self.filepath = filepath
        self.filename = filename
        super().__init__(self.__load(), None, resulting_clustering)
        self.__number_of_vertices = self.coord_of_points.shape[0]

    def __load(self):
        """
        Загружает данные из заданного пути (self.filepath + self.filename).
        Оставляет только первое вхождение записи с одинаковыми longitude, latitude, owner
        :return: coords_of_points
        """
        df = pd.read_csv(self.filepath + self.filename)
        df['latitude'].astype('float64')
        df['longitude'].astype('float64')
        df = df.drop_duplicates(['longitude', 'latitude', 'owner'], keep='first')
        dfloc = df[['longitude', 'latitude']]
        coords_of_points = dfloc.values
        coords_of_points = coords_of_points[coords_of_points[:, 0].argsort()]
        return coords_of_points

    def _make_colors_of_clusters(self, default_cluster_number):
        """
            Определение цвета для каждого класстера
        """
        colors_of_clusters = super()._make_colors_of_clusters()
        red = "#FF0000"
        for i in range(self.__number_of_vertices):
            if colors_of_clusters[self.resulting_clustering[i]] != red and \
                    (self.resulting_clustering[i] == default_cluster_number or
                    np.count_nonzero(self.resulting_clustering == self.resulting_clustering[i]) <= 1):
                colors_of_clusters[self.resulting_clustering[i]] = red
        return colors_of_clusters

    def view_at_map(self, latitude, longitude, filename_of_map, default_cluster_number=0):
        """
        Сохраняет файл с расширением .html, в котором нарисованы точки на карте города.
        Одинаковым цветом помечаются точки, принадлежашие одному и тому же кластеру.
        Красным цветом обозначается точки, являющиеся шумами.
        :param latitude: широта города
        :param longitude: долгота города
        :param filename_of_map: имя файла, полученной карты (без расширения)
        :param default_cluster_number: номер кластера по умолчанию
        :return: None
        """
        fmap = folium.Map([latitude, longitude])
        folium.TileLayer(
            tiles='https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_nolabels/{z}/{x}/{y}{r}.png',
            attr='My').add_to(fmap)
        folium.TileLayer(
            tiles='https://cartodb-basemaps-{s}.global.ssl.fastly.net/rastertiles/voyager_nolabels/{z}/{x}/{y}{r}.png',
            attr='My').add_to(fmap)
        folium.LayerControl().add_to(fmap)
        colors_of_clusters = self._make_colors_of_clusters(default_cluster_number)
        for i in range(self.__number_of_vertices):
            folium.CircleMarker([self.coord_of_points[i, 1], self.coord_of_points[i, 0]],
                                radius=2, fill=True, color=colors_of_clusters[self.resulting_clustering[i]],
                                popup=str(self.resulting_clustering[i])).add_to(fmap)
        fmap.save(filename_of_map + '.html')


    def __str__(self):
        """Получение строковой информации об объекте."""
        return "filename: {0}\n".format(self.filename) + super().__str__()
