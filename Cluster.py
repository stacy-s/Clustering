import numpy as np
import pandas as pd
import random
import tkinter
import folium
import matplotlib.pyplot as plt

"""

"""


class Cluster:
    def __init__(self, coords_of_points : np.array, right_clustering : np.array, resulting_clustering=[]):
        self.coord_of_points, self.right_clustering = self.sort(coords_of_points, right_clustering)
        self.resulting_clustering = resulting_clustering
        self.__number_of_vertices = self.coord_of_points.shape[0]

    def sort(self, coords_of_points, right_clustering):
        index = coords_of_points[:, 0].argsort()
        coords_of_points = coords_of_points[index]
        if right_clustering is not None:
            right_clustering = right_clustering[index]
        return coords_of_points, right_clustering

    def make_colors_of_clusters(self):
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

        cnt_clusters = max(self.resulting_clustering) + 1
        colors_of_clusters = np.full(cnt_clusters, '0000000')
        for i in range(cnt_clusters):
            colors_of_clusters[i] = '#' + rand_color() + rand_color() + rand_color()
        return colors_of_clusters

    def view(self, title='', figsize=(7, 7)):
        colors_of_clusterts = self.make_colors_of_clusters()
        colors_of_points = []
        for i in range(len(self.resulting_clustering)):
            colors_of_points.append(colors_of_clusterts[self.resulting_clustering[i]])

        fig = plt.figure(figsize=figsize)
        plot = fig.add_subplot()
        plt.scatter(self.coord_of_points[:, 0], self.coord_of_points[:, 1], c=colors_of_points, s=10)
        plt.title(title)
        plt.show()

    def __str__(self):
        return "coord_of_points: {0}\n" \
               "right_clustering: {1}\n" \
               "resulting_clustering:{2}".format(self.coord_of_points, self.right_clustering, self.resulting_clustering)


class ClusterGreatCircles(Cluster):
    def __init__(self, filepath, filename,  resulting_clustering=None):
        self.filepath = filepath
        self.filename = filename
        super().__init__(self.load(), None, resulting_clustering)
        self.__number_of_vertices = self.coord_of_points.shape[0]

    def load(self):
        df = pd.read_csv(self.filepath + self.filename)
        df['latitude'].astype('float64')
        df['longitude'].astype('float64')
        df = df.drop_duplicates(['longitude', 'latitude', 'owner'], keep='first')
        dfloc = df[['longitude', 'latitude']]
        coords_of_points = dfloc.values
        coords_of_points = coords_of_points[coords_of_points[:, 0].argsort()]
        return coords_of_points

    def make_colors_of_clusters(self, default_cluster_number):
        colors_of_clusters = super().make_colors_of_clusters()
        red = "#FF0000"
        for i in range(self.__number_of_vertices):
            if colors_of_clusters[self.resulting_clustering[i]] != red and\
                    (self.resulting_clustering[i] == default_cluster_number or
                    np.count_nonzero(self.resulting_clustering == self.resulting_clustering[i]) <= 1):
                colors_of_clusters[self.resulting_clustering[i]] = red
        return colors_of_clusters

    def view_at_map(self, latitude, longitude, default_cluster_number=0):
        fmap = folium.Map([latitude, longitude])
        folium.TileLayer(
            tiles='https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_nolabels/{z}/{x}/{y}{r}.png',
            attr='My').add_to(fmap)
        folium.TileLayer(
            tiles='https://cartodb-basemaps-{s}.global.ssl.fastly.net/rastertiles/voyager_nolabels/{z}/{x}/{y}{r}.png',
            attr='My').add_to(fmap)
        folium.LayerControl().add_to(fmap)
        colors_of_clusters = self.make_colors_of_clusters(default_cluster_number)
        for i in range(self.__number_of_vertices):
            folium.CircleMarker([self.coord_of_points[i, 1], self.coord_of_points[i, 0]],
                                radius=2, fill=True, color=colors_of_clusters[self.resulting_clustering[i]],
                                popup=str(self.resulting_clustering[i])).add_to(fmap)
        fmap.save('test.html')


    def view(self, title='', figsize=(7, 7)):
        super().view(title, figsize)

    def __str__(self):
        return "filename: {0}\n".format(self.filename) + super.__str__()
