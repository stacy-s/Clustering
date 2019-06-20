"""
Cluster module.
Classes contain the coordinates of points, proper clustering, if known, the resulting clustering.
Clusters are being rendered.
"""
import numpy as np
import pandas as pd
import random
import tkinter
import folium
import matplotlib.pyplot as plt


class Cluster:
    """
    The class of the cluster object for the coordinates specified in the Cartesian system.
    """
    def __init__(self, coords_of_points: np.ndarray, right_clustering: np.ndarray, resulting_clustering=None):
        """
        Constructor
        :param coords_of_points: coordinates of points for clustering in a two-dimensional coordinate system.
            The function will be sorted using the function __sort
        :param right_clustering: proper splitting of points into clusters
        :param resulting_clustering: clustering obtained, default is None
        """
        if resulting_clustering is None:
            resulting_clustering = []
        self.coord_of_points, self.right_clustering = self.__sort(coords_of_points, right_clustering)
        self.resulting_clustering = resulting_clustering
        self.__number_of_vertices = self.coord_of_points.shape[0]

    def __sort(self, coords_of_points, right_clustering):
        """
        Sorts points (arrays coords_of_points, right_clustering) in order of increasing the first coordinate,
        with the equality of the first coordinate in the order of increasing the second coordinate of the point
        :param coords_of_points: coordinates of points for clustering in a two-dimensional coordinate system
        :param right_clustering: proper splitting of points into clusters
        :return: sorted arrays coords_of_points, right_clustering
        """
        index = coords_of_points[:, 0].argsort()
        coords_of_points = coords_of_points[index]
        if right_clustering is not None:
            right_clustering = right_clustering[index]
        return coords_of_points, right_clustering

    def _make_colors_of_clusters(self):
        """
        Definition of color for each cluster
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
        The function of drawing points on the coordinate line.
        Dots belonging to the same cluster are marked with the same color.
        :param title: caption to the chart
        :param figsize: graphic size
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
        """Getting string information about the object"""
        return "coord_of_points: {0}\n" \
               "right_clustering: {1}\n" \
               "resulting_clustering:{2}".format(self.coord_of_points, self.right_clustering, self.resulting_clustering)


class ClusterGreatCircles(Cluster):
    """
    Class object cluster for data on the surface of the ball.
    """
    def __init__(self, filepath, filename,  resulting_clustering=None):
        """
        Constructor
        :param filepath: path to the file with longitudes and latitudes of points on the Earth's surface.
        :param filename: filename with .csv expansion. File must contain columns
             longitude (dogot), latitude (latitude), owner (owner)
        :param resulting_clustering: clustering obtained, default is None
        """

        self.filepath = filepath
        self.filename = filename
        super().__init__(self.__load(), None, resulting_clustering)
        self.__number_of_vertices = self.coord_of_points.shape[0]

    def __load(self):
        """
        Loads data from the specified path (self.filepath + self.filename).
        It leaves only the first entry of the record with the same longitude, latitude, owner
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
        Definition of color for each cluster
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
        Saves a file with the .html extension in which points are drawn on the city map.
        Dots belonging to the same cluster are marked with the same color.
        Red color indicates points that are noises.
        :param latitude: city latitude
        :param longitude: city longitude
        :param filename_of_map: the name of the file received map (without extension)
        :param default_cluster_number: default cluster number
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
        """Getting string information about the object."""
        return "filename: {0}\n".format(self.filename) + super().__str__()
