"""
Drawing graphs based on clustering results.
"""
import Cluster
import Clustering
import Metrics
import numpy as np
import matplotlib.pyplot as plt


def make_plot(points: np.ndarray, title, name_validation_metric, xticks: np.ndarray, yticks: np.ndarray, figsize=(7,7)):
    """
    The function of plotting the metric dependence on the value of the eps parameter.
    :param points: The points of the graph of dependence of the metric on the values of the parameter eps.
           (in the form of a numpy array)
    :param title: title of the plot
    :param name_validation_metric: name of the metric
    :param xticks: the numpy array of points at which the grid will be drawn along the x coordinate
    :param yticks: the numpy array of points at which the grid will be drawn along the y coordinate
    :param figsize: size of a figure
    :return: None
    """
    fig = plt.figure(figsize=figsize)
    plt.subplot(111)
    plt.ylabel(name_validation_metric)
    plt.plot(points[:, 0], points[:, 1], 'o', color='blue', linestyle='solid',
             markersize=4)
    plt.xlabel('eps')
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xlim([np.min(xticks), np.max(xticks) + 0.1])
    plt.ylim([np.min(yticks), np.max(yticks) + 0.02])
    plt.title(title)
    plt.rc('grid', linestyle="-", color='black')
    plt.grid(True)
    plt.show()


def find_position_of_max_value_of_metric(value_of_metric):
    """
    The function returns the position number with the maximum metric value.
    :param value_of_metric: array  values of metrics
    :return: position of maximum metric value
    """
    mx = max(value_of_metric)
    return [x for x in range(len(value_of_metric)) if value_of_metric[x] == mx][0]


def build_point(value_of_metric, objects, name_of_metric):
    """
    The function draws the splitting of vertices into clusters.
    :param value_of_metric: value of metric
    :param objects: class object clustering algorithm
    :param name_of_metric: name of clustering quality metric
    :return: None
    """
    objects.cluster.view(title=str(objects) + " {0} = {1:.2f}".format(name_of_metric, value_of_metric))