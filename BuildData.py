"""
Построение графиков на основе результатов кластеризации
"""
import Cluster
import Clustering
import Metrics
import numpy as np
import matplotlib.pyplot as plt


def make_plot(points: np.ndarray, title, name_validation_metric, xticks: np.ndarray, yticks: np.ndarray, figsize=(7, 7)):
    """
    Функция построения графика зависимости метрики от значений параметра eps.
    :param points: Точки графика зависимости метрики от значений параметра eps. (в виде numpy массива)
    :param title: Подпись к графику
    :param name_validation_metric: название метрики
    :param xticks: numpy массив точек, в которых будет проведена сетка по координате x
    :param yticks: numpy массив точек, в которых будет проведена сетка по координате y
    :param figsize: размер фигуры
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
    Возвращает номер позиции с максимальным значением метрики
    :param value_of_metric: Массив метрик
    :return: позиция максимального значения метрики
    """
    mx = max(value_of_metric)
    return [x for x in range(len(value_of_metric)) if value_of_metric[x] == mx][0]


def build_point(value_of_metric, objects, name_of_metric):
    """
    Построение разбиения точек на кластеры
    :param value_of_metric: значение метрики
    :param objects: объект класса алгоритма кластеризации
    :param name_of_metric: название метрики оценки качества кластеризации
    :return: None
    """
    objects.cluster.view(title=str(objects) + " {0} = {1:.2f}".format(name_of_metric, value_of_metric))