import Cluster
import Clustering
import Metrics
import numpy as np
from sklearn.datasets import make_moons, make_blobs, make_circles


def static_experiment(metric, count_experiments, k_range, eps_range, function_get_dataset):
    """
    The function of conducting experiments for a given range of k and eps.
    :param metric: clustering quality assessment metric
    :param count_experiments: number of experiments
    :param k_range: array of k parameter values.
    :param eps_range: array of eps parameter values.
    :param function_get_dataset: function to get a dataset
    :return: experimental results
    """
    size = (count_experiments * len(k_range) * len(eps_range), 2)
    experimental_results = np.zeros(shape=size, dtype=float)
    row = 0
    for i in range(count_experiments):
        print(i)
        points, correct_clustering = function_get_dataset()
        for k in k_range:
            for eps in eps_range:
                d = Cluster.Cluster(points, correct_clustering)
                mxt = Clustering.K_MXT(k=k, eps=eps, cluster=d)
                mxt()
                score_mxt = metric(mxt)
                gauss = Clustering.K_MXTGauss(k=k, eps=eps, cluster=d)
                gauss()
                score_gauss = metric(gauss)
                experimental_results[row][0], experimental_results[row][1] = score_mxt(), score_gauss()
                row += 1
    return experimental_results


def save_results(func_name, experimental_results):
    """
    The function saves the results.
    :param func_name: name of function
    :param experimental_results: results
    :return: None
    """
    np.savetxt(func_name, experimental_results, header='k-MXT   k_MXT-Gauss', fmt='%.2f %.2f')

