import Cluster
import Clustering
import Metrics
import numpy as np
from sklearn.datasets import make_moons, make_blobs, make_circles
import time

def static_experiment(metric, count_experiments, k_range, eps_range, function_get_dataset):
    size = (count_experiments * len(k_range) * len(eps_range), 4)
    experimental_results = np.zeros(shape=size, dtype=float)
    row = 0
    for i in range(count_experiments):
        print(i)
        points, correct_clustering = function_get_dataset()
        for k in k_range:
            # start = time.time()
            for eps in eps_range:
                d = Cluster.Cluster(points, correct_clustering)
                mxt = Clustering.K_MXT(k=k, eps=eps, cluster=d)
                mxt()
                score_mxt = metric(mxt)
                gauss = Clustering.K_MXTGauss(k=k, eps=eps, cluster=d)
                gauss()
                score_gauss = metric(gauss)
                experimental_results[row][0], experimental_results[row][1] = k, eps
                experimental_results[row][2], experimental_results[row][3] = score_mxt(), score_gauss()
                row += 1
            # end = time.time()
            # print(end - start)
    return experimental_results


def save_results(func_name, experimental_results):
    np.savetxt(func_name, experimental_results, header='k   eps   k-MXT   k_MXT-Gauss', fmt='%d %.2f %.2f %.2f')

