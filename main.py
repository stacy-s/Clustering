import Cluster
import Clustering
import Graph
import numpy as np
from sklearn.datasets import make_moons, make_blobs, make_biclusters, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import adjusted_rand_score

# d = Cluster.Cluster(np.array([[1, 1], [2, 2], [3, 3], [4, 4]]), np.zeros(4))
# c1 = Clustering.DBSCAN(10, 0,  d)
# c1()
# print(c1.distance([0, 0], [90, 90]))
# print(np.concatenate((np.array([]), np.array([]))))
# c2 = Clustering.DBSCANGreatCircle(10, 0,  d)
# c2()
# print(c2.distance([0, 0], [90, 90]))
# c3 = Clustering.K_MXT(10, 2,  d)
# c3()
#
#
# g = Graph.StronglyConnectedComponent(3, [[1], [0], [0]])
# g()
# print(g.numbers_of_connected_component)

points, correct_clustering = make_blobs(n_samples=200,
                                               random_state=0,
                                               cluster_std=0.5)
filepath = "~/documents/институт рисков/Кластеризация/clustering_geotagged_data/"
filename = "geoflickr_spt.csv"
d = Cluster.ClusterGreatCircles(filepath, filename)

c = Clustering.K_MXTGaussGreatCircle(50, 7,  d)
c()
print(*c.cluster.resulting_clustering)
# print(*c.cluster.resulting_clustering)
# print(adjusted_rand_score(c.cluster.right_clustering, c.cluster.resulting_clustering))
# c.cluster.view_at_map(59.93863, 30.31413)
c.cluster.view()