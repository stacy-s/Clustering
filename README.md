# Clustering

The datasets folder contains the datasets obtained from the Flickr site and the results of experiments for different data sets.


The geoflickr_spb.csv file contains all the data received.
The geoflickr_spb_drop_duplicates.csv file contains data that contained no more than one record with the same latitude, longitude, and owner name from the geoflickr_spb.csv file.


The kmxt0 file contains the experiment data of the k-MXT algorithm on a blob data set with cluster_std = 0.5.
The kmxt1 file contains the experiment data of the k-MXT algorithm on the blob data set with cluster_std = [1.0, 1.5, 0.5].
The kmxt2 file contains the k-MXT experiment data on a circles data set.
The kmxt3 file contains the experiment data of the k-MXT algorithm on a moons data set.

The kmxt-w0 file contains the experiment data of the k-MXT-W algorithm on a blob data set with cluster_std = 0.5.
The kmxt-w1 file contains the experiment data of the k-MXT-W algorithm on the blob data set with cluster_std = [1.0, 1.5, 0.5].
The kmxt-w2 file contains the k-MXT-W experiment data on a circles data set.
The kmxt-w3 file contains the experiment data of the k-MXT-W algorithm on a moons data set.
