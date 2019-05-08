import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle

data=pd.read_csv("table_417.csv", delimiter=',', header=None)
x=data[1].values
y=data[4].values
XX=np.array(list(zip(x,y)))

# #############################################################################
# Compute clustering with MeanShift

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(XX, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(XX)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

# #############################################################################
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(XX[my_members, 0], XX[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

plt.xlabel('Absolute magnitude')
plt.ylabel('Median age [Myr]')
plt.title('(108138) 2011 GB11')
plt.text(16.82,1.72,'(a)', fontsize=12)
plt.savefig('/home/valerio/PRAVEC/TEXT/ML_417/MEAN-SHIFT_417/Mean-shift_417.eps')
plt.show()
