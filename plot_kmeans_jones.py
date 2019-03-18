import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

#kmeans = KMeans(n_clusters = 3, init = 'random')

data=pd.read_csv("table_jones.csv", delimiter=',', header=None)
x=data[1].values
y=data[4].values
XX=np.array(list(zip(x,y)))
#print(XX)
#print(x,y)
kmeans = KMeans(n_clusters=4)
# Fitting with inputs
kmeans = kmeans.fit(XX)
# Predicting the clusters
labels = kmeans.predict(XX)
# Getting the cluster centers
C = kmeans.cluster_centers_
y_pred=KMeans(n_clusters=4).fit_predict(XX)
#print(y_pred)
plt.scatter(C[:,0],C[:,1], marker='*', c='red', s=1000)
print(C[:,1])
#plt.plot(x,y,'k.')
plt.scatter(XX[:,0],XX[:,1],c=y_pred)
#plt.xlabel('Ejection Velocity [m/s]')
plt.xlabel('Absolute magnitude',fontsize=18)
plt.ylabel('Median age [Myr]',fontsize=18)
plt.title('(3152) Jones',fontsize=18)
plt.text(15.2,4.65,'(b)', fontsize=12)
plt.savefig('/home/valerio/PRAVEC/TEXT/ML_JONES/KMEANS_JONES/KMeans_jones.eps')
plt.show()
