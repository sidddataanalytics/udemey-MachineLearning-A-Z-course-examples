# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 09:28:24 2020

@author: 320001866
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the data
dataset = pd.read_csv('Mall_Customers.csv')

x = dataset.iloc[:,[3,4]].values

from sklearn.cluster import KMeans

wcss = []

for i in range (1,11):
    kmeans = KMeans(n_clusters= i, init = 'k-means++', max_iter = 300, n_init = 10, random_state =0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.tittle('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss - square of cluster distances')
plt.show

#Applying Kmens to mall data set

kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)

#visulize the plot
plt.scatter(x[y_kmeans ==0,0], x[y_kmeans ==0,1], s= 100, c= 'red', Label = 'Carefull')
plt.scatter(x[y_kmeans ==1,0], x[y_kmeans ==1,1], s= 100, c= 'blue', Label = 'standard')
plt.scatter(x[y_kmeans ==2,0], x[y_kmeans ==2,1], s= 100, c= 'green', Label = 'Target')
plt.scatter(x[y_kmeans ==3,0], x[y_kmeans ==3,1], s= 100, c= 'cyan', Label = 'carefull')
plt.scatter(x[y_kmeans ==4,0], x[y_kmeans ==4,1], s= 100, c= 'magenta', Label = 'sensible')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s= 300, c= 'yellow', Label = 'centroids')
plt.title('cluster')
plt.xlabel('Annual income')
plt.ylabel('spending score (1-100)')
plt.legend()
plt.show()




    

    



