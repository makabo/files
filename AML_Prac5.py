# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

%matplotlib inline

# %%
#Displays plots directly inside the notebook.
#Density-Based Spatial Clustering
#Sample Dataset
X, y = make_blobs(n_samples=300,
centers=4,
cluster_std=0.60, #Controls cluster spread. Smaller value means tighterclusters.
random_state=42)

plt.scatter(X[:,0], X[:,1]) #Selects all rows of first column and second column
plt.title("Original Data")
plt.show()

# %%
#Standardize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#DBSCAN Clustering
dbscan = DBSCAN(eps=0.3, min_samples=5)
#Neighborhood radius. Any point within distance 0.3 is considered a neighbor.
#Minimum neighbors needed to form a cluster. If fewer than 5 neighbors exist: Point becomes noise.

db_labels = dbscan.fit_predict(X_scaled)

print("Cluster Labels:")
print(db_labels)

dbscan = DBSCAN(eps=0.3, min_samples=5)

db_labels = dbscan.fit_predict(X_scaled)

print("Cluster Labels:")
print(db_labels)

# %%
#0=cluster 1, 1=cluster 2, 2=cluster 3, -1=noise

#Visualize DBSCAN Clusters
plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:,0],
X_scaled[:,1],
c=db_labels,
cmap='rainbow')
#Merges clusters while minimizing variance. Most commonly used method.

plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


