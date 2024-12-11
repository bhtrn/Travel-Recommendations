import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

#Evaluate Clusters of Kmeans model
def evaluate_clustering(X_scaled, clusters):
    return silhouette_score(X_scaled, clusters)