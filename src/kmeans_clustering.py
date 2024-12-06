import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

#Used to determine the optimal number of clusters needed for K-Means model to prevent over/under fitting data
def cluster_calculator(df):
    inertia = []
    for k in range(1, 750):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        inertia.append(kmeans.inertia_)

    """ 
    To Plot Data Points to find Curve 

    plt.plot(range(1, 500), inertia)
    plt.title("Elbow Method For Optimal k")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.show()
    """
    return find_elbow_point(inertia)

def find_elbow_point(inertia):
    """Find the elbow point in the inertia curve using the maximum distance method."""
    # Coordinates of all points
    x = np.arange(len(inertia))
    y = np.array(inertia)

    # Line between the first and last points
    line_start = np.array([x[0], y[0]])
    line_end = np.array([x[-1], y[-1]])

    # Compute distances from each point to the line
    distances = np.abs(np.cross(line_end - line_start, line_start - np.column_stack((x, y)))) / np.linalg.norm(line_end - line_start)

    # Find the index of the maximum distance (the elbow point)
    elbow_idx = np.argmax(distances)

    return elbow_idx + 1  # +1 because indices start at 0

#Fit the K-means model with the specified number of clusters
def fit_kmeans(X_scaled, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    return kmeans, clusters

if __name__ == "__main__":
    print('Kmeans_Clustering.py')