import pandas as pd

#Analyze the clusters and get the average ratings per category.
def analyze_clusters(X_clustered):
    return X_clustered.groupby('Cluster').mean()

if __name__ == "__main__":
    print("Cluster Analysis")