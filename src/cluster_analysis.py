import pandas as pd

#Load the data with cluster labels
def load_clustered_data(file_path):
    return pd.read_csv(file_path)

#Analyze the clusters and get the average ratings per category.
def analyze_clusters(X_clustered):
    return X_clustered.groupby('Cluster').mean()

if __name__ == "__main__":
    print("Cluster Analysis")