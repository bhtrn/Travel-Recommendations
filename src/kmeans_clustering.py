import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# loading saved data if saved onto CSV file
def load_scaled_data(file_path):
    return pd.read_csv(file_path)

# used to determine the optimal number of clusters needed for K-Means model to prevent over/under fitting data
def elbow_method():
    return 0