import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#Standardization of new user data before recommendations can be done
def standardize_new_user_data(user_data):
    scaler = StandardScaler()
    return scaler.fit_transform(user_data.reshape(1, -1))

#Predict the outcome of the new user data 
def predict_user_cluster(user_scaled, kmeans):
    return kmeans.predict(user_scaled)

#Recommends top 5 categories for new user
def recommend_categories(user_cluster, cluster_means):
    recommended_categories = cluster_means.loc[user_cluster].sort_values(ascending=False).head(5)
    return recommended_categories
