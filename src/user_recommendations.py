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

# Convert user_ratings to a DataFrame with the feature names
def convert_to_df(user_ratings):
    feature_names = [
    "churches", "resorts", "beaches", "parks", "theatres", "museums", "malls",
    "zoos", "restaurants", "pubs/bars", "local services", "burger/pizza shops",
    "hotels/other lodgings", "juice bars", "art galleries", "dance clubs",
    "swimming pools", "gyms", "bakeries", "beauty & spas", "cafes", 
    "view points", "monuments", "gardens"]

    # Convert user_ratings to a DataFrame with the feature names
    return pd.DataFrame([user_ratings], columns=feature_names)

