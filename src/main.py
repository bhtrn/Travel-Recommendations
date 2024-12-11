# Import necessary libraries
import numpy as np
import pandas as pd
import preprocessing as pp
import kmeans_clustering as kc
import cluster_analysis as ca
import user_recommendations as ur

#Saving data to CSV
def save_to_CSV(df_data, file_path):
    df_data.to_csv(file_path, index=True)

#Load the data from
def load_data_CSV(file_path):
    return pd.read_csv(file_path)


if __name__ == '__main__':
    print("Running Travel_Recommendation ML Model")

    """Data Collection and Clean Up"""
    df = pp.preprocessing()
    print("Step 1: Processing Complete")

    elbow_val = 77

    """
    Experimental Clustering and Plot
    elbow_val = kc.cluster_calculator(df['features'])
    """

    """K-means Clustering"""
    kmeans, clusters = kc.fit_kmeans(df['features'], elbow_val)
    
    df['features']['Cluster'] = clusters

    #Average rating for each cluster stored here
    cluster_analysis = ca.analyze_clusters(df['features'])

    print("Step 2: Kmeans Clustering Complete")

    """User Recommendation Testing"""
    #Sample User
    user_ratings = np.array([4, 3, 5, 3, 2, 4, 5, 1, 4, 3, 5, 4, 4, 3, 2, 1, 5, 4, 2, 5, 3, 2, 4, 3])
    standardized_ratings = ur.standardize_new_user_data(user_ratings)

    #Addition of Features:
    standardized_ratings = ur.convert_to_df(standardized_ratings.flatten())

    #Prediction Analysis
    prediction = ur.predict_user_cluster(standardized_ratings, kmeans)
    print(prediction)

    #Making a recommendation based on user choice
    recommendations = ur.recommend_categories(prediction[0], cluster_analysis)
    print(recommendations)

    print('Success')
    
