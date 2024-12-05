# Import necessary libraries
import numpy as np
import pandas as pd
import preprocessing as pp
import kmeans_clustering as kc

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

    print(kmeans)
    print(len(clusters))
    print('Success')
    
