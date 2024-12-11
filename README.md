Question: "Which categories are most popular among users with similar preferences?"

Objective:
Group users with similar preferences and identify the most popular categories for each user group.

DATASET INFORMATION:

Travel Review Ratings - https://archive.ics.uci.edu/dataset/485/tarvel+review+ratings

This data set is populated by capturing user ratings from Google reviews. Reviews on attractions from 24 categories across Europe are considered. Google user rating ranges from 1 to 5 and average user rating per category is calculated. 

Renjith, S. (2018). Travel Review Ratings [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5C31Q.

Tools/Libraries Used:
- ucimlrepo
- pandas
- sklearn
- matplotlib

1. Data Collection and Clean Up: (preprocessing.py)
    - extraction of data from UCI ML Repository
    - cleaning of data by ensuring data values are of type 'int' and 'float'
    - deletion of NaN values (deletion of 2 rows)
    - standardization of data values for all 24 categories of locations

2. K-means Clustering: (kmean_clustering.py)
    - elbow method used to find optimal cluster value for accurate model creation (Optimal Value: 77 Clusters)
    - kmeans model fitted with optimal cluster value
  
3. Cluster Analysis: (cluster_analysis.py)
   - analysis of cluster groupings and assignment of cluster to each data entry (user)
  
4. User Recommendation Trial: (User Recommendations)
   - sample user created for model testing: user_ratings = np.array([4, 3, 5, 3, 2, 4, 5, 1, 4, 3, 5, 4, 4, 3, 2, 1, 5, 4, 2, 5, 3, 2, 4, 3])
   - sample user recommendations returned:
    Cluster 2:
    malls                 1.110795
    art galleries         0.851469
    juice bars            0.785235
    local services        0.480208
    burger/pizza shops    0.428190

    Results: Sample user is suggested to visit the following types of places in Europe based on individual scoring of 24 different categoreis of locations

