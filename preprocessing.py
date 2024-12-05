from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data():
    # fetch dataset and return data as a Pandas DataFrame
    travel_review_ratings = fetch_ucirepo(id=485)
    return travel_review_ratings.data

#if grabbing data from CSV file
def load_data_csv(data):
    travel_review_ratings = pd.read_csv("Your_file_path")
    return travel_review_ratings

def clean_data(df):
    #Ensure Data types are cleaned
    check_data_types(df)
    return 0

def standardize_data(df):
    # Step 1: Extract features (assuming df['features'] is a DataFrame)
    features_df = df['features']
    
    # Step 2: Apply StandardScaler to all numeric columns in features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)

    # Step 3: Convert the result back to a DataFrame
    features_df = pd.DataFrame(features_scaled, columns=features_df.columns)

    df['features'] = features_df

    # (Optional) Check Means and Standard Deviations
    #print("Means:\n", features_scaled_df.mean())
    #print("Standard Deviations:\n", features_scaled_df.std())
    

def check_data_types(df):
    #Ensure userid is the correct data type for all data points
    df['ids'] = df['ids'].astype('category')

    #Setting which data types will be integers or continous
    continuous_cols = ['churches', 'resorts', 'parks', 'theatres', 'museums', 'malls', 'zoos',
                    'pubs/bars', 'local services', 'burger/pizza shops', 'hotels/other lodgings', 'juice bars', 'dance clubs', 'swimming pools',
                    'gyms', 'bakeries', 'beauty & spas', 'cafes', 'view points', 'monuments', 'gardens']

    integer_cols = ['beaches', 'restaurants', 'art galleries']
    #Remove unwanted characters from dataframe tgat aren't numerical values
    df['features'] = df['features'].replace({r'\t': ''}, regex=True)

    #Clean Data and make sure that the values are all floats or ints
    for col in continuous_cols:
        df['features'][col] = df['features'][col].astype(float)

    for col in integer_cols:
        df['features'][col] = df['features'][col].astype(int)

    #Uncomment to do data type check of 'features' column
    #print(df['features'].dtypes)

def preprocessing():
    df = load_data()
    clean_data(df)
    standardize_data(df)
    return df

if __name__ == '__main__':
    preprocessing()
    print("Success")