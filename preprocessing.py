from ucimlrepo import fetch_ucirepo 
import pandas as pd

def load_data():
    # fetch dataset and return data as a Pandas DataFrame
    travel_review_ratings = fetch_ucirepo(id=485)
    return travel_review_ratings.data

def load_data_csv(data):
    return

def clean_data(df):
    #Ensure Data types are cleaned
    check_data_types(df)
    return 0

def standardize_data():
    return 0

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

    def preprocessing():
        df = load_data()
        clean_data(df)

if __name__ == '__main__':
    clean_data(load_data())
    print("Success")