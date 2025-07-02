import pandas as pd

featured_data = pd.read_parquet('featured_data.parquet')

def filter_after_opening(data):
    '''
    takes in a df of stock data with symbols and dates that has 
    data for every minute. Then takes the first 30 min out of 
    each day
    '''
    return data.groupby(['symbol', 'date'], group_keys=False).apply(lambda group: group.iloc[30:])

filtered_data = filter_after_opening(featured_data)
filtered_data.to_parquet('filtered_data.parquet')