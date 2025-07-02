import pandas as pd
import pdb
data = pd.read_parquet('stock_data.parquet')
data.index.name = 'time'
data = data.set_index(['symbol', pd.to_datetime(data.index)])

def get_orb(group):
    '''
    Calculated the ORB features for the group given. Groups should be of same day and symbol
    '''
    first_30 = group.head(30)
    group['orb_high'] = first_30['high'].max()
    group['orb_low'] = first_30['low'].min()
    group['volume_avg'] = first_30['volume'].mean()
    return group

def calculate_features(stock_data):
    '''
    here we must calculate the different features for our actual orb strategy
    '''
    # Create date column from index
    stock_data = stock_data.reset_index('time')
    stock_data['date'] = pd.to_datetime(stock_data['time'].dt.date)
    stock_data = stock_data.set_index('time', append=True)


    # calculate the ORB features and volume ratio
    stock_data = stock_data.groupby(['symbol', 'date'], group_keys=False).apply(get_orb)
    stock_data['volume_ratio'] = stock_data['volume'] / stock_data['volume_avg']


    # check if the stock price reach the target at any point in the next 2 hours?
    # need to make a mini function for this check so I can .apply for the two groups
    def check_target(group):
        group['max_future'] = group['close'][::-1].rolling(120, min_periods=1).max()[::-1]
        group['target'] = (group['max_future'] >= group['close'] * 1.005).astype(int)
        return group
    
    stock_data = stock_data.groupby('symbol', group_keys=False).apply(check_target)
    return stock_data.drop(columns='max_future')
# Execute the functions
featured_data = calculate_features(data)
featured_data.to_parquet('featured_data.parquet')
pdb.set_trace()