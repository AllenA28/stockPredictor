import os
import pandas as pd
import pytz
from datetime import datetime, time as dtime
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import joblib
import time
import pdb

# load environment variables and model
load_dotenv()
model = joblib.load('orb_model.pkl')

# set up Alpaca API
api = tradeapi.REST(
    os.getenv('ALPACA_API_KEY'),
    os.getenv('ALPACA_API_SECRET'),
    base_url = "https://paper-api.alpaca.markets"
)

data = pd.read_parquet("filtered_data.parquet")

TARGET_STOCKS = ["MSFT", "NVDA"]
symbol_codes = {s: i for i, s in enumerate(TARGET_STOCKS)}

# set a baseline for adding the data for each symbol
symbol_state = {
    symbol: {
        'bars': [],
        'orb_high': None,
        'orb_low': None,
        'volume_avg': None,
        'opening_range_complete': False
    } for symbol in TARGET_STOCKS
}

#set time zone
central = pytx.timezone('America/Chicago')

def process_bar(symbol, bar):
    '''
    examines each bar for potential breakouts
    '''
    # a bar is a dict with keys ['t', 'o', 'h', 'l', 'c', 'v']
    state = symbol_state[symbol]
    bar_time = pd.Timestamp(bar['t']).tz_convert(central)
    bar_minute = bar_time.time()
    state['bars'].append(bar)

    #just keep collecting if in opening range (8:30 - 9 AM)
    if dtime(8, 30) <= bar_minute < dtime(9,0):
        return # don't do anything
           
    # if it's 9 we need to calculate the features
    elif not state['opening_range_complete'] and bar_minute >= dtime (9,0):
        opening_bars = [b for b in state['bars'] if dtime(8,30) <= pd.Timesstamp(bar['t']).tz_convert(central).time() < dtime(9,0)]
        if len(opening_bars) < 30:
            print(f"Not enough bars for {symbol} opening range yet.")
            return
        state['orb_high'] = max(b['h'] for b in opening_bars)
        state['orb_low'] = min(b['l'] for b in opening_bars)
        state['volume_avg'] = sum(b['v'] for b in opening_bars) / 30
        state['opening_range_complete'] = True
        print(f"{symbol} ORB set: High={state['orb_high']:.2f}, Low={state['orb_low']:.2f}, VolAvg={state['volume_avg']:.0f}")
        return
    
    # after the ORB is set, check for breakout and print signal
    if state['opening_range_complete']:
        last_bar = bar
        if last_bar['c'] > state['orb_high']:
            # create features to feed to model
            features = pd.DataFrame([{
                'volume_ratio': last_bar['v'] / state['volume_avg'],
                'orb_high': state['orb_high'],
                'orb_low': state['orb_low'],
                'symbol_code': symbol_codes[symbol]
            }])
            probability = model.predict_proba(features)[0][1]
            if probability > 0.7:
                print(f"BUY {symbol} | Time: {bar_time.strftime('%Y-%m-%d %H:%M:%S')} | Price: {last_bar['c']:.2f} | Prob: {probability:.2%}")

def get_latest(symbol):
    '''
    gets the latest 1 min bar from the alpaca API
    '''
    bars = api.get_bars(symbol, "1Min", limit=1).df
    if not bars.empty:
        bar = bars.iloc[-1]
        return {
            't': bar.name,  # pandas.Timestamp
            'o': bar['open'],
            'h': bar['high'],
            'l': bar['low'],
            'c': bar['close'],
            'v': bar['volume']
        }
    return None

##################### Now for the main loop to make this thing work ##########################

if __name__ == "__main__":
    print("Starting live ORB signal engine...")
    while True:
        now = datetime.now(central)
        if now.time() < dtime(8, 30) or now.time() > dtime(15, 0):
            time.sleep(30)
            continue  # Only run during market hours

        for symbol in TARGET_STOCKS:
            bar = get_latest(symbol)
            if bar:
                process_bar(symbol, bar)
        time.sleep(60)  # Wait for next minute bar
     