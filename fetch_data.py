from dotenv import load_dotenv
load_dotenv()

from datetime import timedelta
import alpaca_trade_api as tradeapi
import pandas as pd
import os
import pdb

#load API keys
api = tradeapi.REST(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_API_SECRET"),
    base_url = "https://paper-api.alpaca.markets"
)

TARGET_STOCKS = ["NVDA", "MSFT"]

#get the data to train my ML model
def fetch_historical_data(symbol, years=3):
    '''
    here our goal is to grab the past three years worth of data from the alpaca 
    api (as recent as the past 16 minutes due to subscription problems)
    '''
    end_date = pd.Timestamp.now(tz="America/Chicago") - timedelta(minutes=16)
    start_date = end_date - pd.DateOffset(years=1)
    bars = api.get_bars(symbol, "1Min", start=start_date.isoformat(), end=end_date.isoformat()).df
    bars["symbol"] = symbol

    return bars[["open", "high", "low", "close", "volume", "symbol"]]

all_data = pd.concat([fetch_historical_data(stock) for stock in TARGET_STOCKS])
pdb.set_trace()
all_data.to_parquet('stock_data.parquet')