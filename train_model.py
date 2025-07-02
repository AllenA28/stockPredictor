import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import pdb

data = pd.read_parquet('filtered_data.parquet').reset_index()

#encode the stock symbols as numbers so the model can know
data['symbol_code'] = data['symbol'].astype('category').cat.codes

train_list, test_list = [], []

#seperate the first 80% of data for each stock as training and the rest for testing
for symbol in data["symbol"].unique():
    stock_df = data[data['symbol'] == symbol].sort_values('time')
    split_i = int(len(stock_df) * 0.8)
    train_list.append(stock_df.iloc[:split_i])
    test_list.append(stock_df.iloc[split_i:])

train_df = pd.concat(train_list)
test_df = pd.concat(test_list)

#set up the training and testing variables
X_train = train_df[['volume_ratio', 'orb_high', 'orb_low', 'symbol_code']]
Y_train = train_df['target']
X_test = test_df[['volume_ratio', 'orb_high', 'orb_low', 'symbol_code']]
Y_test = test_df['target']

#train up the model and tes it
model = GradientBoostingClassifier()
model.fit(X_train, Y_train)
joblib.dump(model, 'orb_model.pkl')
print(f"Test Accuracy: {model.score(X_test, Y_test):.2f}")

#per-stock test accuracy
for symbol in data['symbol'].unique():
    mask = test_df['symbol'] == symbol
    acc = model.score(X_test[mask], Y_test[mask])
    print(f"Test Accuracy for {symbol}: {acc:.2f}")