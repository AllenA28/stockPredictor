import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
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
feature_cols = [
    'volume_ratio', 'orb_high', 'orb_low', 'symbol_code',
    'ma_5', 'ma_10', 'vol_10', 'return_5'  
]
X_train = train_df[feature_cols]
Y_train = train_df['target']
X_test = test_df[feature_cols]
Y_test = test_df['target']

#train up the model and tes it
print("Training set class distribution:")
print(Y_train.value_counts(normalize=True))
print("Test set class distribution:")
print(Y_test.value_counts(normalize=True))


# Instantiate and train the XGBoost model
model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=(len(Y_train) - sum(Y_train)) / sum(Y_train),  # handle class imbalance
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train, Y_train)

# Evaluate
y_pred = model.predict(X_test)
print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))
print(f"Test Accuracy: {model.score(X_test, Y_test):.2f}")

joblib.dump(model, 'orb_model.pkl')

# Per-stock accuracy
for symbol in data['symbol'].unique():
    mask = test_df['symbol'] == symbol
    acc = model.score(X_test[mask], Y_test[mask])
    print(f"Test Accuracy for {symbol}: {acc:.2f}")