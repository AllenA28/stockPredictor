# Stock Predictor: ML-Enhanced ORB Strategy

A machine learning-enhanced Opening Range Breakout (ORB) trading bot that analyzes the first 30 minutes of stock trading to predict day trading opportunities.

## Features

- **Opening Range Analysis:** Monitors the first 30 minutes of trading for breakout patterns.
- **Machine Learning Integration:** Uses GradientBoostingClassifier to predict breakout validity and filter out false breakouts.
- **Real-Time Data:** Connects to market data APIs for live analysis.
- **Risk Management:** Implements stop-loss and profit-target logic.
- **Paper Trading Support:** Compatible with Alpaca's paper trading API.
