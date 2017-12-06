import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)



def persistence_model(x):
    return x

FILE_PATH = 'data/bitcoin_price.csv'
RATIO_TRAIN = 0.9

if __name__ == '__main__':
    data = pd.read_csv(FILE_PATH)

    data['Date'] = pd.to_datetime(data['Date'], format='%b %d, %Y')
    close_price = data[['Date', 'Close']]
    close_price.index = close_price['Date']
    close_price_ts = close_price['Close']
    # chronological order
    close_price_ts = close_price_ts.sort_index()

    data1 = pd.concat([close_price_ts.shift(
        1), close_price_ts], axis=1).dropna()
    data1.columns = ['t-1', 't']
    X = data1['t-1'].values
    y = data1['t'].values

    n = len(X)
    n_train = int(RATIO_TRAIN * n)

    X_train, X_test = X[:n_train], X[n_train:n]
    y_train, y_test = y[:n_train], y[n_train:n]

    # predictions with persistence model
    predictions = []

    for x in X_test:
        pred = persistence_model(x)
        predictions.append(pred)

    # evaluation
    mae_baseline = mean_absolute_error(y_test, predictions)
    mse_baseline = mean_squared_error(y_test, predictions)
    r2_baseline = r2_score(y_test, predictions)

    print('MAE: %.9f' % mae_baseline)
    print('MSE: %.9f' % mse_baseline)
    print('R2: %.9f' % r2_baseline)

    plt.plot(y_test, label='real values')
    plt.plot(predictions, label='persistent model predictions')
    title = 'Persistent model predictions for the last %d values' % len(X_test)
    plt.title(title)
    plt.legend()
    plt.show()
