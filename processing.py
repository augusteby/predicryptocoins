import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# function that transform the entire signal into a stationnary one through
# differencing
def difference(dataset, interval=1):
    diff = []
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)

    return pd.Series(diff)

# function that gives the real value from the prediction based on the
# stationnary signal


def inverse_difference(history, pred, interval=1):
    return pred + history[-interval]


def to_supervised(data, nb_lags=1):

    df = pd.DataFrame(data)

    # add independent variables
    columns = [df.shift(i) for i in range(1, nb_lags + 1)]
    headers = ['t-%d' % i for i in range(1, nb_lags + 1)]

    # add dependent variable
    columns.append(df)
    headers.append('t')

    supervised_data = pd.concat(columns, axis=1).dropna()
    supervised_data.columns = headers

    return supervised_data


# scale train and test data to [-1,1]
def scale(train, test):
    # fit on train
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train)

    # transform train
    train_scaled = scaler.transform(train)

    # transform test
    test_scaled = scaler.transform(test)

    return train_scaled, test_scaled, scaler


def invert_scale(scaler, X, value):

    new_row = [x for x in X] + [value]

    inverted = scaler.inverse_transform([new_row])
    return inverted[0, -1]
