import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import processing as proc
import utils


def fit_lstm(train, batch_size, nb_epochs, nb_neurons, optimiser):
    X, y = train[:, 0:-1], train[:, -1]

    # reshape training data according to "samples", "time steps" and "features"
    X = X.reshape(X.shape[0], 1, X.shape[1])

    model = Sequential()
    model.add(LSTM(nb_neurons, batch_input_shape=(
        batch_size, X.shape[1], X.shape[2]), stateful=True))
    # model.add(LSTM(10, batch_input_shape=(
    #     batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer=optimiser)

    hist_train_loss = []
    hist_val_loss = []

    for i in range(nb_epochs):
        hist = model.fit(X, y, epochs=1, batch_size=batch_size,
                         verbose=1, shuffle=False, validation_split=0.33)
        hist_train_loss += hist.history['loss']
        hist_val_loss += hist.history['val_loss']
        model.reset_states()
    return model, hist_train_loss, hist_val_loss


def forecast_lstm(model, batch_size, X):
    # X will be one row of independent variables
    X = X.reshape(1, 1, len(X))
    y_hat = model.predict(X, batch_size=batch_size)
    return y_hat[0, 0]

FILEPATH = 'data/bitcoin_price.csv'

STATIONARITY_INTERVAL = 1
NB_LAGS_AR = 1
RATIO_TRAINING_SET = 0.9

# parameters of the neural network
NB_NEURONS = 6
BATCH_SIZE_TEST = 1
NB_EPOCHS = 50
if __name__ == '__main__':
    data = pd.read_csv(FILEPATH)

    data['Date'] = pd.to_datetime(data['Date'], format='%b %d, %Y')
    close_price = data[['Date', 'Close']]
    close_price.index = close_price['Date']
    close_price_ts = close_price['Close']

    # chronological order
    close_price_ts = close_price_ts.sort_index()
    close_price_ts.head()

    # transform signal
    stationnary = proc.difference(close_price_ts, STATIONARITY_INTERVAL)
    supervised = proc.to_supervised(stationnary, nb_lags=NB_LAGS_AR)

    supervised_val = supervised.values

    n = len(supervised_val)
    n_train = int(RATIO_TRAINING_SET * n)

    train, test = supervised_val[:n_train], supervised_val[n_train:]

    # normalize the dataset (inverse_transform to inverse the normalistion)
    train, test, scaler = proc.scale(train, test)

    possible_batch_size_train = utils.prime_factors(len(train))
    batch_size_train = utils.user_select_train_batch_size(
        possible_batch_size_train)

    optim = Adam(lr=0.01, beta_1=0.999, beta_2=0.999)
    model, hist_train_loss, hist_val_loss = fit_lstm(train, batch_size_train,
                                                     NB_EPOCHS, NB_NEURONS, optim)

    # forecast on the entire training dataset to build up state for forecasting
    # X_train_reshaped = train[
    #     :, 0:-1].reshape(train.shape[0], 1, train.shape[1] - 1)
    # model.predict(X_train_reshaped, batch_size=batch_size_train)

    new_model = Sequential()
    new_model.add(LSTM(NB_NEURONS, batch_input_shape=(
        BATCH_SIZE_TEST, 1, train[:, 0:-1].shape[1]), stateful=True))
    # model.add(LSTM(10, batch_input_shape=(
    #     batch_size, X.shape[1], X.shape[2]), stateful=True))
    new_model.add(Dense(1))
    old_weights = model.get_weights()
    new_model.set_weights(old_weights)
    new_model.compile(loss='mean_squared_error', optimizer=optim)

    predictions = []

    for i in range(len(test)):
        X, y = test[i, 0:-1], test[i, -1]
        y_hat = forecast_lstm(new_model, BATCH_SIZE_TEST, X)
        y_hat = proc.invert_scale(scaler, X, y_hat)
        y_hat = proc.inverse_difference(
            close_price_ts, y_hat, len(test) + 1 - i)
        predictions.append(y_hat)
        expected = close_price_ts[len(train) + i + 1]
        #print('Day=%d, Predicted=%f, Expected=%f' % (i+1, y_hat, expected))

    # report performance
    mse = mean_squared_error(close_price_ts.values[
                             n_train + NB_LAGS_AR + STATIONARITY_INTERVAL:], predictions)
    mae = mean_absolute_error(close_price_ts.values[
                              n_train + NB_LAGS_AR + STATIONARITY_INTERVAL:], predictions)
    r2 = r2_score(close_price_ts.values[
                  n_train + NB_LAGS_AR + STATIONARITY_INTERVAL:], predictions)

    print('Test MAE: %.3f' % mae)
    print('Test MSE: %.3f' % mse)
    print('Test R2: %.3f' % r2)

    # line plot of observed vs predicted
    plt.plot(close_price_ts.values[
             n_train + NB_LAGS_AR + STATIONARITY_INTERVAL:],
             label='real values')
    plt.plot(predictions,
             label='lstm model predictions')
    plt.title('LSTM model predictions for the last %d values' % len(test))
    plt.legend()

    epochs = range(1, len(hist_val_loss) + 1)

    fig1, ax1 = plt.subplots()
    ax1.plot(epochs, hist_train_loss)
    ax1.plot(epochs, hist_val_loss)
    ax1.legend(['train_loss', 'validation_loss'], loc='upper left')

    plt.show()
