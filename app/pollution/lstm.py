import warnings
warnings.filterwarnings('ignore')


import logging
import os
from datetime import datetime
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from app.config import config
import pandas as pd

pd.set_option('display.max_columns', None)
from app.time_series.prep import series_to_supervised

pollution_path = os.path.join(config.DATA_FOLDER_PATH, 'pollution', 'pollution.csv')
scaler = MinMaxScaler(feature_range=(0, 1))


def transform_raw():
    raw_path = os.path.join(config.DATA_FOLDER_PATH, 'pollution', 'raw.csv')
    dataset = read_csv(raw_path, parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
    dataset.drop('No', axis=1, inplace=True)
    # manually specify column names
    dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    dataset.index.name = 'date'
    # mark all NA values with 0
    dataset['pollution'].fillna(0, inplace=True)
    # drop the first 24 hours
    dataset = dataset[24:]
    # summarize first 5 rows
    print(dataset.head(5))
    # save to file
    dataset.to_csv(pollution_path)


def show_pollution():
    # load dataset
    dataset = read_csv('pollution.csv', header=0, index_col=0)
    values = dataset.values
    # specify columns to plot
    groups = [0, 1, 2, 3, 5, 6, 7]
    i = 1
    # plot each column
    pyplot.figure()
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(values[:, group])
        pyplot.title(dataset.columns[group], y=0.5, loc='right')
        i += 1
    pyplot.show()


# load data
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')


def prep_pollution_data():
    # load dataset
    dataset = read_csv(pollution_path, header=0, index_col=0)

    values = dataset.values
    # integer encode direction
    encoder = LabelEncoder()
    values[:, 4] = encoder.fit_transform(values[:, 4])
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
    print(reframed.head())

    return reframed, scaler


def split_train_test(df):
    # split into train and test sets
    values = df.values
    n_train_hours = 365 * 24
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    logging.info(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    return train_X, train_y, test_X, test_y


def main():
    epochs = 50
    df, scaler = prep_pollution_data()

    train_X, train_y, test_X, test_y = split_train_test(df)

    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))

    print('Test RMSE: %.3f' % rmse)

    # logging.info(inv_y)
    # return

