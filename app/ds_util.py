# https://deeplearningcourses.com/c/data-science-supervised-machine-learning-in-python
# https://www.udemy.com/data-science-supervised-machine-learning-in-python
from __future__ import print_function, division

import logging
import os
from builtins import range
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
# Note: you may need to update your version of future
# sudo pip install -U future
from sklearn.preprocessing import LabelEncoder

from app.config import config


def get_mnist(limit=None, otherize_digits=[]):
    print("Reading in and transforming data...")
    df = pd.read_csv(os.path.join(config.DATA_FOLDER_PATH, 'train.csv'))

    df = otherize_classes(df, otherize_digits)

    data = df.values

    np.random.shuffle(data)

    X, Y = scale_split(data)

    X, Y = impose_limit(X, Y, limit)

    X = X.astype('float32')

    return X, Y


def parse(x):
    return datetime.strptime(x, '%Y-%m-%d')


def get_basketball_manual_dl_raw():
    # Show all the columns
    pd.set_option('display.max_columns', None)

    df = pd.read_csv(os.path.join(config.DATA_FOLDER_PATH, 'sports', 'basketball_2017_raw.csv'))
    # df.index.name = 'Date'

    # logging.info(f'Head: {df.head()}')

    # df = df.drop(df.columns[range(20, len(df.columns))], axis=1)

    df = df.drop(['Rk'], axis=1)

    df = df.join(df['Result'].str.split(' ', expand=True).add_prefix('rez_'))
    df = df.join(df['rez_1'].str.split('-', expand=True).add_prefix('rez_score'))

    df = df.drop(['rez_1', 'rez_2'], axis=1)

    col_name_away_home = 'Away_Or_Home'
    df = df.rename(columns={'Unnamed: 3': col_name_away_home})

    # df = df.replace(np.nan, 'away', regex=True)
    df[col_name_away_home] = df[col_name_away_home].fillna('away')

    logging.debug(f'Col: {df.columns}')
    logging.debug(f'Head: {df.head()}')

    df = df.rename(columns={'rez_0': 'result_w1'})
    df = df.rename(columns={'rez_score0': 'team_score'})
    df = df.rename(columns={'rez_score1': 'opp_score'})

    df = df.drop(['Result'], axis=1)

    le_team = LabelEncoder()
    df['team_encoded'] = le_team.fit_transform(df['Tm'])
    df['opp_encoded'] = le_team.fit_transform(df['Opp'])

    lb_away_home = sklearn.preprocessing.LabelBinarizer()
    df['away_home_encoded'] = lb_away_home.fit_transform(df[col_name_away_home])

    df = df.drop(['Tm'], axis=1)
    df = df.drop(['Opp'], axis=1)
    df = df.drop([col_name_away_home], axis=1)

    df['Date'] = pd.to_datetime(df['Date']).astype(np.int64)
    df = df.set_index('Date')

    logging.debug(f'Col: {df.columns}')
    logging.debug(f'Head: {df.head()}')

    lb_result_w1 = LabelEncoder()
    lb_result_w1.fit(df['result_w1'])
    df['result_w1'] = lb_result_w1.transform(df['result_w1'])

    return df, lb_result_w1


def one_hot_code_y_labels(y_labels):
    lb = sklearn.preprocessing.LabelBinarizer()
    lb.fit(y_labels)
    return lb.transform(y_labels)


def impose_limit(X, Y, limit=None):
    if limit is not None:
        X, Y = X[:limit], Y[:limit]

    return X, Y


def scale_split(data):
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0  # data is from 0..255
    Y = data[:, 0]

    return X, Y


def otherize_classes(df, classes=[]):
    if len(classes) >= 10:
        raise Exception(f'Error. Too many classes: {len(classes)}')

    if len(classes) > 0:
        df_new = df.copy(deep=True)
        df_new = df_new.iloc[0:0]
        for digit in range(0, 10):
            df_digits = df[df['label'] == digit]
            if digit in classes:
                df_digits['label'] = classes[0]
            df_new = pd.concat([df_new, df_digits])
    else:
        df_new = df

    return df_new.sample(frac=1).reset_index(drop=True)


def get_xor():
    X = np.zeros((200, 2))
    X[:50] = np.random.random((50, 2)) / 2 + 0.5  # (0.5-1, 0.5-1)
    X[50:100] = np.random.random((50, 2)) / 2  # (0-0.5, 0-0.5)
    X[100:150] = np.random.random((50, 2)) / 2 + np.array([[0, 0.5]])  # (0-0.5, 0.5-1)
    X[150:] = np.random.random((50, 2)) / 2 + np.array([[0.5, 0]])  # (0.5-1, 0-0.5)
    Y = np.array([0] * 100 + [1] * 100)
    return X, Y


def get_donut():
    N = 200
    R_inner = 5
    R_outer = 10

    # distance from origin is radius + random normal
    # angle theta is uniformly distributed between (0, 2pi)
    R1 = np.random.randn(N // 2) + R_inner
    theta = 2 * np.pi * np.random.random(N // 2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

    R2 = np.random.randn(N // 2) + R_outer
    theta = 2 * np.pi * np.random.random(N // 2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([X_inner, X_outer])
    Y = np.array([0] * (N // 2) + [1] * (N // 2))
    return X, Y


def save_model(model, basename):
    model_json = model.to_json()
    with open("{}.json".format(basename), "w") as json_file:
        json_file.write(model_json)

    # Serialize weights to HDF5
    save_path = "{}.h5".format(basename)
    model.save_weights(save_path)
    print("Saved model \'{}\' to disk".format(save_path))
