import logging
import os

from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dense
from mlxtend.classifier import StackingCVClassifier
from sklearn import svm, model_selection
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import tree
import pandas as pd

import numpy as np

from app import util
from app.config import config
from app.time_series.plot import show_performance

RANDOM_SEED = 42


def learn():
    num_teams = 30
    num_histories = 1
    epochs = 50
    neurons = 1000

    # num_teams = 5
    # num_histories = 9
    # epochs = 2
    # neurons = 1000

    train_X, train_y, test_X, test_y = get_basketball_manual_dl(num_teams, num_histories)

    logging.debug(f'test: {train_X[:, 0].shape}')

    # train_X = train_X.reshape(train_X[0], train_X[1], train_X[2]) # np.reshape(train_X, (train_X[0], train_X[2], train_X[1]))

    train_X = train_X[:, 0]

    # Best: MultinomialNB Score: 0.5913793103448276
    accs = []
    clfs, clf_names = get_stacked_classifiers()
    for i in range(len(clf_names)):
        clf = clfs[i]
        clf_name = clf_names[i]
        scores =  model_selection.cross_val_score(clf, train_X, train_y, cv=3, scoring='accuracy')
        logging.info(f'{clf_name} accuracy: {scores.mean()} (+/- {scores.std()})')

        clf = clf.fit(train_X, train_y)

        result = clf.score(test_X[:, 0], test_y)

        logging.info(f'{clf_name} Score: {result}')

        accs.append(result)
        # return

    logging.info('')
    logging.info(f'Highest score {np.max(accs)}')

    # history = None
    # logging.debug(f'df_team_one.shape: {df_team_one.shape}')
    #
    # # df_team_one = df_team_one.drop(['team_encoded'], axis=1)
    # #
    # single_team = prep_for_time_series(df_team_one, num_histories)
    #
    # logging.debug(f'single_team.shape: {single_team.shape}')
    #
    # df_result_w1 = single_team['result_w1']
    # single_team = single_team.drop(['result_w1'], axis=1)
    #
    # final_X_arr = reshape_for_lstm(single_team.values)

    # final_X_arr = single_team.values[:, :-1]

    # logging.debug(f'final_X_arr_X_arr.shape: {final_X_arr.shape}')

    # single_team_lstm = reshape_for_lstm(final_X_arr)
    #
    # logging.debug(f'single_team_lstm.shape: {single_team_lstm.shape}')
    #
    # logging.debug(single_team_lstm.shape)
    #
    # best_model = load_model(get_best_val_path())

    # logging.info(f'history: {history}')

    # if hasattr(history, 'history'):
    #     best_val_acc = np.max(history.history['val_acc'])
    # else:
    #     best_val_acc = scores.mean()
    #
    # logging.info(f'val_acc: {best_val_acc}')
    #
    # roi = calc_return(best_val_acc, 100, 10)
    # logging.info(f'Roi: {roi}')

    # y_predict = best_model.predict(single_team_lstm)
    #
    # logging.info(f'y pred: {y_predict}')

    return

def get_basketball_manual_dl(num_teams, num_histories):
    df, lb_result_w = util.get_basketball_manual_dl_raw()

    df_team_one = df.loc[df['team_encoded'] == 1].copy()
    logging.debug(f'df_team_one.columns: {df_team_one.columns}')
    logging.debug(f'df_team_one.shape: {df_team_one.shape}')

    df = df.loc[(df['team_encoded'] != 1) & (df['opp_encoded'] != 1)]

    logging.debug(f'df.columns: {df_team_one.columns}')
    logging.debug(f'df.shape: {df.shape}')

    # show_time_series_chart(df, range(17))

    train_X, train_y, test_X, test_y = None, None, None, None

    for i in range(num_teams):

        df_slim = df.loc[df['team_encoded'] == i + 1]
        # df_slim = df_slim.drop(['team_encoded'], axis=1)

        if len(df_slim) > 0:
            reframed_single_team = prep_for_time_series(df_slim, num_histories)

            train_X_single, train_y_single, test_X_single, test_y_single = split_train_test(reframed_single_team, .7)

            logging.debug(f'train_X_single.shape: {train_X_single.shape}')

            train_X = append_arr(train_X, train_X_single)
            train_y = append_arr(train_y, train_y_single)
            test_X = append_arr(test_X, test_X_single)
            test_y = append_arr(test_y, test_y_single)

    logging.debug(f'train_X.shape: {train_X.shape}')

    return train_X, train_y, test_X, test_y


def get_lstm_model(num_neurons, train_X, train_y, test_X, test_y, epochs):
    # design network
    model = Sequential()
    model.add(LSTM(num_neurons, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])

    history = model.fit(train_X, train_y, epochs=epochs, batch_size=72, validation_data=(test_X, test_y), verbose=1,
                        shuffle=False, callbacks=get_callbacks())

    show_performance(history)


def get_stacked_classifiers():
    dict_clfs = get_classifiers()

    kwargs = {'C': 100.0, 'dual': False, 'fit_intercept': True, 'multi_class': 'multinomial', 'penalty': 'l2',
              'solver': 'saga'}
    lr = LogisticRegression(**kwargs)

    names = [clf_name for clf_name in dict_clfs.keys()]
    classifiers = [dict_clfs[clf_name] for clf_name in names]

    names = names[0:1]
    classifiers = classifiers[0:1]

    clf_stacked = StackingCVClassifier(classifiers=classifiers, use_probas=True,
                                       use_features_in_secondary=True,
                                       meta_classifier=lr)
    names.append(CLF_TYPES.StackingCVClassifier)
    classifiers.append(clf_stacked)

    return classifiers, names


class CLF_TYPES():
    KNN = 'KNN',
    GBC = 'GBC'
    GNB = 'GNB'
    MultinomialNB = 'MultinomialNB'
    ComplementNB = 'ComplementNB'
    BernoulliNB = 'BernoulliNB'
    DecisionTreeClassifier = 'DecisionTreeClassifier'
    ExtraTreeClassifier = 'ExtraTreeClassifer'
    SVC = 'SVC'
    LogReg100 = 'Log100'
    LogReg10k = 'Log10k'
    StackingCVClassifier = 'StackingCVClassifier'
    RandomForestClassifier50 = 'RandomForestClassifier50'
    RandomForestClassifier5 = 'RandomForestClassifier5'
    BernoulliRBM = 'BernoulliRBM'
    MLPClassifier = 'MLPClassifier'
    AdaBoostClassifier = 'AdaBoostClassifier'


def get_classifiers():
    dict_clfs = {}

    dict_clfs[CLF_TYPES.RandomForestClassifier50] = RandomForestClassifier(n_estimators=50, n_jobs=12)

    dict_clfs[CLF_TYPES.RandomForestClassifier5] = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, n_jobs=12)

    dict_clfs[CLF_TYPES.BernoulliRBM] = Pipeline(steps=[('rbm', BernoulliRBM(n_components=200,
                                                                             n_iter=1,
                                                                             learning_rate=0.01,
                                                                             verbose=False)),
                                                        ('logistic', LogisticRegression(C=10000))])

    dict_clfs[CLF_TYPES.MLPClassifier] = MLPClassifier(hidden_layer_sizes=(75,), max_iter=250, alpha=1e-4, solver='sgd', verbose=0, tol=1e-4,
                                                       random_state=RANDOM_SEED, learning_rate_init=.1, early_stopping=True)

    kwargs = dict(n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=RANDOM_SEED)
    dict_clfs[CLF_TYPES.AdaBoostClassifier] = AdaBoostClassifier(**kwargs)

    kwargs = {'algorithm': 'auto', 'leaf_size': 5, 'metric': 'minkowski', 'n_jobs': 12, 'n_neighbors': 6, 'p': 1, 'weights': 'distance'}
    # kwargs = {}
    dict_clfs[CLF_TYPES.KNN] = KNeighborsClassifier(**kwargs)

    kwargs = dict(learning_rate=0.1)
    # kwargs = {}
    dict_clfs[CLF_TYPES.GBC] = GradientBoostingClassifier(**kwargs)

    dict_clfs[CLF_TYPES.GNB] = GaussianNB()

    kwargs = {'alpha': 0.10526315789473684}
    # kwargs = {}
    dict_clfs[CLF_TYPES.MultinomialNB] = MultinomialNB(**kwargs)

    # kwargs = None
    kwargs = {'alpha': 0.10526315789473684, 'norm': False}
    # kwargs = {}
    dict_clfs[CLF_TYPES.ComplementNB] = ComplementNB(**kwargs)

    # kwargs = None
    kwargs = {'alpha': 0.05263157894736842, 'binarize': 0.9473684210526315}
    # kwargs = {}
    dict_clfs[CLF_TYPES.BernoulliNB] = BernoulliNB(**kwargs)

    kwargs = {'criterion': 'gini', 'max_depth': 1.0, 'max_features': 3, 'min_samples_leaf': 0.4, 'min_samples_split': 0.01, 'min_weight_fraction_leaf': 0.4, 'random_state': 42, 'splitter': 'best'}
    # kwargs = {}
    dict_clfs[CLF_TYPES.DecisionTreeClassifier] = tree.DecisionTreeClassifier(**kwargs)

    dict_clfs[CLF_TYPES.ExtraTreeClassifier] = tree.ExtraTreeClassifier()

    kwargs = {'C': 10, 'gamma': 0.001, 'kernel': 'rbf', 'random_state': 42, 'probability': True}
    # kwargs = {}
    dict_clfs[CLF_TYPES.SVC] = svm.SVC(**kwargs)

    kwargs = {'C': 1.0, 'dual': False, 'fit_intercept': True, 'max_iter': 1000, 'multi_class': 'ovr', 'penalty': 'l1', 'solver': 'liblinear'}
    # kwargs = {}
    dict_clfs[CLF_TYPES.LogReg100] = LogisticRegression(**kwargs)
    dict_clfs[CLF_TYPES.LogReg10k] = LogisticRegression(C=10000)

    return dict_clfs


def get_best_val_path():
    return os.path.join(config.SESSION_DIR, 'best_basketball_model.h5')


def get_callbacks():
    return [ModelCheckpoint(get_best_val_path(), 'val_acc', 1, save_best_only=True)]


def calc_return(val_acc, amount_bet, number_bets):
    raw_result = ((val_acc * number_bets) * (amount_bet * 1.91)) - (((1 - val_acc) * number_bets) * amount_bet)
    tot_invest = amount_bet * number_bets
    roi = (raw_result - (tot_invest)) / tot_invest
    return round(roi, 4)

def append_arr(accumulator, new_arr):
    if accumulator is None:
        accumulator = new_arr
    else:
        accumulator = np.concatenate((accumulator, new_arr), axis=0)

    return accumulator


def prep_for_time_series(df, num_histories):
    logging.debug(f'Len Cols: {len(df.columns)}')
    logging.debug(f'Cols: {df.columns}')

    chained_assignment_orig = pd.options.mode.chained_assignment
    pd.options.mode.chained_assignment = None
    df['team_score'] = df['team_score'].astype('float32')
    df['opp_score'] = df['opp_score'].astype('float32')
    df['diff_score'] = df['team_score'].sub(df['opp_score'], axis=0)
    pd.options.mode.chained_assignment = chained_assignment_orig

    del_cols = ['MP', 'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 'FT',
                'FTA', 'FT%', 'PTS', 'OFG', 'OFGA', 'OFG%', 'O2P', 'O2PA', 'O2P%',
                'O3P', 'O3PA', 'O3P%', 'OFT', 'OFTA', 'OFT%', 'OPTS', 'opp_score', 'team_score']
    df = df.drop(del_cols, axis=1)

    data = df.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)
    # reframed = series_to_supervised(scaled, columns=df.columns, n_in=num_histories, n_out=1, skip_rows=num_histories)

    del_cols = ['diff_score']

    reframed = pd.DataFrame(scaled, columns=df.columns)
    reframed = reframed.drop(del_cols, axis=1)

    # del_cols = ['MP', 'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 'FT',
    #             'FTA', 'FT%', 'PTS', 'OFG', 'OFGA', 'OFG%', 'O2P', 'O2PA', 'O2P%',
    #             'O3P', 'O3PA', 'O3P%', 'OFT', 'OFTA', 'OFT%', 'OPTS', 'team_score', 'opp_score']
    # reframed_thinned = reframed.drop(del_cols, axis=1)
    # pd.DataFrame(scaled, columns=df.columns)

    # logging.debug(f'reframed.columns {reframed.columns}')
    # logging.debug(f'reframed_thinned.columns {reframed_thinned.columns}')

    # 'team_encoded', 'opp_encoded', 'result_w1'
    # 'away_home_encoded'

    # logging.debug(f'Cols: {reframed.columns}')

    return reframed

def split_train_test(df, train_split_fraction):
    # split into train and test sets

    df_result_w1 = df['result_w1']
    df = df.drop(['result_w1'], axis=1)

    values = df.values
    tot_size = len(values)

    logging.debug(f'Tot size: {tot_size}')

    train_size = int(np.floor(train_split_fraction * tot_size))

    logging.debug(f'train_size: {train_size}')

    # test_size = tot_size - train_size

    train = values[:train_size, :]
    test = values[train_size:, :]
    # split into input and outputs

    # train_X, train_y = train[:, :-1], train[:, -1]
    # test_X, test_y = test[:, :-1], test[:, -1]

    train_X = train
    test_X = test

    y_stuff = df_result_w1.values
    train_y = y_stuff[:train_size, ]
    test_y = y_stuff[train_size:, ]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = reshape_for_lstm(train_X)  # train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = reshape_for_lstm(test_X)  # test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    logging.debug(f'{train_X.shape}, {train_y.shape}, {test_X.shape}, {test_y.shape}')

    return train_X, train_y, test_X, test_y


def reshape_for_lstm(X):
    return X.reshape((X.shape[0], 1, X.shape[1]))

