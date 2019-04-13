from __future__ import print_function

import os

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from keras.utils.np_utils import to_categorical
import sklearn
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import logging

from app import util
from app.config import config
from app.stacking import mnist
from app.time_series.nba_team_game_learning import get_train_and_test_data

RANDOM_SEED = 42

def get_grad_boost_clf():
    # best =
    param_grids = []
    param_grids.append(
        dict(loss=['deviance', 'exponential'],
             learning_rate=[.1, .01, .01],
             n_estimators=[100, 1000, 10000],
             subsample=[.5, .7, .9, 1.0],
             criterion=['friedman_mse', 'mse', 'mae'],
             min_samples_split=[.1, .3, .5, .7, .9],
             min_samples_leaf=[0.1, .5, 1, 2],
             min_weight_fraction_leaf=[.0, .1, .2, .3, .5],
             max_depth=[3, 5, 7, 10],
             random_state=[RANDOM_SEED]
             ))

    clf = GradientBoostingClassifier()

    return clf, param_grids


def get_decisiontree_clf(num_features):
    # best = {'criterion': 'gini', 'max_depth': 4.0, 'max_features': None, 'min_samples_leaf': 0.1, 'min_samples_split': 0.1, 'min_weight_fraction_leaf': 0.0, 'random_state': 42, 'splitter': 'best'}
    param_grids = []

    max_depth = list(np.linspace(1, 10, 32, endpoint=True))
    max_depth = np.append(max_depth, None)

    max_features = list(range(1, num_features))
    max_features.append('auto')
    max_features.append('sqrt')
    max_features.append('log2')
    max_features.append(None)

    param_grids.append(dict(criterion=['gini', 'entropy'],
                            splitter=['best', 'random'],
                            max_depth=max_depth,
                            min_samples_split=np.linspace(0.01, 0.12, 10, endpoint=True),
                            min_samples_leaf=np.linspace(0.4, 0.5, 5, endpoint=True),
                            min_weight_fraction_leaf=np.linspace(0.4, 0.5, 5),
                            max_features=max_features,
                            random_state=[RANDOM_SEED]))

    clf = DecisionTreeClassifier()

    return clf, param_grids


def get_knn():
    # best = {'algorithm': 'auto', 'leaf_size': 20, 'metric': 'minkowski', 'n_jobs': 6, 'n_neighbors': 5, 'p': 2, 'weights': 'distance'}
    param_grids = []
    param_grids.append(dict(n_neighbors=[5, 6, 7, 8],
                            weights=['uniform', 'distance'],
                            algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'],
                            leaf_size=[5, 10, 15, 20, 21],
                            p=[1, 2],
                            n_jobs=[12], metric=['minkowski', 'euclidean', 'manhattan', 'chebyshev']))

    clf = KNeighborsClassifier()

    return clf, param_grids


def get_log_reg():
    # best = {'C': 100.0, 'dual': False, 'fit_intercept': True, 'multi_class': 'multinomial', 'penalty': 'l2', 'solver': 'saga'}
    param_grids = []
    param_grids.append(dict(penalty=['l1'], solver=['liblinear', 'saga'], dual=[False], C=[1.0, 10.0, 100.0, 1000.0],
                            fit_intercept=[True, False], multi_class=['ovr', 'auto'], max_iter=[1000]))
    param_grids.append(
        dict(penalty=['l2'], solver=['newton-cg', 'saga', 'lbfgs'], dual=[False], C=[1.0, 10.0, 100.0, 1000.0],
             fit_intercept=[True, False], multi_class=['ovr', 'multinomial', 'auto'], max_iter=[1000]))

    clf = LogisticRegression()

    return clf, param_grids


def get_bernouilli_nb():
    # best = {'alpha': 0.3157894736842105, 'binarize': 0.5263157894736842}
    alpha_lin_space = np.linspace(0, 1, 20)[1:]
    binarize_lin_space = np.linspace(0, 2, 20)[1:]
    tuned_parameters = dict(alpha=alpha_lin_space, binarize=binarize_lin_space)

    clf = BernoulliNB()

    return clf, tuned_parameters


def get_complement_nb():
    # best = dict(alpha=1.894736842105263, norm=False)
    tuned_parameters = dict(alpha=np.linspace(0, 2, 20)[1:], norm=[True, False])

    clf = ComplementNB()

    logging.info(f'Esimator keys: {clf.get_params().keys()}')

    return clf, tuned_parameters


def get_mutlinomial_nb():
    # best = dict(alpha=0.631578947368421)
    tuned_parameters = dict(alpha=np.linspace(0, 2, 20)[1:])

    clf = MultinomialNB()

    return clf, tuned_parameters


def get_svc():
    # tuned_parameters = [
    #     {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000], 'random_state': [RANDOM_SEED]},
    #     {'kernel': ['linear'], 'C': [0.25, 1, 10, 100, 1000], 'cache_size': [100, 200], 'probability': [True],
    #      'random_state': [RANDOM_SEED]}
    # ]
    # tuned_parameters = [{'kernel': ['linear'], 'C': [0.025], 'cache_size': [200], 'probability': [True], 'random_state': [RANDOM_SEED]}]

    # best dict(C=10, gamma=0.001, kernal='rbf', random_state=RANDOM_SEED)
    tuned_parameters = [
        {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000], 'random_state': [RANDOM_SEED]}
    ]

    clf = SVC()

    return clf, tuned_parameters


def get_clfs(num_features=None):
    clfs = []
    # clfs.append(get_svc())
    # clfs.append(get_mutlinomial_nb())
    # clfs.append(get_complement_nb())
    # clfs.append(get_bernouilli_nb())
    # clfs.append(get_knn())
    # clfs.append(get_log_reg())
    clfs.append(get_decisiontree_clf(num_features))
    clfs.append(get_grad_boost_clf())
    return clfs


def main_search():
    logging.info('XXXX')

    # _, _, X_train, X_test, Y_train, Y_test = mnist.get_data(limit=1000)
    # X_train, Y_train, X_test, Y_test = util.get_basketball_manual_dl(30, 2)

    X_test, X_train, Y_test, Y_train = get_nba_team_game_data()

    logging.info(f'Shape: {X_train.shape}')

    num_features = X_train.shape[1]

    for clf, param_grid in get_clfs(num_features):
        scores = ['precision_macro']
        # scores = ['roc_auc']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(clf, param_grid=param_grid, cv=5,
                               scoring=score, n_jobs=12)
            clf.fit(X_train, Y_train)

            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = Y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()

        # Note the problem is too easy: the hyperparameter plateau is too flat and the
        # output model is the same for precision and recall with ties in quality.


def get_nba_team_game_data():
    start_year = 2017
    end_year = None
    years = start_year if end_year is None else f'{start_year}-{end_year}'
    shelf_path = os.path.join(config.SHELVES_DIR, f'{years}_w_last_player_pos_A')

    X, y, X_train, Y_train, X_test, Y_test, _ = get_train_and_test_data(2017, None, None, 14, 12, .7,
                                                                        shelf_path, empty_shelves=False)
    return X_test, X_train, Y_test, Y_train


if __name__ == "__main__":
    main_search()
