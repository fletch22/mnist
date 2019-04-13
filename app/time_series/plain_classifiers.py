import logging

import numpy as np
import pandas as pd
import sklearn
from mlxtend.classifier import StackingCVClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import tree
from xgboost import XGBClassifier

from app.models.lightgbm.LightGbm import LightGbm

RANDOM_SEED = 42


class CLF_TYPES():
    KNN = 'KNN',
    GBC_1 = 'GBC_1'
    GBC_2 = 'GBC_2'
    GNB = 'GNB'
    MultinomialNB = 'MultinomialNB'
    ComplementNB = 'ComplementNB'
    BernoulliNB = 'BernoulliNB'
    DecisionTreeClassifier = 'DecisionTreeClassifier'
    ExtraTreeClassifier = 'ExtraTreeClassifer'
    SVC = 'SVC'
    LogReg100 = 'Log100'
    LogReg10k = 'Log10k'
    LR_L2_10 = 'LR_L2_10'
    StackingCVClassifier = 'StackingCVClassifier'
    RandomForestClassifier500 = 'RandomForestClassifier500'
    RandomForestClassifier50 = 'RandomForestClassifier50'
    RandomForestClassifier5 = 'RandomForestClassifier5'
    BernoulliRBM = 'BernoulliRBM'
    MLPClassifier = 'MLPClassifier'
    AdaBoostClassifier = 'AdaBoostClassifier'
    XGBoost = 'XGB'
    LightGBM = 'LightGBM'


def score_plain_classifiers(years, train_X, train_y, test_X, test_y):
    accs = []
    pipelines = []
    clfs, clf_names = get_stacked_classifiers()

    clf_focussed_name = CLF_TYPES.XGBoost
    clf_focussed = None
    for i in range(len(clf_names) - 1):
        clf = clfs[i]
        clf_name = clf_names[i]

        logging.debug(f'clf_name: {clf_name}')

        estimator = []
        estimator.append(('clf', clf))
        pipeline = Pipeline(estimator)

        pipelines.append(pipeline.fit(train_X, train_y))

        result = pipeline.score(test_X, test_y)

        if clf_name == clf_focussed_name:
            clf_focussed = clf

        logging.info(f'{clf_name} Score: {result}')

        accs.append(result)

    logging.info(f'Years: {years}')
    logging.info(f'Highest score {np.max(accs)}')

    probas_arr = []
    for i in range(len(pipelines) - 1):
        clf = pipelines[i]
        clf_probas = clf.predict_proba(test_X)

        probas_arr.append(clf_probas)

    accuracy_by_thresh = score_by_average_over_thresh(probas_arr, test_y)
    logging.info(f'average_over_thresh accuracy: {accuracy_by_thresh}')

    majority_vote_acc = score_by_majority_vote(probas_arr, test_y)
    logging.info(f'majority vote accuracy: {majority_vote_acc}')

    accuracy_by_high = score_by_highest_of_all(probas_arr, test_y)
    logging.info(f'score_by_highest_of_all accuracy: {accuracy_by_high}')

    accuracy_by_drop = score_by_drop_low([clf_focussed.predict_proba(test_X)], test_y)
    logging.info(f'score_by_drop_low accuracy: {accuracy_by_drop}')


def score_by_average_over_thresh(all_probas, y):
    acc = put_probas_in_score_array(all_probas)

    mean_res = np.mean(acc, axis=1)
    wins = (mean_res >= .5)  # , dtype=int)

    return ((y == wins).sum()) / wins.shape[0]


def score_by_majority_vote(all_probas, y):
    probas_array = put_probas_in_score_array(all_probas)

    acc = []
    for row in probas_array:
        win_count = 0
        for v in row:
            if v >= .5:
                win_count += 1
        acc.append(win_count/len(row))

    acc_arr = np.array(acc)

    wins = (acc_arr >= .5)

    return ((y == wins).sum()) / wins.shape[0]


def score_by_drop_low(all_probas, y):
    acc = put_probas_in_score_array(all_probas)

    mean_res = np.mean(acc, axis=1)

    orig_size = mean_res.shape[0]

    df = pd.DataFrame(mean_res, columns=['avg'])
    df['label'] = y

    df_sorted = df.set_index('avg').sort_index()

    frac_sample = .10
    tot_size = df_sorted.shape[0]
    size = int(tot_size * frac_sample)
    bottom_range = range(size)
    top_range = range(tot_size - 1, tot_size - 1 - size, -1)
    tot_range = list(bottom_range) + list(top_range)

    df_filtered = df_sorted.iloc[tot_range]

    df_unindexed = df_filtered.reset_index(level='avg')

    y = df_unindexed['label'].values
    mean_res = df_unindexed['avg'].values

    wins = (mean_res >= .5)  # , dtype=int)

    if wins.shape[0] == 0:
        raise Exception('Encounterd problem with sample size. After filtering, there were no samples left to test.')

    logging.info(f'Num samples {wins.shape[0]}/{orig_size}')

    return ((y == wins).sum()) / wins.shape[0]


def put_probas_in_score_array(all_probas):
    arr1 = all_probas[0][:, 1]
    acc = arr1.reshape(arr1.shape[0], 1)
    for i in range(1, len(all_probas)):
        arr_current = all_probas[i][:, 1]
        arr_current = arr_current.reshape(arr_current.shape[0], 1)
        acc = np.append(acc, arr_current, axis=1)
    return acc


def score_by_highest_of_all(all_probas, y):
    acc = all_probas[0]
    for i in range(1, len(all_probas)):
        arr_current = all_probas[i]
        acc = np.append(acc, arr_current, axis=1)

    def did_win(row):
        index_max = list(row).index(max(list(row)))
        return index_max % 2 > 0

    results = []
    for res in list(acc):
        results.append(did_win(res))

    wins = np.array(results)

    return ((y == wins).sum()) / wins.shape[0]


def manual_stacking(pipelines, test_X, test_y, clf_names):
    X_probas = None
    for i in range(len(pipelines) - 1):
        clf = pipelines[i]
        clf_probas = clf.predict_proba(test_X)

        if X_probas is None:
            X_probas = clf_probas
        else:
            X_probas = np.append(X_probas, clf_probas, axis=1)

    lr = LogisticRegression()
    lr.fit()
    result = lr.score(X_probas, test_y)
    logging.info(f'Combined Score: {result}')

    for i in range(len(clf_names) - 1):
        clf = pipelines[i]
        clf_name = clf_names[i]

        logging.debug(f'clf_name: {clf_name}')

        result = clf.score(test_X, test_y)

        logging.info(f'{clf_name} Score: {result}')


def get_stacked_classifiers():
    dict_clfs = get_classifiers()

    kwargs = {'C': 100.0, 'dual': False, 'fit_intercept': True, 'multi_class': 'multinomial', 'penalty': 'l2',
              'solver': 'saga'}
    lr = LogisticRegression(**kwargs)

    names = [clf_name for clf_name in dict_clfs.keys()]
    classifiers = [dict_clfs[clf_name] for clf_name in names]

    # names = names[0:1]
    # classifiers = classifiers[0:1]

    clf_stacked = StackingCVClassifier(classifiers=classifiers, use_probas=True,
                                       use_features_in_secondary=True,
                                       meta_classifier=lr)
    names.append(CLF_TYPES.StackingCVClassifier)
    classifiers.append(clf_stacked)

    return classifiers, names


def get_classifiers():
    dict_clfs = {}

    dict_clfs[CLF_TYPES.RandomForestClassifier500] = RandomForestClassifier(n_estimators=500, n_jobs=-1, max_features='auto')

    dict_clfs[CLF_TYPES.RandomForestClassifier50] = RandomForestClassifier(n_estimators=50, n_jobs=-1)

    dict_clfs[CLF_TYPES.RandomForestClassifier5] = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, n_jobs=-1)

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


    # dict_clfs[CLF_TYPES.GNB] = GaussianNB()

    kwargs = {'alpha': 0.10526315789473684}
    # kwargs = {}
    # dict_clfs[CLF_TYPES.MultinomialNB] = MultinomialNB(**kwargs)

    # kwargs = None
    kwargs = {'alpha': 0.10526315789473684, 'norm': False}
    # kwargs = {}
    # dict_clfs[CLF_TYPES.ComplementNB] = ComplementNB(**kwargs)

    # kwargs = None
    kwargs = {'alpha': 0.05263157894736842, 'binarize': 0.9473684210526315}
    # kwargs = {}
    # dict_clfs[CLF_TYPES.BernoulliNB] = BernoulliNB(**kwargs)

    kwargs = {'criterion': 'gini', 'max_depth': 1.0, 'max_features': 2, 'min_samples_leaf': 0.4, 'min_samples_split': 0.01, 'min_weight_fraction_leaf': 0.4,
              'random_state': 42, 'splitter': 'best'}
    # kwargs = {}
    dict_clfs[CLF_TYPES.DecisionTreeClassifier] = tree.DecisionTreeClassifier(**kwargs)

    dict_clfs[CLF_TYPES.ExtraTreeClassifier] = tree.ExtraTreeClassifier()

    kwargs = {'C': 10, 'gamma': 0.001, 'kernel': 'rbf', 'random_state': 42, 'probability': True}
    # kwargs = {}
    dict_clfs[CLF_TYPES.SVC] = sklearn.svm.SVC(**kwargs)

    kwargs = {'penalty': 'l2', 'C': 10}
    dict_clfs[CLF_TYPES.LR_L2_10] = LogisticRegression(**kwargs)

    kwargs = {'C': 1.0, 'dual': False, 'fit_intercept': True, 'max_iter': 1000, 'multi_class': 'ovr', 'penalty': 'l1', 'solver': 'liblinear'}
    # kwargs = {}
    dict_clfs[CLF_TYPES.LogReg100] = LogisticRegression(**kwargs)
    dict_clfs[CLF_TYPES.LogReg10k] = LogisticRegression(C=10000)

    # dict_clfs = {}
    #
    # kwargs = dict(learning_rate=0.1)
    # # kwargs = {}
    # dict_clfs[CLF_TYPES.GBC_1] = GradientBoostingClassifier(**kwargs)
    #
    # kwargs = dict(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    # dict_clfs[CLF_TYPES.GBC_2] = GradientBoostingClassifier(**kwargs)

    dict_clfs[CLF_TYPES.XGBoost] = XGBClassifier()

    # dict_clfs = {}
    # dict_clfs[CLF_TYPES.LightGBM] = LightGbm()

    return dict_clfs


def logits_ridge_selection(X, y, columns):
    param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100]}
    pipe = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2'))
    grid = GridSearchCV(pipe, param_grid, cv=10)
    grid.fit(X, y)

    print(grid.best_params_)

    c = grid.best_params_['logisticregression__C']

    X_scaled = StandardScaler().fit_transform(X)
    clf = LogisticRegression(penalty='l2', C=c)
    clf.fit(X_scaled, y)

    abs_feat = []
    for i in range(X.shape[1]):
        coef = clf.coef_[0, i]
        abs_feat.append((abs(coef), columns[i]))

    features = sorted(abs_feat, reverse=True)
    logging.info(f'Features: {features}')
    logging.info(f'The five best features: {features[:5]}')
    logging.info(f'The five worst features: {features[-5:]}')
