import warnings

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.model_selection import train_test_split
import time
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn import tree, model_selection, preprocessing
from sklearn import svm
from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import xgboost as xgb

import numpy as np
import logging

from app import util

CLF_GBC = "GradientBoostingClassifier"
CLF_EXTRA_TREE = "ExtraTreeClassifier"
CLF_SVC = "SVC"
CLF_KNN = "KNeighborsClassifier"
CLF_RANDOM_50 = "RandomForestClassifier_50"
CLF_RANDOM_5 = "RandomForestClassifier_5"
CLF_MLP = "MLPClassifier"
CLF_BERNOULLI_RBM = "BernoulliRBM"
CLF_F22CNN = "F22Cnn"

RANDOM_SEED = 42

shuffle_sample_data = False


def split_X_Y(data):
    X = data[:, 1:] / 255.0  # data is from 0..255
    Y = data[:, 0]

    return X, Y


def get_mnist(file_path, limit):
    df = pd.read_csv(file_path)

    data = df.values
    if shuffle_sample_data is True:
        np.random.shuffle(data)

    if limit is not None:
        data = data[:limit]

    return split_X_Y(data)


def get_train(limit):
    return get_mnist("../input/train.csv", limit)


def get_test():
    df = pd.read_csv("../input/test.csv")

    return df.values / 255.0


def get_data():
    X, y = util.get_basketball_manual_dl_raw()

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, train_size=0.7)

    X_train_scaled = X_train
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)

    return X, y, X_train_scaled, X_test, y_train, y_test


def score_single_clf(clf_list, label, X_test, Y_test):
    clf = find_clf(clf_list, label)
    clf.fit(X_test, Y_test)
    result = clf.score(X_test, Y_test)
    logging.info("{}: Samples: {}; accuracy: {}".format(label, X_test.shape[0], result))
    return clf, clf.predict(X_test)


def find_clf(clf_list, label_to_find):
    for label, clf in clf_list:
        if label == label_to_find:
            return clf

    raise Exception('Could not find classifier \'{}\''.format(label_to_find))


def get_stacking(X_train):
    from sklearn.ensemble import GradientBoostingClassifier

    kwargs = dict(learning_rate=0.1)
    clf_gbc = (CLF_GBC, GradientBoostingClassifier(**kwargs))

    clf_gnb = ('GaussianNB', GaussianNB())

    kwargs = None
    kwargs = dict(alpha=0.631578947368421)
    clf_multinomial_nb = ('MultinomialNB', MultinomialNB(**kwargs))

    # kwargs = None
    kwargs = dict(alpha=1.894736842105263, norm=False)
    clf_complement_nb = ('ComplementNB', ComplementNB(**kwargs))

    # kwargs = None
    kwargs = {'alpha': 0.3157894736842105, 'binarize': 0.5263157894736842}
    clf_bernoulli_nb = ('BernoulliNB', BernoulliNB(**kwargs))

    # kwargs = {'max_depth': 5}
    kwargs = {'criterion': 'gini', 'max_depth': 4.0, 'max_features': None, 'min_samples_leaf': 0.1,
              'min_samples_split': 0.1, 'min_weight_fraction_leaf': 0.0, 'random_state': 42, 'splitter': 'best'}
    clf_tree_dec = ('DecisionTreeClassifier', tree.DecisionTreeClassifier(**kwargs))

    clf_tree_extra = (CLF_EXTRA_TREE, tree.ExtraTreeClassifier())

    # kwargs = dict(kernel="linear", C=0.025, cache_size=200, probability=True)
    kwargs = dict(C=10, gamma=0.001, kernel='rbf', random_state=RANDOM_SEED, probability=True)
    clf_svm = (CLF_SVC, svm.SVC(**kwargs))

    # kwargs = {'C': 100.0}
    kwargs = {'C': 100.0, 'dual': False, 'fit_intercept': True, 'multi_class': 'multinomial', 'penalty': 'l2',
              'solver': 'saga'}
    clf_log_reg_100 = ('LogisticRegression_100', LogisticRegression(**kwargs))
    clf_log_reg_10k = ('LogisticRegression_10k', LogisticRegression(C=10000))

    # kwargs = {'n_neighbors': 3}
    kwargs = {'algorithm': 'auto', 'leaf_size': 20, 'metric': 'minkowski', 'n_jobs': 6, 'n_neighbors': 5, 'p': 2,
              'weights': 'distance'}
    clf_kneighbors = (CLF_KNN, KNeighborsClassifier(**kwargs))

    clf_rand_forest_50 = (CLF_RANDOM_50, RandomForestClassifier(n_estimators=50, n_jobs=12))

    clf_rand_forest_10 = (
        CLF_RANDOM_5, RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, n_jobs=12))

    bernoulli_rbm = (CLF_BERNOULLI_RBM, Pipeline(steps=[('rbm', BernoulliRBM(n_components=200,
                                                                             n_iter=1,
                                                                             learning_rate=0.01,
                                                                             verbose=False)),
                                                        ('logistic', LogisticRegression(C=10000))]))

    mlp = (CLF_MLP, MLPClassifier(hidden_layer_sizes=(75,), max_iter=250, alpha=1e-4,
                                  solver='sgd', verbose=0, tol=1e-4, random_state=RANDOM_SEED,
                                  learning_rate_init=.1, early_stopping=True))

    adaboost = ('AdaBoostClassifier', AdaBoostClassifier())

    # base_clfs = [clf_gnb, clf_multinomial_nb, clf_complement_nb, clf_bernoulli_nb, clf_tree_dec, clf_tree_extra,
    #              clf_svm, clf_log_reg_100, clf_log_reg_10k, clf_kneighbors, clf_rand_forest_50, clf_rand_forest_10,
    #              bernoulli_rbm, mlp, adaboost]

    clf_xgb = ('xgb', xgb.XGBClassifier(objective="multi:softprob", random_state=RANDOM_SEED))

    # base_clfs = [clf_gnb, clf_gbc, clf_xgb, clf_multinomial_nb, clf_complement_nb, clf_bernoulli_nb, clf_tree_dec, clf_tree_extra,
    #              clf_svm, clf_log_reg_100, clf_log_reg_10k, clf_kneighbors, clf_rand_forest_50, clf_rand_forest_10,
    #              bernoulli_rbm, mlp, adaboost]

    logging.info('Shape: {}'.format(X_train.shape))

    # base_clfs = [clf_gnb, clf_gbc, clf_xgb, clf_multinomial_nb, clf_complement_nb, clf_bernoulli_nb,
    #              clf_tree_dec, clf_tree_extra,
    #              clf_svm, clf_log_reg_100, clf_log_reg_10k, clf_kneighbors, clf_rand_forest_50, clf_rand_forest_10,
    #              bernoulli_rbm, mlp, adaboost]

    base_clfs = [clf_gnb, clf_gbc, clf_bernoulli_nb, clf_tree_dec, clf_tree_extra,
                 clf_svm, clf_log_reg_100, clf_log_reg_10k, clf_kneighbors, clf_rand_forest_50, clf_rand_forest_10,
                 bernoulli_rbm, mlp, adaboost]

    kwargs = {'C': 100.0, 'dual': False, 'fit_intercept': True, 'multi_class': 'multinomial', 'penalty': 'l2',
              'solver': 'saga'}
    lr = LogisticRegression(**kwargs)

    classifiers = [x[1] for x in base_clfs]
    stacking = ('StackingCVClassifier', StackingCVClassifier(classifiers=classifiers,
                                                             use_probas=True,
                                                             use_features_in_secondary=True,
                                                             meta_classifier=lr))

    clf_stacking = stacking[1]

    clf_list = base_clfs.copy()
    clf_list.append(stacking)

    return clf_list, clf_stacking


def fit(clf_list, X, y):
    start = time.time()
    val_acc = None

    for label, clf in clf_list:
        logging.info('Processing {}'.format(label))
        scores = model_selection.cross_val_score(clf, X, y, cv=3, scoring='accuracy')

        end = time.time()
        elapsed = (end - start) / 60

        logging.info("Accuracy: {:0.2f} (+/- {:0.2f}) {}: elapsed: {:.2f} min".format(scores.mean(), scores.std(), label, elapsed))
        val_acc = scores.mean()

        start = time.time()

    return val_acc


def show_chart(clf, confusion_matrix, X_test, Y_test):
    plt.figure(figsize=(9, 9))
    sns.heatmap(confusion_matrix, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(clf.score(X_test, Y_test))
    plt.title(all_sample_title, size=15)

    plt.show()


def get_meta_score(X, y, clf_cnn, clf_stacking, clf_meta):
    # 2nd Test Section
    X_cnn = X.reshape(-1, 28, 28, 1)
    pred_cnn = clf_cnn.predict_proba(X_cnn)
    pred_stacking = clf_stacking.predict_proba(X)
    X_ext = np.concatenate([pred_cnn, pred_stacking], axis=1)

    # clf_finals.predict(X_2_test)
    result = clf_meta.score(X_ext, y)
    logging.info("Meta CLF: Samples: {} accuracy: {}".format(X.shape[0], result))


def main():
    # Add StandardScaler to all Scalable columns.
    # Need to turn this into a time series data set for each team.

    total_start = time.time()

    X, y, X_train, X_test, y_train, y_test = get_data()

    logging.info(f'{y[0:5]}')

    y_test = y_test.reshape(-1)
    y_train = y_train.reshape(-1)

    logging.info(f'y_test: {y_test.shape}')

    n_classes = len(set(y_test))
    logging.info('Number of classes {}.'.format(n_classes))

    start_time = total_start

    accuracies = []
    training_iter = 1
    for i in range(0, training_iter):
        clf_list, clf_stacking = get_stacking(X_train)

        val_acc = fit(clf_list, X_train, y_train)

        accuracies.append(val_acc)

        end = time.time()
        elapsed = (end - start_time) / 60
        logging.info("Sub learning stage elapsed time: %.2f minutes" % (elapsed))

        start_time = time.time()

    total_end = time.time()
    elapsed = (total_end - total_start) / 60
    logging.info("Total Elapsed Time: :{:.2f} minutes: Average accuracy: {}".format(elapsed, np.mean(accuracies)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('-sample_number', type=int, help='Your sample amount, enter it.')
    parser.add_argument('-epochs', type=int, help='Your epochs, enter it.')
    args = parser.parse_args()

    main(args.sample_number, args.epochs)
