import time

import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import logging

benchmark_cols = ['Variance retained', 'n_Components', 'Time(s)', 'Accuracy_percentage']



def benchmark_pca(benchmark, variance, train_X, train_y, test_X, test_y):
    # logging.info(f'Train X shape: {train_X.shape}')
    pca = PCA(variance)
    pca.fit(train_X)
    n_components = pca.n_components_
    train_X = pca.transform(train_X)

    # pca.fit(test_img)
    test_X = pca.transform(test_X)
    logisticRegr = LogisticRegression(solver='lbfgs')
    start = time.time()
    logisticRegr.fit(train_X, train_y)
    end = time.time()

    timing = end - start

    # Predict for Multiple Observations (images) at Once
    predicted = logisticRegr.predict(test_X)

    # generate evaluation metrics
    accuracy = (metrics.accuracy_score(test_y, predicted))

    logging.info(f'ac: {accuracy}')
    # return
    a = dict(zip(benchmark_cols, [variance, n_components, timing, accuracy]))

    # row_df = pd.DataFrame([variance, n_components, timing, accuracy], benchmark_cols)
    benchmark = benchmark.append(a, ignore_index=True)
    # benchmark = pd.concat([row_df, benchmark], axis=0).reset_index()

    return benchmark


# def show_fit(benchmark, train_X, train_y, test_X, test_y):
#     variance = 1.0
#     n_components = train_X.shape[1]
#
#     logisticRegr = LogisticRegression(solver='lbfgs')
#     start = time.time()
#     logisticRegr.fit(train_X, train_y)
#     end = time.time()
#     timing = end - start
#     # Predict for Multiple Observations (images) at Once
#     predicted = logisticRegr.predict(test_X)
#     # generate evaluation metrics
#     accuracy = (metrics.accuracy_score(test_y, predicted))
#
#     logging.info(f'acc: {accuracy}')
#
#     a = dict(zip(benchmark_cols, [variance, n_components, timing, accuracy]))
#     benchmark.append(a, ignore_index=True)
#
#     print(benchmark)


def show_pca(train_X, train_y, test_X, test_y):
    # benchmark = pd.DataFrame(columns=benchmark_cols)
    # variance_list = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
    #
    # for variance in variance_list:
    #     benchmark = benchmark_pca(benchmark, variance, train_X, train_y, test_X, test_y)
    #
    # import matplotlib.pyplot as plt
    # benchmark.plot(x=0, y=-1)
    # plt.title("variance vs accuracy")
    # plt.show()
    #
    # benchmark.plot(x=1, y=-1)
    # plt.title("no of components vs accuracy")
    # plt.show()
    #
    # benchmark.plot(x=2, y=-1)
    # plt.title("time vs accuracy")
    # plt.show()

    pca = PCA(n_components=train_X.shape[1])
    pca.fit(train_X)

    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')

    plt.show()
