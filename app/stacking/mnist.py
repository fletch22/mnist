from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import time
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn import tree
from sklearn import svm
from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Conv2D, Activation, GlobalAveragePooling1D, Dense, Dropout, Lambda, Flatten, Convolution2D, \
    MaxPooling2D, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
import argparse
import numpy as np
from app.config import config
import os
from keras.models import load_model

from app import util

CLF_EXTRA_TREE = "ExtraTreeClassifier"
CLF_SVC = "SVC"
CLF_KNN = "KNeighborsClassifier"
CLF_RANDOM_50 = "RandomForestClassifier_50"
CLF_RANDOM_5 = "RandomForestClassifier_5"
CLF_MLP = "MLPClassifier"
CLF_BERNOULLI_RBM = "BernoulliRBM"

RANDOM_SEED = 42


def get_data(limit=None):
    X, Y = util.get_mnist(limit=limit, otherize_digits=[])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, train_size=0.6)

    return X, Y, X_train, X_test, Y_train, Y_test


def main(sample_number, epochs):
    total_start = time.time()

    X, Y, X_train, X_test, Y_train, Y_test = get_data(limit=sample_number)
    n_classes = len(set(Y_train))

    print('Number of classes {}.'.format(n_classes))

    clf_list, clf_stacking = get_stacking()

    y_predict_proba_cnn = predict_proba_cnn(X_train, X_test, Y_train, Y_test, epochs)

    end = time.time()
    elapsed = (end - total_start) / 60
    print("CNN Elapsed Time: %.2f minutes" % (elapsed))

    total_start = time.time()

    fit(clf_list, X_train, Y_train)

    result = clf_stacking.score(X_test, Y_test)
    print("Stacking accuracy: {}".format(result))

    y_predict_proba_stack = clf_stacking.predict_proba(X_test)

    X_predict_proba_combine = np.append(y_predict_proba_cnn, y_predict_proba_stack, 1)

    X_train, X_test, Y_train, Y_test = train_test_split(X_predict_proba_combine, Y_test, shuffle=True, train_size=0.8)

    log_reg = LogisticRegression(random_state=RANDOM_SEED)
    log_reg.fit(X_train, Y_train)
    result = log_reg.score(X_test, Y_test)
    print("Logistic Regression accuracy: {}".format(result))

    # clf_extra, y_predict_final_1 = score_single_clf(clf_list, CLF_EXTRA_TREE, X_test, Y_test)
    # score_single_clf(clf_list, CLF_SVC, X_test, Y_test)
    # score_single_clf(clf_list, CLF_KNN, X_test, Y_test)
    # score_single_clf(clf_list, CLF_MLP, X_test, Y_test)
    # score_single_clf(clf_list, CLF_RANDOM_50, X_test, Y_test)
    # score_single_clf(clf_list, CLF_RANDOM_5, X_test, Y_test)
    # score_single_clf(clf_list, CLF_BERNOULLI_RBM, X_test, Y_test)

    clf_extra, y_predict_final_1 = score_single_clf(clf_list, CLF_EXTRA_TREE, X_test, Y_test)
    # score_single_clf(clf_list, CLF_SVC, X_test, Y_test)
    score_single_clf(clf_list, CLF_KNN, X_test, Y_test)
    # score_single_clf(clf_list, CLF_MLP, X_test, Y_test)
    score_single_clf(clf_list, CLF_RANDOM_50, X_test, Y_test)
    score_single_clf(clf_list, CLF_RANDOM_5, X_test, Y_test)
    # score_single_clf(clf_list, CLF_BERNOULLI_RBM, X_test, Y_test)

    end = time.time()
    elapsed = (end - total_start) / 60
    print("Total Elapsed Time: %.2f minutes" % (elapsed))

    confusion_matrix = metrics.confusion_matrix(Y_test, y_predict_final_1)
    show_chart(clf_extra, confusion_matrix, X_test, Y_test)


def score_single_clf(clf_list, label, X_test, Y_test):
    clf = find_clf(clf_list, label)
    clf.fit(X_test, Y_test)
    result = clf.score(X_test, Y_test)
    print("{} accuracy: {}".format(label, result))
    return clf, clf.predict(X_test)


def find_clf(clf_list, label_to_find):
    for label, clf in clf_list:
        if label == label_to_find:
            return clf

    raise Exception('Could not find classifier \'{}\''.format(label_to_find))


def get_stacking():
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
    kwargs = {'criterion': 'gini', 'max_depth': 4.0, 'max_features': None, 'min_samples_leaf': 0.1, 'min_samples_split': 0.1, 'min_weight_fraction_leaf': 0.0, 'random_state': 42, 'splitter': 'best'}
    clf_tree_dec = ('DecisionTreeClassifier', tree.DecisionTreeClassifier(**kwargs))

    clf_tree_extra = (CLF_EXTRA_TREE, tree.ExtraTreeClassifier())

    # kwargs = dict(kernel="linear", C=0.025, cache_size=200, probability=True)
    kwargs = dict(C=10, gamma=0.001, kernel='rbf', random_state=RANDOM_SEED, probability=True)
    clf_svm = (CLF_SVC, svm.SVC(**kwargs))

    # kwargs = {'C': 100.0}
    kwargs = {'C': 100.0, 'dual': False, 'fit_intercept': True, 'multi_class': 'multinomial', 'penalty': 'l2', 'solver': 'saga'}
    clf_log_reg_100 = ('LogisticRegression_100', LogisticRegression(**kwargs))
    clf_log_reg_10k = ('LogisticRegression_10k', LogisticRegression(C=10000))

    # kwargs = {'n_neighbors': 3}
    kwargs = {'algorithm': 'auto', 'leaf_size': 20, 'metric': 'minkowski', 'n_jobs': 6, 'n_neighbors': 5, 'p': 2, 'weights': 'distance'}
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
    base_clfs = [clf_tree_dec, clf_tree_extra,
                     clf_log_reg_100, clf_kneighbors, clf_rand_forest_50, clf_rand_forest_10,
                     adaboost]

    lr = LogisticRegression()
    classifiers = [x[1] for x in base_clfs]
    stacking = ('StackingCVClassifier', StackingCVClassifier(classifiers=classifiers,
                                                             use_probas=True,
                                                             meta_classifier=lr))
    clf_stacking = stacking[1]

    clf_list = base_clfs.copy()
    clf_list.append(stacking)

    return clf_list, clf_stacking


def fit(clf_list, X_train, Y_train):
    start = time.time()

    for label, clf in clf_list:
        # scores = cross_val_score(clf, X_train, Y_train, cv=3, scoring='accuracy', n_jobs=12)

        clf.fit(X_train, Y_train)

        end = time.time()
        elapsed = (end - start) / 60

        # print("cross_val Accuracy: %.4f (+/- %.4f) [%s] (%.2f minutes) ..." % (
        #     scores.mean(), scores.std(), label, elapsed))
        print("Fit {}. ({:.2f} minutes)".format(label, elapsed))

        start = time.time()


def fit_cnn(X_train, X_test, Y_train, Y_test, epochs):
    X_train_cnn = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], 28, 28, 1)

    Y_train_cat = to_categorical(Y_train)
    Y_test_cat = to_categorical(Y_test)

    batches, val_batches = get_image_generator(X_train_cnn, Y_train_cat, X_test_cnn, Y_test_cat)

    clf_cnn = get_cnn(X_train)

    history = fit_generator(clf_cnn, batches, val_batches, epochs=epochs, callbacks=get_callbacks())

    return load_model(get_best_val_path()), X_test_cnn

def predict_proba_cnn(X_train, X_test, Y_train, Y_test, epochs):

    clf, X_test_cnn = fit_cnn(X_train, X_test, Y_train, Y_test, epochs)

    y_predict = clf.predict_proba(X_test_cnn)

    save_path = os.path.join(config.DATA_FOLDER_PATH, 'y_predict_cnn.csv')

    np.savetxt(save_path, y_predict, delimiter=",")

    return y_predict


def get_best_val_path():
    return os.path.join(config.SESSION_DIR, 'bestval.h5')


def get_callbacks():
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=.000001, patience=50, verbose=1, mode='auto',
                                   baseline=None,
                                   restore_best_weights=True)
    val_checkpoint = ModelCheckpoint(get_best_val_path(), 'val_loss', 1, save_best_only=True)
    return [early_stopping, ReduceLROnPlateau(patience=5, verbose=1, min_delta=.00001), val_checkpoint]


def get_image_generator(X_train, Y_train, X_test, Y_test):
    batch_size = 32

    gen = image.ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                                   height_shift_range=0.08, zoom_range=0.08)
    batches = gen.flow(X_train, Y_train, batch_size=batch_size)

    gen_val = image.ImageDataGenerator()
    val_batches = gen_val.flow(X_test, Y_test, batch_size=batch_size)

    return batches, val_batches


def fit_generator(model, batches, val_batches, epochs, callbacks):
    steps_per_epoch = batches.n
    history = model.fit_generator(generator=batches, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                  validation_data=val_batches, validation_steps=val_batches.n,
                                  callbacks=callbacks)

    return history


def show_chart(clf, confusion_matrix, X_test, Y_test):
    plt.figure(figsize=(9, 9))
    sns.heatmap(confusion_matrix, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(clf.score(X_test, Y_test))
    plt.title(all_sample_title, size=15)

    plt.show()


def get_cnn(X_train):
    import numpy as np
    mean_px = X_train.mean().astype(np.float32)
    std_px = X_train.std().astype(np.float32)

    def standardize(x):
        return (x - mean_px) / std_px

    def build_fn():
        clf = Sequential([
            Lambda(standardize, input_shape=(28, 28, 1)),
            Convolution2D(32, (3, 3), activation='relu'),
            BatchNormalization(axis=1),
            Convolution2D(32, (3, 3), activation='relu'),
            MaxPooling2D(),
            BatchNormalization(axis=1),
            Convolution2D(64, (3, 3), activation='relu'),
            BatchNormalization(axis=1),
            Convolution2D(64, (3, 3), activation='relu'),
            MaxPooling2D(),
            Flatten(),
            BatchNormalization(axis=1),
            Dense(512, activation='relu'),
            Dense(10, activation='softmax')
        ])

        clf.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        return clf

    return build_fn()


if __name__ == "__main__":
    # python -W ignore -m app.stacking.mnist
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('-sample_number', type=int, help='Your sample amount, enter it.')
    parser.add_argument('-epochs', type=int, help='Your epochs, enter it.')
    args = parser.parse_args()

    main(args.sample_number, args.epochs)
