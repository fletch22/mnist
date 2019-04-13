import os
import warnings
from random import randint

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.engine.saving import load_model
from keras.utils import to_categorical

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
import time
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import pandas as pd

from keras import Sequential
from keras.layers import Convolution2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Lambda, Activation
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from keras_preprocessing import image

import numpy as np
import logging

RUNNING_LOCAL = True

if RUNNING_LOCAL is True:
    from app.config.config import SESSION_DIR
else:
    SESSION_DIR = ''
    logging.info = print

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


def get_image_generator(X_train, Y_train, X_test, Y_test, batch_size):
    # gen_args = dict(rotation_range=10, width_shift_range=0.1, shear_range=0.3,
    #                                height_shift_range=0.1, zoom_range=0.1)

    # gen_args = dict(rotation_range=8, width_shift_range=0.08, shear_range=0.08,
    #                 height_shift_range=0.08, zoom_range=0.08)

    gen_args = dict(featurewise_center=False,  # set input mean to 0 over the dataset
                    samplewise_center=False,  # set each sample mean to 0
                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                    samplewise_std_normalization=False,  # divide each input by its std
                    zca_whitening=False,  # apply ZCA whitening
                    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                    zoom_range=0.1,  # Randomly zoom image
                    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                    horizontal_flip=False,  # randomly flip images
                    vertical_flip=False)

    gen = image.ImageDataGenerator(**gen_args)
    batches = gen.flow(X_train, Y_train, batch_size=batch_size)

    gen_val = image.ImageDataGenerator()
    val_batches = gen_val.flow(X_test, Y_test, batch_size=batch_size)

    return batches, val_batches


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


def get_data(limit=None):
    if RUNNING_LOCAL is True:
        from app import util
        X, Y = util.get_mnist(limit=limit, otherize_digits=[])
    else:
        X, Y = get_train(limit)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=shuffle_sample_data, train_size=0.7)

    return X, Y, X_train, X_test, Y_train, Y_test


def show_chart(clf, confusion_matrix, X_test, Y_test):
    plt.figure(figsize=(9, 9))
    sns.heatmap(confusion_matrix, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(clf.score(X_test, Y_test))
    plt.title(all_sample_title, size=15)

    plt.show()


def create_model(X):
    mean_px = X.mean().astype(np.float32)
    std_px = X.std().astype(np.float32)

    def standardize(x):
        return (x - mean_px) / std_px

    model = Sequential([
        Lambda(standardize, input_shape=(28, 28, 1)),

        Convolution2D(32, (3, 3), input_shape=(28, 28, 1)),
        BatchNormalization(axis=-1),
        Activation('relu'),

        Convolution2D(32, (3, 3)),
        BatchNormalization(axis=-1),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Convolution2D(64, (3, 3)),
        BatchNormalization(axis=-1),
        Activation('relu'),

        Convolution2D(64, (3, 3)),
        BatchNormalization(axis=-1),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(512),
        BatchNormalization(),
        Activation('relu'),
        # Dropout(0.2),
        Dense(10, activation='softmax'),
    ])

    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_cnn(epochs=10, batch_size=32):
    return KerasClassifier(build_fn=create_model,
                           epochs=epochs,
                           batch_size=batch_size,
                           verbose=0)


def get_best_val_path(session_dir):
    return os.path.join(session_dir, 'bestval.h5')


def fit_generator(model, batches, val_batches, epochs, callbacks):
    steps_per_epoch = batches.n
    history = model.fit_generator(generator=batches, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                  validation_data=val_batches, validation_steps=val_batches.n,
                                  callbacks=callbacks)

    return history


def reshape_for_cnn(X, y):
    return X.reshape(X.shape[0], 28, 28, 1), to_categorical(y)


def fit_cnn(clf, X_train, y_train, X_test, y_test, epochs, session_dir, batch_size):
    logging.debug('Shape: {}'.format(X_train.shape))

    X_train_cnn, y_train_cat = reshape_for_cnn(X_train, y_train)
    X_test_cnn, y_test_cat = reshape_for_cnn(X_test, y_test)

    batches, val_batches = get_image_generator(X_train_cnn, y_train_cat, X_test_cnn, y_test_cat, batch_size)

    history = fit_generator(clf, batches, val_batches, epochs=epochs, callbacks=get_callbacks(session_dir))

    return load_model(get_best_val_path(session_dir)), history


def get_callbacks(session_dir):
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=.000001, patience=8, verbose=1, mode='auto',
                                   baseline=None,
                                   restore_best_weights=True)
    val_checkpoint = ModelCheckpoint(get_best_val_path(session_dir), 'val_loss', 1, save_best_only=True)

    # reduce_args = dict(patience=5, verbose=1, min_delta=.00001)
    reduce_args = dict(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

    return [early_stopping, ReduceLROnPlateau(**reduce_args), val_checkpoint]


def save_model(model):
    base_path = os.path.join(SESSION_DIR, 'clf-{}'.format(randint(100, 999)))

    model_json = model.to_json()
    with open("{}.json".format(base_path), "w") as json_file:
        json_file.write(model_json)

    # Serialize weights to HDF5
    save_path = "{}.h5".format(base_path)
    model.save_weights(save_path)
    print("Saved model \'{}\' to disk".format(save_path))


def score_model(model, X, y):
    score = model.evaluate(X, y, verbose=0)
    # print('Test score:', score[0])
    logging.info('Test accuracy: {}'.format(score[1]))


def show_plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def main(sample_number, epochs):
    batch_size = 86
    num_cnns_to_try = 10
    logging.getLogger().setLevel(logging.INFO)

    total_start = time.time()

    _X, _Y, X_train, X_test, y_train, y_test = get_data(limit=sample_number)

    n_classes = len(set(y_test))
    logging.debug('Number of classes {}.'.format(n_classes))

    start = total_start

    cnn_collection = []
    for i in range(0, num_cnns_to_try):
        clf, history = fit_cnn(create_model(X_train), X_train, y_train, X_test, y_test, epochs, SESSION_DIR, batch_size)

        cnn_collection.append(clf)
        # save_model(clf)

        # show_plot_loss(history)

        end = time.time()
        elapsed = (end - start) / 60
        logging.debug("CNN Elapsed Time: :{:.2f} minutes: ".format(elapsed))

    get_pred_from_collection(cnn_collection, X_test, y_test)

    total_end = time.time()
    elapsed = (total_end - total_start) / 60
    logging.info("Total Elapsed Time: :{:.2f} minutes: ".format(elapsed))

    # predict results
    if RUNNING_LOCAL is False:
        X_test = get_test()

        y_pred = get_pred_from_collection(cnn_collection, X_test, None)

        logging.info("Results {}:".format(y_pred))

        results = pd.Series(y_pred, name="Label")

        logging.info("Shape Result: {}".format(results.shape))
        submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)

        logging.info("Shape Submission: {}".format(submission.shape))
        # submission.to_csv("cnn_mnist_datagen.csv",index=False)


def get_pred_from_collection(cnn_collection, X, y_test=None):
    X_test_pred = X.reshape(X.shape[0], 28, 28, 1)

    y_test_cat = None
    if y_test is not None:
        y_test_cat = keras.utils.np_utils.to_categorical(y_test)

    results_combined = []
    for cnn in cnn_collection:
        results = cnn.predict(X_test_pred)

        results_combined.append(results)

        if y_test_cat is not None:
            score_model(cnn, X_test_pred, y_test_cat)

    if len(results_combined) > 1:
        meta = np.mean(np.array(results_combined), axis=0)
    else:
        meta = results_combined[0]

    logging.debug('meta: {}'.format(meta))
    logging.debug('meta shape: {}'.format(meta.shape))
    logging.debug('mc: {}'.format(meta[0:2]))

    y_pred = np.argmax(meta, axis=1)

    logging.debug('y_pred: {}'.format(y_pred))
    logging.debug('argedmax: {}'.format(y_pred[0: 2]))

    if y_test is not None:
        score = np.mean(y_test == y_pred)
        logging.info('Score: {:.4f}%'.format(score * 100))

    return y_pred


def get_mean_of_cols(array):
    return np.mean(array, 0)


if RUNNING_LOCAL is False:
    main(10000, 3)
else:
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Say hello')
        parser.add_argument('-sample_number', type=int, help='Your sample amount, enter it.')
        parser.add_argument('-epochs', type=int, help='Your epochs, enter it.')
        args = parser.parse_args()

        main(args.sample_number, args.epochs)
