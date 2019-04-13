import os
import warnings
from random import randint

import keras
import sklearn
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.engine.saving import load_model
from sklearn import metrics

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
import time
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import pandas as pd

from keras import Sequential, Model
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


def get_otherize_complement(otherized_digits):
    keep_digits = []
    for i in range(0, 10):
        if i not in otherized_digits:
            keep_digits.append(i)

    return keep_digits


def get_best_and_worst_from_confusion_matrix(confusion_matrix, otherize_digits=[], threshold_accuracy=1.0):
    keep_digits = get_otherize_complement(otherize_digits)

    anchor_array = [otherize_digits[0]] if len(otherize_digits) > 0 else []
    keep_digits = sorted(anchor_array + keep_digits)
    logging.info(f'Will use these digits to keep: {keep_digits}')

    df = pd.DataFrame(confusion_matrix, columns=[str(x) for x in keep_digits])

    anchor_digit = otherize_digits[0] if len(otherize_digits) > 0 else None

    return get_best_and_worst(df, anchor_digit, threshold_accuracy)


def get_best_and_worst(df, anchor_digit=None, threshold_accuracy=1.0):
    worst_unordered = []
    best = []
    column_index = 0
    for col_name in df.columns:
        col_id = int(col_name)
        if anchor_digit is None or col_id != anchor_digit:
            logging.info(f'Col name: {col_name}')

            col_all = df[col_name]
            logging.info(f'col_all: {col_all}')

            total = col_all.sum()

            logging.info(f'total: {total}')

            coll_all_array = col_all.values
            col = np.delete(coll_all_array, [column_index], axis=0)

            logging.info(col)

            accuracy = 1.0
            if total > 0:
                logging.info(f'total: {total}; Sum: {col.sum()}')
                accuracy = 1 - (float(col.sum()) / float(total))

            logging.info(f'col_name {col_name} accuracy: {accuracy}')

            if accuracy < threshold_accuracy:
                worst_unordered.append((col_id, accuracy))
            else:
                best.append(col_id)

        column_index += 1

    def sort_tuple(item):
        return item[1]

    worst_tuples = sorted(worst_unordered, key=sort_tuple)

    worst = [x[0] for x in worst_tuples]

    return best, worst


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


def filter_label_vector(label_vector, otherize_digits=[]):
    final_vector = label_vector
    if len(otherize_digits) > 0:
        anchor_digit = otherize_digits[0]

        def map(x):
            result = x
            if x in otherize_digits:
                return anchor_digit
            return result

        vf = np.vectorize(map)

        final_vector = vf(label_vector)

    return final_vector


def get_train(limit):
    return get_mnist("../input/train.csv", limit)


def get_test():
    df = pd.read_csv("../input/test.csv")

    return df.values / 255.0


def get_data(limit=None):
    if RUNNING_LOCAL is True:
        from app import util
        X, y = util.get_mnist(limit=limit)
    else:
        X, y = get_train(limit)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, train_size=0.7)

    return X, y, X_train, X_test, y_train, y_test


def show_chart(clf, confusion_matrix, X_test, Y_test):
    plt.figure(figsize=(9, 9))
    sns.heatmap(confusion_matrix, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(clf.score(X_test, Y_test))
    plt.title(all_sample_title, size=15)

    plt.show()


def load_trained_model(weights_path, X, num_classes):
    model = create_model(X, num_classes)
    model.load_weights(weights_path)

    return model


def create_model(X, num_classes):
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
        Dense(num_classes, name='layer_softmax', activation='softmax'),
    ])

    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_cnn(epochs=10, batch_size=32):
    return KerasClassifier(build_fn=create_model,
                           epochs=epochs,
                           batch_size=batch_size,
                           verbose=0)


def get_best_val_path(session_dir):
    return os.path.join(session_dir, 'bestval_mws.h5')


def fit_generator(model, batches, val_batches, epochs, callbacks):
    steps_per_epoch = batches.n
    history = model.fit_generator(generator=batches, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                  validation_data=val_batches, validation_steps=val_batches.n,
                                  callbacks=callbacks)

    return history


def reshape_for_cnn(X, y):
    num_classes = len(set(y))

    logging.debug(f'Will reshape for y: {y}')
    logging.debug(f'Has num classes: {num_classes}')

    return X.reshape(X.shape[0], 28, 28, 1), one_hot_code_y_labels(y)


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


def create_model_from_otherize_digits(X, y, otherize_digits):
    y = filter_label_vector(y, otherize_digits)

    n_classes = len(set(y))
    logging.info('Number of classes {}.'.format(n_classes))

    return create_model(X, n_classes), y


# def create_model_and_data(sample_number, otherize_digits):
#     _X, _Y, X_train, X_test, y_train, y_test = get_data(limit=sample_number)
#
#     y_test = filter_label_vector(y_test, otherize_digits)
#
#     n_classes = len(set(y_train))
#     logging.info('Number of classes {}.'.format(n_classes))
#
#     return create_model(X_train, n_classes), X_train, X_test, y_train, y_test

def sub_prediction_layer(clf, num_classes):
    pass
    # create model
    # vgg = keras.applications.vgg19.VGG19(include_top=True, weights='imagenet')

    clf.summary()

    clf.layers.pop()
    clf.layers[-1].outbound_nodes = []
    clf.outputs = [clf.layers[-1].output]

    output = clf.get_layer('activation_5').output
    output = Dense(num_classes, name='layer_softmax', activation='softmax')(output)
    new_model = Model(clf.input, output)

    new_model.summary()

    # output = clf.get_layer('avg_pool').output

    # output = Flatten()(output)
    # output = Dense(output_dim=10000, activation='relu')(output)  # your newlayer Dense(...)
    # new_model = Model(model.input, output)

    # new_model = keras.models.Model(clf.get_input_at(0), clf.layers[-2].get_output_at(0))
    # new_model.set_weights(clf.get_weights())
    #
    # dense_last = Dense(num_classes, activation='softmax')
    # last_model = keras.models.Model(input=new_model.get_input_at(0), output=dense_last)
    # last_model.set_weights(new_model.get_weights())
    #
    # last_model.summary()

    # remove predictions layer
    # vgg = keras.models.Model(inputs=vgg.get_input_at(0), outputs=vgg.layers[-2].get_output_at(0))

def main(sample_number, epochs):
    otherize_digits = []  # 1, 3, 6, 7, 9
    batch_size = 86
    num_cnns_to_try = 10
    logging.getLogger().setLevel(logging.INFO)

    total_start = time.time()
    start = total_start

    cnn_collection = []

    _, _, X, X_holdout, y, y_holdout = get_data(limit=sample_number)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

    for i in range(0, num_cnns_to_try):
        model, y_train = create_model_from_otherize_digits(X_train, y_train, otherize_digits)
        y_test = filter_label_vector(y_test, otherize_digits)

        clf, history = fit_cnn(model, X_train, y_train, X_test, y_test, epochs, SESSION_DIR,
                               batch_size)

        anchor_digit = [otherize_digits[0]] if len(otherize_digits) > 0 else []
        scored_digits = sorted(anchor_digit + get_otherize_complement(otherize_digits))

        logging.debug(f'scored_digits: {scored_digits}')
        # raise Exception('Index of otherize_digits[0] should be inserted into otherize_complement.')

        y_pred = get_pred_from_collection([clf], scored_digits, X_test, y_test)
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

        logging.debug(f'y_test: c{len(set(y_test))}')
        logging.info(f'Confusion matrix: {confusion_matrix}')

        best, worst = get_best_and_worst_from_confusion_matrix(confusion_matrix, otherize_digits)

        sample_multiplier = 0.4
        if len(best) > 0:
            cnn_collection.append((best, clf))
            # save_model(clf)
            # epochs = epochs - 1 if epochs > 1 else 1
            # sample_number -= int(sample_number * sample_multiplier)
            logging.info(f'Removing digits: {best}')
        else:
            epochs += 2
            sample_number += int(sample_number * sample_multiplier)

        otherize_digits = otherize_digits + best

        logging.info(f'Epochs: {epochs}')

        end = time.time()
        elapsed = (end - start) / 60
        logging.debug("CNN Elapsed Time: :{:.2f} minutes: ".format(elapsed))

        if len(otherize_digits) > 8:
            logging.info('Found 100% for all matches.')
            break

    # Add column focus for each cnn as tuple (?)

    # y_pred = get_pred_from_collection(cnn_collection, X_test, y_test)

    # confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    # X_test_cnn, y_test_cat = reshape_for_cnn(X_test, y_test)

    total_end = time.time()
    elapsed = (total_end - total_start) / 60
    logging.info("Total Elapsed Time: :{:.2f} minutes: ".format(elapsed))

    # predict results
    if RUNNING_LOCAL is False:
        X_test = get_test(otherize_digits)

        y_pred = get_pred_from_collection(cnn_collection, X_test, None)

        logging.info("Results {}:".format(y_pred))

        results = pd.Series(y_pred, name="Label")

        logging.info("Shape Result: {}".format(results.shape))
        submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)

        logging.info("Shape Submission: {}".format(submission.shape))
        # submission.to_csv("cnn_mnist_datagen.csv",index=False)


def one_hot_code_y_labels(y_labels):
    lb = sklearn.preprocessing.LabelBinarizer()
    lb.fit(y_labels)
    return lb.transform(y_labels)


def get_pred_from_collection(cnn_collection, column_indices, X, y_test=None):
    X_test_pred = X.reshape(X.shape[0], 28, 28, 1)

    y_test_cat = None
    if y_test is not None:
        y_test_cat = one_hot_code_y_labels(y_test)

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

    logging.info('meta: {}'.format(meta))
    logging.info('meta shape: {}'.format(meta.shape))
    logging.info('mc: {}'.format(meta[0:2]))
    logging.info(f'ci: {column_indices}')

    y_pred_raw = np.argmax(meta, axis=1)

    def map_col(x):
        return int(column_indices[x])

    map_col_vectorize = np.vectorize(map_col)
    y_pred = map_col_vectorize(y_pred_raw)

    logging.debug(f'y_pred_raw: {y_pred_raw}')
    logging.debug(f'y_pred: {y_pred}')

    if y_test is not None:
        score = np.mean(y_test == y_pred)
        logging.info('Score: {:.4f}%'.format(score * 100))

    return y_pred


def get_mean_of_cols(array):
    return np.mean(array, 0)


if RUNNING_LOCAL is False:
    main(50000, 3)
else:
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Say hello')
        parser.add_argument('-sample_number', type=int, help='Your sample amount, enter it.')
        parser.add_argument('-epochs', type=int, help='Your epochs, enter it.')
        args = parser.parse_args()

        main(args.sample_number, args.epochs)
