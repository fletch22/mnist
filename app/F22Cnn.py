import os

from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.engine.saving import load_model
from keras.layers import Convolution2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_preprocessing import image
import numpy as np
import logging


class F22Cnn():

    def fit(self, session_dir=None, X_train=None, y_train=None, X_test=None, y_test=None, epochs=0, batch_size=0):
        mean_px = X_train.mean().astype(np.float32)
        std_px = X_train.std().astype(np.float32)

        def standardize(x):
            return (x - mean_px) / std_px

        num_classes = len(set(y_train))
        print('num_classes: {}'.format(num_classes))

        self.clf = Sequential([
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
            Dense(10, activation='softmax'),
        ])

        self.clf.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        self.batch_size = batch_size

        self.clf, history = fit_cnn(self.clf, X_train, y_train, X_test, y_test, epochs, session_dir, self.batch_size)

        return history

    def score(self, X, y):
        y_cat = to_categorical(y)
        logging.info("Metric names: {}".format(self.clf.metrics_names))
        return self.clf.evaluate(X, y_cat, batch_size=self.batch_size)[1]

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


def fit_cnn(clf, X_train, y_train, X_test, y_test, epochs, session_dir, batch_size):
    logging.info('Shape: {}'.format(X_train.shape))

    X_train_cnn = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], 28, 28, 1)

    Y_train_cat = to_categorical(y_train)
    Y_test_cat = to_categorical(y_test)

    batches, val_batches = get_image_generator(X_train_cnn, Y_train_cat, X_test_cnn, Y_test_cat, batch_size)

    history = fit_generator(clf, batches, val_batches, epochs=epochs, callbacks=get_callbacks(session_dir))

    return load_model(get_best_val_path(session_dir)), history


def get_best_val_path(session_dir):
    return os.path.join(session_dir, 'bestval.h5')


def get_image_generator(X_train, Y_train, X_test, Y_test, batch_size):
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


def get_callbacks(session_dir):
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=.000001, patience=50, verbose=1, mode='auto',
                                   baseline=None,
                                   restore_best_weights=True)
    val_checkpoint = ModelCheckpoint(get_best_val_path(session_dir), 'val_loss', 1, save_best_only=True)
    return [early_stopping, ReduceLROnPlateau(patience=5, verbose=1, min_delta=.00001), val_checkpoint]
