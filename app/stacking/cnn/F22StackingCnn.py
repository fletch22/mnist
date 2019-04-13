import inspect

from keras import Sequential
from keras.layers import Convolution2D, BatchNormalization, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
import logging

# logging.info('X shape: {}'.format(X.shape))
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# mean_px = X.mean().astype(np.float32)
# std_px = X.std().astype(np.float32)
#
# def standardize(x):
#     return (x - mean_px) / std_px
from sklearn.base import BaseEstimator, ClassifierMixin


class F22StackingCnn(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(self, batch_size=0, epochs=0):
        """
        Called when initializing the classifier
        """
        # self.batch_size = batch_size
        # self.epochs = epochs
        #
        # self.params = dict(batch_size=batch_size, epochs=epochs)

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        self.params = args

        self.params = {}
        for arg, val in values.items():
            self.params.update({arg: val})
            setattr(self, arg, val)

        self.clf = get_cnn(epochs, batch_size)

    def fit(self, X, y=None):
        X = X.reshape(-1, 28, 28, 1)
        return self.clf.fit(X, y)

    def get_params(self, deep=False):
        return self.params

    def predict(self, X):
        X = X.reshape(-1, 28, 28, 1)
        return self.clf.predict(X)

    def predict_proba(self, X, **kwargs):
        X = X.reshape(-1, 28, 28, 1)
        return self.clf.predict_proba(X, **kwargs)


def create_model():
    model = Sequential([
        # Lambda(standardize, input_shape=(28, 28, 1)),
        Convolution2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
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

    model.compile(Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def get_cnn(epochs=10, batch_size=32):
    return KerasClassifier(build_fn=create_model,
                           epochs=epochs,
                           batch_size=batch_size,
                           verbose=0)
