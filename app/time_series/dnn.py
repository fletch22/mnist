import os

import matplotlib.pyplot as pyplot
from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense

from app.config import config


def learn(epochs, train_X, train_y, test_X, test_y):
    batch_size = 32
    input_dim = train_X.shape[1]

    #   https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
    # See https://johannfaouzi.github.io/pyts/user_guide.html#image for more.
    def create_baseline():
        model = Sequential()
        model.add(Dense(input_dim, input_dim=input_dim, activation='relu'))
        model.add(Dense(1500, input_dim=input_dim, activation='relu'))
        # model.add(Dense(1500, input_dim=input_dim, activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    model = create_baseline()

    # estimators = []
    # estimators.append(('standardize', StandardScaler()))
    # estimators.append(('model', model))
    # pipeline = Pipeline(estimators)

    history = model.fit(train_X, train_y,
                        batch_size=batch_size, epochs=epochs,
                        verbose=1, validation_data=(test_X, test_y), callbacks=get_callbacks())

    score = model.evaluate(test_X, test_y, verbose=1)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    pyplot.plot(history.history['acc'], label='train')
    pyplot.plot(history.history['val_acc'], label='test')
    pyplot.legend()
    pyplot.show()

def get_callbacks_adv(session_dir):
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=.000001, patience=8, verbose=1, mode='auto',
                                   baseline=None,
                                   restore_best_weights=True)
    val_checkpoint = ModelCheckpoint(get_best_val_path(session_dir), 'val_loss', 1, save_best_only=True)

    # reduce_args = dict(patience=5, verbose=1, min_delta=.00001)
    reduce_args = dict(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

    return [early_stopping, ReduceLROnPlateau(**reduce_args), val_checkpoint]


def get_best_val_path():
    return os.path.join(config.SESSION_DIR, 'best_basketball_model.h5')


def get_callbacks():
    return [ModelCheckpoint(get_best_val_path(), 'val_acc', 1, save_best_only=True)]


