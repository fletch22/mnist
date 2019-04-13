from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.datasets import mnist
from keras.regularizers import l1_l2

from app import util

def build_logistic_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim, activation='softmax'))

    return model


def score(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def main():
    batch_size = 128
    nb_classes = 10
    epochs = 100
    input_dim = 784

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(60000, input_dim)
    X_test = X_test.reshape(10000, input_dim)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print(X_train.shape, 'train samples')
    print(X_test.shape, 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = build_logistic_model(input_dim, nb_classes)
    # compile the model
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, Y_train,
                        batch_size=batch_size, epochs=epochs,
                        verbose=0, validation_data=(X_test, Y_test))
    score(model, X_test, Y_test)

    # 2-class logistic regression in Keras
    model = build_logistic_model(input_dim, nb_classes)
    # compile the model
    model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, epochs=epochs, verbose=0, validation_data=(X_test, Y_test))
    score(model, X_test, Y_test)

    # logistic regression with L1 and L2 regularization
    reg = l1_l2(l1=0.01, l2=0.01)

    # compile the model
    model = Sequential()
    model.add(Dense(nb_classes, input_dim=input_dim, activation='sigmoid', kernel_regularizer=reg))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, epochs=epochs, verbose=0, validation_data=(X_test, Y_test))
    score(model, X_test, Y_test)


main()