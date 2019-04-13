import warnings

from app.models.mnist_pretrained import get_vgg16_for_mnist

warnings.filterwarnings('ignore')

import logging
import os
from unittest import TestCase

import cv2
import numpy as np
from keras import backend as K, Sequential
from keras.applications.vgg16 import VGG16
from keras.datasets import mnist
from keras.layers import Input, Flatten, Dense, Convolution2D, MaxPooling2D
from keras.models import Model
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from app.config import config
from app.services import file_services
from app.services.image_services import get_flow_from_directory, save_to_directory

class TestPretrainedCnn(TestCase):

    def test_resize_from_array(self):
        # Get back the convolutional part of a VGG network trained on ImageNet
        img_dim_ordering = 'tf'
        K.set_image_dim_ordering(img_dim_ordering)

        # loading the data
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        image_size = 44

        # converting it to RGB
        x_train = [cv2.cvtColor(cv2.resize(i, (image_size, image_size)), cv2.COLOR_GRAY2BGR) for i in x_train]
        x_train = np.concatenate([arr[np.newaxis] for arr in x_train]).astype('float32')

        x_test = [cv2.cvtColor(cv2.resize(i, (image_size, image_size)), cv2.COLOR_GRAY2BGR) for i in x_test]
        x_test = np.concatenate([arr[np.newaxis] for arr in x_test]).astype('float32')

        x_train = x_train / 255.
        x_test = x_test / 255.

        # x_train = preprocess_input(x_train)
        # x_test = preprocess_input(x_test)

        train_Y_one_hot = to_categorical(y_train)
        test_Y_one_hot = to_categorical(y_test)

        train_X, valid_X, train_label, valid_label = train_test_split(x_train,
                                                                      train_Y_one_hot,
                                                                      test_size=0.2,
                                                                      random_state=13
                                                                      )
        print(f'Shape: {train_X.shape}')

        # (x_train, y_train), (x_test, y_test) = mnist.load_data()
        num_channels = 3 if len(train_X.shape) > 3 else 1
        train_X = train_X.reshape(train_X.shape[0], num_channels, image_size, image_size)

        datagen = get_image_gen()
        # fit parameters from datai
        # datagen.fit(train_X)
        # configure batch size and retrieve one batch of images
        # show_images(datagen, image_size, train_X, train_label, num_channels)
        # raise Exception('Bar')

        logging.debug(f'train_X.shape[1:]: {train_X.shape[1:]}')

        train_X = train_X.reshape(train_X.shape[0], image_size, image_size, 3)

        # training the model
        model = get_vgg16_for_mnist(train_X.shape[1:], len(set(y_train)))
        hist = model.fit(train_X, train_label, epochs=10, validation_data=(valid_X, valid_label), verbose=1)

        # hist = self.fit_generator(model, train_X, train_label, 10, validation_data=(valid_X, valid_label), verbose=1)
        # history = model.fit_generator(generator=batches, steps_per_epoch=steps_per_epoch, epochs=epochs,
        #                               validation_data=val_batches, validation_steps=val_batches.n,
        #                               callbacks=callbacks)

    def test_flow_from_directory(self):
        # Get back the convolutional part of a VGG network trained on ImageNet
        # img_dim_ordering = 'tf'
        # K.set_image_dim_ordering(img_dim_ordering)

        dimension = 28
        img_height = dimension
        img_width = dimension
        batch_size = 32

        train_gen, val_gen, train_size, test_size = get_flow_from_directory(img_height, img_width, batch_size)

        # model = pretrained_model((img_height, img_width, 3), 10)
        model = self.get_simple_model(img_height)

        model.fit_generator(
            train_gen,
            steps_per_epoch=train_gen.n,
            epochs=50,
            validation_data=val_gen,
            validation_steps=val_gen.n,
            verbose=1)

    def get_simple_model(self, dimension):
        model = Sequential()
        model.add(Convolution2D(16, 5, 5, activation='relu', input_shape=(dimension, dimension, 3)))
        model.add(MaxPooling2D(2, 2))

        model.add(Convolution2D(32, 5, 5, activation='relu'))
        model.add(MaxPooling2D(2, 2))

        model.add(Flatten())
        model.add(Dense(1000, activation='relu'))

        model.add(Dense(10, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        return model

    def test_save_to_directory(self):
        dimension = 28
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        save_to_directory(X_train, y_train, dimension, 'train')
        save_to_directory(X_test, y_test, dimension, 'test')


def load_up_train_test():
    test_ratio = 0.2
    mnist_path = os.path.join(config.DATA_FOLDER_PATH, 'mnist')
    file_paths = file_services.get_files(mnist_path)

    np.random.shuffle(file_paths)

    num_test_files = int(len(file_paths) * test_ratio)

    test_files = file_paths[0: num_test_files]
    train_files = file_paths[num_test_files:]

    file_services.copy_file(test_files, os.path.join(config.DATA_FOLDER_PATH, 'mnist', 'test'))
    file_services.copy_file(train_files, os.path.join(config.DATA_FOLDER_PATH, 'mnist', 'train'))

    return file_paths


def get_image_gen():
    gen_args = dict(featurewise_center=False,  # set input mean to 0 over the dataset
                    samplewise_center=False,  # set each sample mean to 0
                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                    samplewise_std_normalization=False,  # divide each input by its std
                    zca_whitening=False,  # apply ZCA whitening
                    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                    zoom_range=0.30,  # Randomly zoom image
                    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                    horizontal_flip=False,  # randomly flip images
                    vertical_flip=False)

    # gen_args = dict(featurewise_center=True, featurewise_std_normalization=True)

    return ImageDataGenerator(**gen_args)
