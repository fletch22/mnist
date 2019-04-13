import warnings

from app.models.mnist_pretrained import get_vgg16_for_mnist

warnings.filterwarnings('ignore')

import os
from unittest import TestCase

from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

from app.config import config

dimension = 44
img_width, img_height = dimension, dimension

train_data_dir = os.path.join(config.DATA_FOLDER_PATH, 'mnist', 'train')
validation_data_dir = os.path.join(config.DATA_FOLDER_PATH, 'mnist', 'test')

train_samples = 60000

validation_samples = 10000

epoch = 30

batch_size = 32


class TestMnistFlow(TestCase):

    def get_model_simple(self):
        model = Sequential()
        model.add(Convolution2D(16, 5, 5, activation='relu', input_shape=(img_width, img_height, 3)))
        model.add(MaxPooling2D(2, 2))

        model.add(Convolution2D(32, 5, 5, activation='relu'))
        model.add(MaxPooling2D(2, 2))

        model.add(Flatten())
        model.add(Dense(1000, activation='relu'))

        model.add(Dense(10, activation='softmax'))
        # ** Model Ends **

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        return model

    def test_flow(self):
        # ** Model Begins **
        model = get_vgg16_for_mnist((dimension, dimension, 3), 10)
        # model = self.get_model_simple()

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')

        validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')

        model.fit_generator(
            train_generator,
            samples_per_epoch=train_generator.n,
            nb_epoch=epoch,
            validation_data=validation_generator,
            nb_val_samples=validation_samples, workers=12)

        # model.save_weights('mnistneuralnet.h5')
