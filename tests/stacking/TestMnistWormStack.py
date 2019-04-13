import logging
import os
import warnings
from unittest import TestCase

from app import util

warnings.filterwarnings('ignore')

import numpy as np
from sklearn import metrics

from app.config import config
from app.stacking import mnist_worm_stack
import pandas as pd


class TestMnistWormStack(TestCase):
    csv_path = os.path.join(config.DATA_FOLDER_PATH, 'confusion_matrix.csv')

    def test_load(self):
        # Arrange
        file_path = os.path.join(config.DATA_FOLDER_PATH, 'bestval_mws.h5')
        measurement_in_pixels = 28

        X, Y, X_train, X_test, Y_train, y_test = mnist_worm_stack.get_data()

        num_classes = len(set(Y_train))
        # Act
        model = mnist_worm_stack.load_trained_model(file_path, X_train, num_classes, measurement_in_pixels)

        # Assert
        X_reshaped = X_test.reshape(X_test.shape[0], measurement_in_pixels, measurement_in_pixels, 1)
        y_pred_raw = model.predict(X_reshaped)

        y_pred = np.argmax(y_pred_raw, axis=1)

        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

        logging.info(confusion_matrix)

        np.savetxt(self.csv_path, confusion_matrix, delimiter=',')

        X_test_cnn, y_test_cat = mnist_worm_stack.reshape_for_cnn(X_test, y_test)

        # mnist_worm_stack.show_chart(model, confusion_matrix, X_test_cnn, y_test_cat)

    def test_best_and_worst(self):
        # Arrange
        from numpy import genfromtxt

        my_data = genfromtxt(self.csv_path, delimiter=',')

        logging.debug(my_data)

        my_dataframe = pd.DataFrame(my_data)
        my_dataframe.columns = [str(x) for x in range(0, 10)]

        logging.debug(my_dataframe)

        # Act
        best, worst = mnist_worm_stack.get_best_and_worst(my_dataframe, 1.0)

        logging.debug(f'my_data: {my_data}')

        # Assert
        assert (best == [0, 1, 7])

    def test_best_and_worst_from_numpy_array(self):
        # Arrange
        from numpy import genfromtxt

        my_data = genfromtxt(self.csv_path, delimiter=',')

        logging.debug(my_data)

        # Act
        best, worst = mnist_worm_stack.get_best_and_worst_from_confusion_matrix(my_data)

        logging.debug(f'my_data: {my_data}')

        # Assert
        assert (best == [0, 1, 7])

    def test_one_hot_encode_labels(self):
        # Arrange
        y_labels = np.array([0, 1, 1, 2, 3, 9])

        y_unique = np.array(set(y_labels))
        logging.debug(f'y_unique: {y_unique}')

        # Act
        encoded = mnist_worm_stack.one_hot_code_y_labels(y_labels)

        logging.debug(f'Encoded: {encoded}')

        # Assert
        assert (len(encoded) == 6)

    def test_load_reform_model(self):
        # Arrange
        file_path = os.path.join(config.DATA_FOLDER_PATH, 'bestval_mws.h5')
        measurement_in_pixels = 28

        X, Y, X_train, X_test, Y_train, y_test = mnist_worm_stack.get_data()

        num_classes = len(set(Y_train))

        model = mnist_worm_stack.load_trained_model(file_path, X_train, num_classes, measurement_in_pixels)

        # Act
        mnist_worm_stack.sub_prediction_layer(model, 3)

    def test_image_conversion(self):
        # Arrange
        from PIL import Image

        X, y = util.get_mnist(1)

        X = X.astype('float32')
        # image_arr = X[0].reshape(1,-1)

        image_arr = X[0]

        logging.info(f'Shape: {image_arr.shape}')

        # im = Image.fromarray(np.uint8(X[0]))

        measurement_in_pixels = 28

        image_arr = image_arr.reshape(measurement_in_pixels, measurement_in_pixels)

        logging.info(f'Type: {type(image_arr)}')

        # image_path = os.path.join(config.DATA_FOLDER_PATH, 'sample.PNG')
        # im = Image.open(image_path)
        # image_arr = np.asarray(im)

        logging.info(f'a shape:{image_arr.shape}')

        image = Image.fromarray(np.uint8(image_arr), mode='L')

        # import PIL
        # import numpy
        # from PIL import Image
        #
        # def resize_image(numpy_array_image, new_height):
        #     # convert nympy array image to PIL.Image
        #     image = Image.fromarray(numpy.uint8(numpy_array_image))
        #     old_width = float(image.size[0])
        #     old_height = float(image.size[1])
        #     ratio = float(new_height / old_height)
        #     new_width = int(old_width * ratio)
        #     image = image.resize((new_width, new_height), PIL.Image.ANTIALIAS)
        #     # convert PIL.Image into nympy array back again
        #     return array(image)

