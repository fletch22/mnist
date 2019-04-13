import logging
from unittest import TestCase
import numpy as np

from app.stacking.mnist_man_stack import get_mean_of_cols

class TestMnistManStack(TestCase):

    def test_meaning(self):
        # Arrange
        arr_1 = np.array([[1, 2, 3], [4, 5, 6]])
        arr_2 = np.array([[6, 7, 8], [9, 10, 11]])

        # arr_1 = arr_1.reshape(arr_1.shape[0], -1)
        # arr_2 = arr_2.reshape(arr_2.shape[0], -1)

        logging.info('Arr_1 shape: {}:'.format(arr_1.shape))
        logging.info('Arr_2 shape: {}:'.format(arr_2.shape))

        # arr_c = np.concatenate([arr_1, arr_2], axis=0)

        # arr_c = np.append(arr_1, arr_2, axis=0)

        # arr_c = np.column_stack((arr_1, arr_2))

        # arr_c = np.vstack((arr_1, arr_2))

        combined = [arr_1, arr_2]
        arr_c = np.mean(np.array(combined), axis=0)

        logging.info('Combined: {}:'.format(arr_c))

        # a = np.array([1, 2, 3])
        # b = np.array([2, 3, 4])
        # logging.info(np.vstack((a, b)))

        # Act
        result = get_mean_of_cols(arr_c)

        # Assert
        logging.info(result)

    def test_fix_arr(self):
        # Arrange
        list_thing = [1, 2, 3]

        arr = np.array(list_thing)

        logging.info(arr.shape)

        arr = arr.reshape(arr.shape[0], -1)

        logging.info(arr.shape)

    def test_get_max_column(self):
        # Arrange
        list_thing = [[0.04003451, 0.09324495, 0.15717622, 0.15460602, 0.09149451, 0.05787251, 0.09058939, 0.0798719, 0.12287291, 0.11223709],
                      [1.04003451, 0.09324495, 0.15717622, 0.15460602, 0.09149451, 0.05787251, 0.09058939, 0.0798719,
                       0.12287291, 0.11223709]]

        arr = np.array(list_thing)

        # Act
        max_indices = np.argmax(arr, axis=1)

        logging.info(max_indices)

        # Assert
        # assert(max == 2)
