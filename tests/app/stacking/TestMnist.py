import os
from unittest import TestCase
import numpy as np

from app.stacking import mnist
from app.config import config


class TestMnist(TestCase):

    def test_array_blending(self):
        # Arrange
        data_path = os.path.join(config.DATA_FOLDER_PATH, 'y_predict_cnn.csv')

        # Act
        arr_1 = np.loadtxt(data_path, delimiter=",")

        arr_2 = arr_1.copy()


        arr_3 = np.append(arr_1, arr_2, 1)

        print(arr_1.shape)
        print(arr_3.shape)

        print(arr_1[0])
        print(arr_2[0])
        print(arr_3[0])

        from app import bayes

        bayes.main()
