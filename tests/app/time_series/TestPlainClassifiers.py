from unittest import TestCase

from app.time_series import plain_classifiers
import logging

import numpy as np

class TestPlainClassifiers(TestCase):

    def test_score_by_average_over_thresh(self):
        # Arrange
        arr1 = np.array([[.9, .1], [.8, .2], [.7, .3], [.9, .1]])
        arr2 = np.array([[.8, .2], [.1, .9], [.2, .8], [.8, .2]])
        y = [False, True, True, True]

        all_probas = [arr1, arr2]

        # Act
        accuracy = plain_classifiers.score_by_average_over_thresh(all_probas, y)

        logging.info(f'Acc: {accuracy}')

        # Assert
        assert accuracy == .75

    def test_score_by_highest_of_all(self):
        # Arrange
        arr1 = np.array([[.9, .1], [.8, .2], [.7, .3], [.9, .1]])
        arr2 = np.array([[.2, .8], [.1, .9], [.2, .8], [.8, .2]])
        y = [False, True, True, True]

        all_probas = [arr1, arr2]

        # Act
        accuracy = plain_classifiers.score_by_highest_of_all(all_probas, y)

        logging.info(f'Acc: {accuracy}')

        # Assert
        assert accuracy == .75

    def test_score_by_drop_low(self):
        # Arrange
        arr1 = np.array([[.9, .1], [.2, .9], [.7, .3], [.9, .1]])
        arr2 = np.array([[.6, .4], [.1, .9], [.2, .8], [.8, .2]])
        y = [False, True, True, True]

        all_probas = [arr1, arr2]

        # Act
        accuracy = plain_classifiers.score_by_drop_low(all_probas, y)

        logging.info(f'Acc: {accuracy}')

        # Assert
        assert accuracy == 1.0

    def test_score_by_majority_vote(self):
        # Arrange
        arr1 = np.array([[.9, .1], [.2, .9], [.7, .3], [.9, .1]])
        arr2 = np.array([[.6, .4], [.1, .9], [.2, .8], [.8, .2]])
        arr2 = np.array([[.6, .4], [.9, .1], [.1, .9], [.8, .2]])
        y = [False, True, True, False]

        all_probas = [arr1, arr2]

        # Act
        accuracy = plain_classifiers.score_by_majority_vote(all_probas, y)

        # Assert
        assert accuracy == 1.0