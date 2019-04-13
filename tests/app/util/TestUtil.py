import warnings

from app.grid_search.search import main_search
from app.stacking.mnist_quick import main_mnist_quick
from app.time_series import nba_first_crack

warnings.filterwarnings('ignore')

import logging
import os
from unittest import TestCase

import pandas as pd

from app import util
from app.config import config


class TestUtil(TestCase):

    def test_foo(self):
        # Arrange
        df = pd.read_csv(os.path.join(config.DATA_FOLDER_PATH, 'train.csv'))
        logging.debug(f'Cols: {df.columns}')

        df = df.sample(frac=1).reset_index(drop=True)

        classes = [1, 2, 3]

        # Act
        df_new = util.otherize_classes(df, classes=classes)

        unique = sorted(df_new.label.unique())
        logging.info(f'Unique: {unique}')

        # Assert
        assert (len(unique) == 8)

    def test_basketball(self):
        # Arrange
        # Act
        nba_first_crack.learn()

    def test_search(self):
        main_search()

    def test_mnist_quick(self):
        main_mnist_quick(2000, 10)

    def test_calc_profit(self):
        # Arrange
        # result = nba.calc_return(0.83, 100, 10)

        # Act
        # assert (result == .4153)

        result = nba_first_crack.calc_return(0.693, 100, 10)

        logging.info(f'Return {result}')
