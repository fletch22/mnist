import logging
from datetime import datetime, timedelta
from unittest import TestCase

from app.basketball.nba.big_data_ball import big_data_util
import pandas as pd
import numpy as np

from app.basketball.nba.big_data_ball.big_data_util import COLUMN_SL_1, COLUMN_SL_2, COLUMN_SL_3, COLUMN_SL_4, COLUMN_SL_5

team_map = big_data_util.team_map


class TestBigDataUtil(TestCase):

    def test_distance_calculator_1(self):
        # Arrange
        city_1 = 'Cleveland Cavaliers'
        city_2 = 'Chicago Bulls'

        # Act
        distance = big_data_util.calculate_distance(city_1, city_2)

        # Assert
        assert round(distance, 1) == 495.2

    def test_distance_calculator_2(self):
        # Arrange
        city_2 = 'Washington Wizards'
        city_1 = 'Boston Celtics'

        # Act
        distance = big_data_util.calculate_distance(city_1, city_2)

        # Assert
        assert round(distance, 1) == 2582.2

    def test_time_zone_diff_1(self):
        # Arrange
        city_2 = 'Washington Wizards'
        city_1 = 'Boston Celtics'

        # Act
        diff = big_data_util.get_timezone_diff(city_1, city_2)

        # Assert
        assert diff == 0

    def test_time_zone_diff_2(self):
        # Arrange
        origin = 'Seattle SuperSonics'
        destination = 'Boston Celtics'

        # Act
        diff = big_data_util.get_timezone_diff(origin, destination)

        # Assert
        assert diff == 3

    def test_time_zone_diff_3(self):
        # Arrange
        origin = 'Boston Celtics'
        destination = 'Seattle SuperSonics'

        # Act
        diff = big_data_util.get_timezone_diff(origin, destination)

        # Assert
        assert diff == -3

    def test_delete_games_not_matching(self):
        # Arrange
        t1_g1 = datetime.now()
        t1_g2 = t1_g1 + timedelta(days=1)
        t1_g3 = t1_g2 + timedelta(days=1)
        t1_games = [t1_g1.timestamp(), t1_g2.timestamp(), t1_g3.timestamp()]

        t2_g1 = t1_g1 + timedelta(hours=5)
        t2_g2 = t1_g2 + timedelta(hours=5)
        t2_g3 = t1_g3 + timedelta(hours=5)
        t2_g4 = t2_g3 + timedelta(days=1)
        t2_games = [t2_g1.timestamp(), t2_g2.timestamp(), t2_g3.timestamp(), t2_g4.timestamp()]

        columns = ['date']
        df_tg = pd.DataFrame(np.array(t1_games), columns=columns).set_index('date').sort_index()
        df_bd = pd.DataFrame(np.array(t2_games), columns=columns).set_index('date').sort_index()

        # Act
        big_data_util.delete_games_not_matching(df_tg, df_bd)
        # Assert

    def test_fix_starting_lineup(self):
        # Arrange
        year = 2007
        df_bs = big_data_util.get_box_score_dataframe(year)

        # Act
        ds_fixed = big_data_util.fix_starting_lineup_columns(df_bs)

        # Assert
        assert len([c for c in [COLUMN_SL_1, COLUMN_SL_2, COLUMN_SL_3, COLUMN_SL_4, COLUMN_SL_5] if c in ds_fixed.columns])


