import logging
import os
from datetime import datetime, timedelta
from unittest import TestCase

import numpy as np
import pandas as pd

from app.basketball.nba.big_data_ball import big_data_util
from app.basketball.nba.big_data_ball.big_data_util import join_with_big_data_ball
from app.config import config
from app.time_series import nba_team_game_learning
from app.time_series.nba_team_game_learning import convert_game_time_to_seconds, COLUMN_DATE, get_windowed_records


class TestNbaTeamGameLearning(TestCase):

    def test_pull_out_team_from_games(self):
        # Arrange
        for i in range(2007, 2019):
            filename = f'games_{i}.csv'
            # filename = 'games_2002-2018.csv'
            file_path = os.path.join(config.DATA_FOLDER_PATH, 'sports', filename)

            # Act
            df = pd.read_csv(file_path, delimiter=',')

            prefix_team_1 = 'away_'
            prefix_team_2 = 'home_'

            df_team_games = nba_team_game_learning.pull_out_team_from_games(df, prefix_team_1, prefix_team_2)

            # logging.info(f'Found {df_team_games.shape[0]} games.')
            # return

            df_team_games = df_team_games.drop(['index'], axis='columns')

            output_dir = os.path.join(config.DATA_FOLDER_PATH, 'sports', 'nba', 'team_game')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'tg_{filename}')

            df_team_games.to_csv(output_path, sep=",", index=False)
            logging.info(f'Wrote file {output_path}.')

    def test_learn(self):
        # Arrange
        start_year = 2007
        end_year = 2018
        # shelf_path = os.path.join(config.TEMP_DIR, 'shelves', f'{years}_verification_stage')
        years = start_year if end_year is None else f'{start_year}-{end_year}'
        shelf_path = os.path.join(config.TEMP_DIR, 'shelves', f'{years}_w_last_player_pos_D')

        game_limit = None
        num_players = 12
        min_num_histories = 14

        empty_shelves = False

        for i in range(1):
            num_histories = min_num_histories + i

            # Act
            nba_team_game_learning.learn(start_year, end_year, game_limit=game_limit, num_histories=num_histories, num_players=num_players, train_pct=.7,
                                         shelf_path=shelf_path, empty_shelves=empty_shelves)

        # Assert

    def test_prepped(self):
        # Arrange
        file_path = os.path.join(config.TEAM_GAME_DIR, 'prepped', 'nba_prepped_2007-2018_b.csv')

        # Act
        df = pd.read_csv(file_path)

        nba_team_game_learning.learn_from_df(df, 2007, 2007, 12, .90)

    def test_pd_series(self):
        # Arrange
        row = pd.Series()
        row['rd'] = "foo"

        thing = pd.Series()
        thing['bar'] = 'bvalue'

        # Act
        series_enhanced = pd.concat([row, thing], axis=1)

        logging.info(f'se: {series_enhanced.index}')

        # Assert

    def test_time_convert_seconds(self):
        # Arrange
        time_values = np.empty([100])
        time_values = time_values.astype(dtype=str)
        time_values[:] = '12:30:59'  #:30:00'

        df = pd.DataFrame(time_values, columns=['time'])

        df['time'] = convert_game_time_to_seconds(df, 'time')

        logging.info(f'df h: {df.head()}')

        # Assert
        assert (df['time'][0] == 45059)

    def test_windowing(self):
        # Arrange
        years = '2011'  # 2002-2018
        basename = f'games_{years}'
        filename = f'tg_{basename}.csv'
        file_path = os.path.join(config.TEAM_GAME_DIR, filename)

        df = pd.read_csv(file_path)

        df[COLUMN_DATE] = df[COLUMN_DATE].astype('int')

        df_indexed = df.set_index(COLUMN_DATE).sort_index()

        # Act
        df_windowed = None
        for index, row in df_indexed.iterrows():
            date = index

            logging.info(f'Date: {date}')

            df_windowed = get_windowed_records(df, date, -250)

            logging.info(f'dfw: {df_windowed.shape[0]}')
            break

        # Assert
        assert df_windowed.shape[0] == 6

    def test_simple_date_conversion(self):
        # Arrange
        years = '2010-2011'
        basename = f'{years}_NBA_Box_Score_Team'
        filename = f'{basename}.csv'
        file_path = os.path.join(config.SPORTS_DIR, 'big_data_ball', filename)

        df_big_data = pd.read_csv(file_path)

        date = df_big_data['DATE'][0]

        dt_converted = big_data_util.convert_to_unix_epoch(date)

    def test_join(self):
        years = '2008'  # 2002-2018
        basename = f'games_{years}'
        filename = f'tg_{basename}.csv'
        file_path = os.path.join(config.TEAM_GAME_DIR, filename)

        df_team_game = pd.read_csv(file_path)

        # df_tg_indexed = df_team_game.set_index(['name', 'date']).sort_index()

        basename = f'{years}_NBA_Box_Score_Team'
        filename = f'{basename}.csv'
        file_path = os.path.join(config.SPORTS_DIR, 'big_data_ball', filename)

        df_bd = pd.read_csv(file_path)
        df_joined = join_with_big_data_ball(df_team_game, df_bd)

        logging.info(f'joined size: {df_joined.shape[0]}')
        logging.info(f'bd Team name: {df_joined.iloc[0]}')

        # Assert
        assert df_joined.shape[0] == 2622

    def test_convert_date(self):
        dt = datetime.now()

        result = big_data_util.convert_to_unix_epoch('10/26/2010')

        logging.info(f'Result: {result}')

    def test_add_distance(self):
        # Arrange
        now = datetime.now()
        two_days_previous = (now + timedelta(days=-2)).timestamp()
        three_days_previous = (now + timedelta(days=-3)).timestamp()
        nine_days_previous = (now + timedelta(days=-9)).timestamp()

        arr1 = [1, nine_days_previous, 'Chicago Bulls', 'Detroit Pistons', False]
        arr2 = [2, three_days_previous, 'Chicago Bulls', 'Cleveland Cavaliers', False]
        arr3 = [3, two_days_previous, 'Chicago Bulls', 'Houston Rockets', True]
        arr4 = [4, two_days_previous, 'Des Moines Hawks', 'Charlotte Bobcats', False]
        arr5 = [5, now.timestamp(), 'Chicago Bulls', 'Boston Celtics', False]

        all_rows = [arr1]
        all_rows.append(arr2)
        all_rows.append(arr3)
        all_rows.append(arr4)
        all_rows.append(arr5)
        arr = np.array(all_rows)

        columns = ['ndx', nba_team_game_learning.COLUMN_DATE, nba_team_game_learning.COLUMN_NAME, nba_team_game_learning.COLUMN_OPP_NAME,
                   nba_team_game_learning.COLUMN_IS_HOME_GAME]

        df = pd.DataFrame(arr, columns=columns)

        df[nba_team_game_learning.COLUMN_DATE] = df[nba_team_game_learning.COLUMN_DATE].astype('float32')

        df = df.set_index(nba_team_game_learning.COLUMN_DATE).sort_index()

        # Act
        df_dist = nba_team_game_learning.add_distance_data(df)

        # Assert
        assert nba_team_game_learning.COLUMN_DISTANCE_TO_VENUE in df_dist.columns

        for index, row in df_dist.iterrows():
            logging.info(f'ndx: {row["ndx"]}; distances: {row[nba_team_game_learning.COLUMN_DISTANCE_TO_VENUE]}')

        distance_traveled_in_prev_four_days = df_dist.iloc[1][nba_team_game_learning.COLUMN_DISTANCE_TO_VENUE]
        assert distance_traveled_in_prev_four_days == 0

        most_recent_distance_traveled = df_dist.iloc[4][nba_team_game_learning.COLUMN_DISTANCE_TO_VENUE]
        assert round(most_recent_distance_traveled, 1) == 2582.2

    def test_get_matched_game(self):
        # Arrange
        year = 2007
        df = nba_team_game_learning.get_data(year, 14)

        away_or_home = 'home'

        count_not_present = 0
        df_new_cols = pd.DataFrame()
        for index, row in df.iterrows():
            original = pd.Series([1], index=['ASDF'])
            enhanced, count_missing = nba_team_game_learning.add_mean_games_score(away_or_home, original, row)

            count_not_present += count_missing

            df_new_cols = df_new_cols.append(enhanced, ignore_index=True)

        if count_not_present > 0:
            logging.warning(f'Number of player names not found in starting lineup: {count_not_present}.')

        df_final = pd.concat([df, df_new_cols], axis=1)

        # Assert
        assert 'home_starting_C' in df_final.columns

    def test_averaging_values_1(self):
        # Arrange
        df = pd.DataFrame([.51, .49, .98, .21], columns=['label'])
        acc = df.values

        y = np.array([True, False, True, False])

        # Act
        mean_res = np.mean(acc, axis=1)
        wins = (mean_res >= .5)  # , dtype=int)

        accuracy = ((y == wins).sum()) / wins.shape[0]

        # Assert
        assert accuracy == 1.0

    def test_averaging_values_2(self):
        # Arrange
        df = pd.DataFrame([.51, .49, .98, .21], columns=['label'])
        acc = df.values

        y = np.array([True, False, False, False])

        # Act
        mean_res = np.mean(acc, axis=1)
        wins = (mean_res >= .5)  # , dtype=int)

        accuracy = ((y == wins).sum()) / wins.shape[0]

        # Assert
        assert accuracy == .75

    def test_stitch_files_together(self):
        # Arrange
        start_year = 2007
        end_year = 2018
        acc = pd.DataFrame()
        df = nba_team_game_learning.get_data(start_year, end_year, 14)

        file_path = os.path.join(config.SPORTS_DIR, 'nba', f'joined-{start_year}-{end_year}.csv')
        df.to_csv(file_path)

def convert_date_utc(unix_timestamp):
    return (datetime.fromtimestamp(unix_timestamp) + timedelta(hours=+5)).timestamp()
