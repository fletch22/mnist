import os
import warnings

from app.config import config

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import logging
from unittest import TestCase

from app.time_series import nba_bf_firehost, nba_features


class TestNbaBfFireHost(TestCase):

    def test_learn(self):
        # Arrange
        nba_bf_firehost.learn(None)

    def test_learn_get_team_rollups(self):
        # Arrange
        df = get_games()
        series_current_game = df.iloc[0]

        # logging.info(f'df_curren: {type(df_current_game)}')
        logging.info(f'Columns: {series_current_game.index}')

        team_name = series_current_game['home_name']
        team_name_unencoded = nba_bf_firehost.get_team_name(team_name)
        date_game = series_current_game['date']
        num_past_games = 9

        logging.info(f'Team name: {team_name_unencoded}')

        df = df.set_index('date').sort_index()

        series_current_game = nba_bf_firehost.get_team_rollups(df, series_current_game, team_name, date_game, num_past_games, "away")

        # logging.info(f'foo: {df_current_game.iloc[0]['']}')

    def test_get_team_tot_diffs(self):
        # Arrange
        df = get_games()
        df = df.set_index('date')

        series_current_game = df.iloc[0]
        date = df.index[0]
        num_histories = 2

        team_name_encoded = series_current_game['home_name']

        series_current_game_enh = nba_bf_firehost.get_team_tot_diffs(df, series_current_game, team_name_encoded, date, num_histories)

        logging.info(f'new enh: {series_current_game_enh.index}')

    def test_balance_team(self):
        # Arrange
        df = get_games()
        df = df.set_index('date')

        logging.info(f'cols: {df.columns}')

        # logging.info(f'w: {df["did_win"]}')
        df = nba_bf_firehost.balance_data(df)

        df['did_win'] = ((df['home_score'] - df['away_score']) > 0)

        df_wins = df[df['did_win'] == 1]
        df_losses = df[df['did_win'] == 0]

        num_games = df.shape[0]
        num_wins = df_wins.shape[0]
        num_losses = df_losses.shape[0]

        logging.info(f'after num games: {num_games}; num wins: {num_wins}; num losses: {num_losses}')

    def test_probab_extraction(self):
        # Arrange
        proba1 = [[0.42, 0.58],
                  [0.38, 0.62],
                  [0.48, 0.52],
                  [0.34, 0.66],
                  [0.46, 0.54],
                  [0.4, 0.6],
                  [0.16, 0.84],
                  [0.78, 0.22],
                  [0.68, 0.32],
                  [0.5, 0.5]]

        y = [0, 1, 1, 1, 1, 1, 1, 0, 0, 1]
        y = np.array(y)
        y = np.reshape(y, (y.shape[0], 1))

        # logging.info(f'y: {y.shape}')

        all_probas = []
        all_probas.append(proba1)
        all_probas.append(proba1)

        # Act
        win_pct = nba_bf_firehost.score_many_clf_probas(all_probas, y)

        logging.info(win_pct)

        # Assert
        assert (win_pct == .9)

    def test_features(self):
        # Arrange
        features = nba_features.features

        # Act
        top_n = 10
        columns = []
        for i in range(top_n):
            score, column_name = features[i]
            columns.append(column_name)

        # Assert
        logging.info(f'columns: {columns}')

    def test_subtracting_array(self):
        telling_columns = [1, 2, 3]
        b = [4, 5, 6]

        c = [x for x in telling_columns if x not in b]

        c = telling_columns - b

    def test_keep_best_columns(self):
        # Arrange
        df = get_games()
        df = df.set_index('date')

        num_to_keep = 10

        logging.info(f'is greater inf > 10 ? {np.inf < 10}')

        # Act
        df = nba_bf_firehost.keep_best_columns(df, np.inf)

        logging.info(f'columns: {df.columns}')

        # Assert
        assert (df.columns.all(['away_score', 'away_name', 'home_score', 'home_name']))

    def test_save_to_csv(self):
        # Arrange
        df = get_games()
        df = df.set_index('date')

        # Act
        nba_bf_firehost.save_to_csv(df, 123, 123)

    def test_read_big_file(self):
        # Arrange
        file_path = os.path.join(config.DATA_FOLDER_PATH, 'sports', 'processed_basketball_2002-2012.csv')

        # Act
        df = pd.read_csv(file_path, delimiter=',')

        # Assert
        assert 10164 == df.shape[0]
        assert 2343 == df.shape[1]

    def test_learn_from_preprocessed_file(self):
        # Arrange

        file_path = os.path.join(config.DATA_FOLDER_PATH, 'sports', 'processed_basketball_2002-2012.csv')
        num_players = 10

        for i in range(20, 30):
            keep_cols = nba_bf_firehost.get_col_names_from_best_features(i)

            nba_bf_firehost.learn_from_preprocessed_file(file_path, num_players, keep_cols)

        # Assert


def get_games():
    df = nba_bf_firehost.get_dataframe()
    df = df.reset_index('date').sort_index()
    return df
