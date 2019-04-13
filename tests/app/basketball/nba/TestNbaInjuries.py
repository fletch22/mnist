import logging
import os
from unittest import TestCase

import pandas as pd

from app.basketball.nba import nba_injuries
from app.basketball.nba.nba_injuries import get_nba_player_names, create_health_states_map, is_injured, get_player_state_per_game
from app.config import config
from app.time_series import nba_team_game_learning
from app.util.object_utils import convert_series_array_to_df


class TestNbaInjuries(TestCase):

    def test_load_data(self):
        # Arrange
        # Act
        df = nba_injuries.load_raw_data()

        # Assert
        assert (df is not None)

    def test_is_injured(self):
        df_injuries = nba_injuries.load_data()

        health_states = create_health_states_map(df_injuries)

        logging.info(f'Sample: {health_states["Richard Hamilton-Detroit Pistons"]}')

        player_name = 'Richard Hamilton'
        team_name = 'Detroit Pistons'
        # Start Time: 1300424448.0

        self.determine_injury(health_states, player_name, team_name, 1300424449.0, True)
        self.determine_injury(health_states, player_name, 'thisteamnamedoesnotexist', 1300424449.0, False)
        self.determine_injury(health_states, 'thisplayerdoesnotexist', team_name, 1300424449.0, False)
        self.determine_injury(health_states, player_name, team_name, 1200424449.0, False)

    def test_set_player_state(self):
        df_injuries = nba_injuries.load_data()

        health_states = create_health_states_map(df_injuries)

        nba_file_path = os.path.join(config.NBA_DIR, 'joined-2007-2008.csv')
        df_nba = pd.read_csv(nba_file_path)
        # df_nba = df_nba.iloc[-203:]
        df_nba = df_nba.set_index(nba_team_game_learning.COLUMN_DATE).sort_index()

        states_per_game = get_player_state_per_game(df_nba, health_states)

        df_states = convert_series_array_to_df(states_per_game)

        logging.info(f'df_states shape: {df_states.shape}')
        logging.info(f'df_nba shape: {df_nba.shape}')

        df_combined = pd.concat([df_nba, df_states], axis=1, join_axes=[df_nba.index])

        num_players = 12
        for i in range(num_players):
            col = f'{i}_injury_state'
            # logging.info(df_combined[col].head(20))

        # for c in df_combined:
        #     logging.info(f'c: {c}')

        assert (len(states_per_game) > 0)

    def determine_injury(self, health_states, player_name, team_name, game_day_timestamp, expected_injury_status):
        has_injury = is_injured(health_states, player_name, team_name, game_day_timestamp)

        assert has_injury is expected_injury_status

    def test_get_nba_player_names(self):
        # Arrange
        get_nba_player_names()

        # Act
        # Assert
        assert (1 == 1)
