import logging
import os
from unittest import TestCase

from pandas import DataFrame

from app.basketball.nba.load import load_games, load_player_games, convert_games_to_dataframe
from app.config import config
from app.services import file_services


class TestJon(TestCase):
    season_four = os.path.join(config.NBA_MATCHES_DIR, '2003-2004')

    def test_one(self):
        # Arrange
        json_files = file_services.walk_dir(self.season_four, '.json')

        # Act

        # Assert
        assert (len(json_files) > 0)
        logging.info(f'Num: {len(json_files)}')

    def test_parse_json(self):
        # Arrange
        games = load_games([2007], 2)

        # Act
        game = games[0]

        assert len(game.away.players) == 12
        assert game.away.totals.STL_pct == 4.1
        assert game.away.players[0].name == 'Travis Best'

        logging.info(f'Weight: {game.away.players[0].weight}')
        logging.info(f'Weight: {game.away.players[0].STL}')

    def test_get_all_player_games_in_order(self):
        # Arrange
        player_games = load_player_games(10)

        assert len(player_games) == 204

    def test_get_games_dataframe(self):
        # Act

        for i in range(2007, 2019):
            years = range(i, i + 1)
            all_rows, columns = convert_games_to_dataframe(years, None)

            logging.info(f'All rows size: {len(all_rows)}')

            df = DataFrame(all_rows, columns=columns)

            csv_path = os.path.join(config.SPORTS_DIR, f'games_{years[0]}.csv.new')
            df.to_csv(csv_path, index=False)

            # logging.info(f'College: {df["away_3_college"]}')
            logging.debug(f'Columns: {columns[0]}')

            logging.info(df.loc[0])

            # Assert
            assert len(columns) == len(all_rows[0])



