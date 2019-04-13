import json
import logging
import operator
import os

from app.basketball.nba import Player
from app.basketball.nba.Game import Game
from app.config import config
from app.services import file_services
from app.util.object_utils import to_dict


def load_games(arr_season_end_year, num_games=None):
    games = []
    for season_end_year in arr_season_end_year:
        game_dir_path = os.path.join(config.NBA_MATCHES_DIR, f'{season_end_year - 1}-{season_end_year}')

        # logging.debug(f'Game Path: {game_dir_path}')

        json_files = file_services.walk_dir(game_dir_path, '.json')

        for json_path in json_files[:num_games]:
            logging.info(f'Getting json file {json_path}')
            with open(json_path) as f:
                dict_game = json.load(f)
                games.append(Game(dict_game))

    return games


def load_player_games(num_games):
    games = load_games([2004], num_games)

    player_games = []
    for g in games:
        for p in g.away.players:
            p.game_date = g.date
            player_games.append(to_dict(p))

        for p in g.home.players:
            p.game_date = g.date
            player_games.append(to_dict(p))

    from operator import itemgetter

    return sorted(player_games, key=itemgetter('game_date'), reverse=False)


def get_attrs_from_obj(obj):
    return [a for a in dir(obj) if not a.startswith('__') and not callable(getattr(obj, a))]


def get_player_columns(player, col_prefix):
    p_attr_list = get_attrs_from_obj(player)

    p_cols = [f'{col_prefix}{p_attr}' for p_attr in p_attr_list]
    logging.debug(f'p_cols: {p_cols}')
    return p_cols


# def collect_columns_from_year_range(years, arr_season_end_year, num_games):
#     games = load_games(arr_season_end_year, num_games)

def convert_games_to_dataframe(arr_season_end_year, num_games):
    logging.info("loading game")
    games = load_games(arr_season_end_year, num_games)

    all_rows = []
    columns = []
    row = []
    for g in games:
        logging.info("converting game")
        attr_list = get_attrs_from_obj(g)

        logging.debug(f'attr_list: {attr_list}')

        game_dict = to_dict(g)

        for game_attr in attr_list:

            if game_attr != 'home' and game_attr != 'away':
                if not columns.__contains__(game_attr):
                    columns.append(game_attr)
                row.append(game_dict[game_attr])
            else:
                player_values, player_columns = get_player_data(game_dict, game_attr)
                if not columns.__contains__(player_columns[0]):
                    logging.debug(f'Len pc: {len(player_columns)}')
                    logging.debug(f'play_col: {player_columns[0]}')
                    columns += player_columns
                logging.debug(f'Len pv: {len(player_values)}')
                row += player_values

                game_total_values, game_total_columns = get_team_game_data(game_dict, game_attr)
                if not columns.__contains__(game_total_columns[0]):
                    logging.debug(f'game_total_columns: {game_total_columns[0]}')
                    logging.debug(f'Len gtc: {len(game_total_columns)}')
                    columns += game_total_columns
                logging.debug(f'Len gtv: {len(game_total_values)}')
                row += game_total_values

        all_rows.append(row)
        row = []

    return all_rows, columns


def get_player_data(game_dict, game_attr):
    p_values = []
    p_columns = []
    players = game_dict[game_attr].players
    players = sorted(players, key=operator.attrgetter('MP'), reverse=True)

    logging.debug(f'Total number players found: {len(players)}')

    for index in range(14):
        if index >= len(players):
            p = Player.Player(dict_loader=None)
        else:
            p = players[index]

        col_prefix = f'{game_attr}_{index}_'
        p_columns += get_player_columns(p, col_prefix)

        player_dict = to_dict(p)
        p_attr_list = get_attrs_from_obj(p)

        logging.debug(f'p_attr_list: {p_attr_list}')

        for p_key in p_attr_list:
            logging.debug(f'player_dict: {player_dict}')
            p_values.append(player_dict[p_key])

    # logging.info(f'Number of p cols: {len(p_columns)}')

    return p_values, p_columns


def get_team_game_data(game_dict, game_attr):
    values = []
    team_game = game_dict[game_attr]

    game_totals = team_game.totals
    columns = get_attrs_from_obj(game_totals)

    game_totals_dict = to_dict(game_totals)
    game_tot_columns = []
    for c in columns:
        game_tot_columns.append(f'{game_attr}_tot_{c}')
        values.append(game_totals_dict[c])

    attr = 'score'
    game_tot_columns.append(f'{game_attr}_score')
    values.append(team_game.__getattribute__(attr))

    attr = 'name'
    game_tot_columns.append(f'{game_attr}_name')
    values.append(team_game.__getattribute__(attr))

    return values, game_tot_columns
