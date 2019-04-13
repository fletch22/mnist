import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from app.config import config
from app.util import object_utils

COLUMN_SL_5 = 'SL_5'

COLUMN_SL_4 = 'SL_4'

COLUMN_SL_3 = 'SL_3'

COLUMN_SL_2 = 'SL_2'

COLUMN_SL_1 = 'SL_1'

COLUMN_POS = 'POSS'

COLUMN_PACE = 'PACE'

COLUMN_OEFF = 'OEFF'

COLUMN_DEFF = 'DEFF'

COLUMN_CLOSING_TOTAL = 'CLOSING TOTAL'

COLUMN_CLOSING_ODDS = 'CLOSING ODDS'

COLUMN_OPENING_TOTAL = 'OPENING TOTAL'

COLUMN_OPENING_SPREAD = 'OPENING SPREAD'

COLUMN_TEAM_REST_DAYS = 'TEAM REST DAYS'

COLUMN_MONEYLINE = 'MONEYLINE'

COLUMN_CLOSING_SPREAD = 'CLOSING SPREAD'

COLUMN_NAME = 'TEAM'

COLUMN_DATE = 'DATE'

COLUMN_NAME_FULL = 'name'

COLUMN_UNIX_DATE = 'date'


def enhance_for_processing(df):
    df = df.replace('#VALUE!', '0', regex=True)

    vf_convert_date = np.vectorize(convert_to_unix_epoch)

    df[COLUMN_UNIX_DATE] = vf_convert_date(df[COLUMN_DATE]).astype('int64')

    df[COLUMN_MONEYLINE] = df[COLUMN_MONEYLINE].astype('float32')
    df[COLUMN_OPENING_SPREAD] = df[COLUMN_OPENING_SPREAD].astype('float32')
    df[COLUMN_CLOSING_SPREAD] = df[COLUMN_CLOSING_SPREAD].astype('float32')
    df[COLUMN_OPENING_TOTAL] = df[COLUMN_OPENING_TOTAL].astype('float32')
    df[COLUMN_CLOSING_TOTAL] = df[COLUMN_CLOSING_TOTAL].astype('float32')

    vf_make_team_col = np.vectorize(convert_name)

    df[COLUMN_NAME_FULL] = vf_make_team_col(df[COLUMN_NAME])

    vf_strip = np.vectorize(str.strip)

    df[COLUMN_NAME_FULL] = vf_strip(df[COLUMN_NAME_FULL])

    df_fixed = fix_starting_lineup_columns(df)

    return df_fixed


def convert_to_unix_epoch(str_date):
    return (datetime.strptime(str_date, "%m/%d/%Y") + timedelta(hours=-5)).timestamp()


def convert_name(team_name_old):
    return team_map[team_name_old]


team_map = {
    'Atlanta': 'Atlanta Hawks',
    'Boston': 'Boston Celtics',
    'Brooklyn': 'New Jersey Nets',
    'Charlotte': 'Charlotte Bobcats',
    'Chicago': 'Chicago Bulls',
    'Cleveland': 'Cleveland Cavaliers',
    'Dallas': 'Dallas Mavericks',
    'Denver': 'Denver Nuggets',
    'Detroit': 'Detroit Pistons',
    'Golden State': 'Golden State Warriors',
    'Houston': 'Houston Rockets',
    'Indiana': 'Indiana Pacers',
    'LA Clippers': 'Los Angeles Clippers',
    'LA Lakers': 'Los Angeles Lakers',
    'Memphis': 'Memphis Grizzlies',
    'Miami': 'Miami Heat',
    'Milwaukee': 'Milwaukee Bucks',
    'Minnesota': 'Minnesota Timberwolves',
    'New Jersey': 'New Jersey Nets',
    'New Orleans': 'New Orleans Hornets',
    'New Orleans/Oklahoma City': 'New Orleans Hornets',
    'New York': 'New York Knicks',
    'Oklahoma City': 'Oklahoma City Thunder',
    'Orlando': 'Orlando Magic',
    'Philadelphia': 'Philadelphia 76ers',
    'Phoenix': 'Phoenix Suns',
    'Portland': 'Portland Trail Blazers',
    'Sacramento': 'Sacramento Kings',
    'San Antonio': 'San Antonio Spurs',
    'Seattle': 'Seattle Supersonics',
    'Toronto': 'Toronto Raptors',
    'Utah': 'Utah Jazz',
    'Washington': 'Washington Wizards'
}


def has_matching_game(index, df):
    diff_max = 9 * 60 * 60
    result = False

    for index_other, row_other in df.iterrows():
        # logging.info(f'row date: {index}')
        # logging.info(f'row_other date: {index_other}')

        dt_1 = datetime.fromtimestamp(index)
        dt_2 = datetime.fromtimestamp(index_other)

        dt = dt_1 - dt_2

        if abs(dt.total_seconds()) <= diff_max:
            result = True

        # logging.info(f'ts: {dt.total_seconds()}: {diff_max}')

    return result

    # raise Exception('foo')
    # date = datetime.fromtimestamp(timestamp)
    # date_other = datetime.fromtimestamp(index)
    #
    # logging.info(date)
    # logging.info(date_other)
    #
    # diff = abs((date - date_other).total_seconds())
    #
    # logging.info(f'Diff: {diff}')


def delete_games_not_matching(df_team, df_bg_team):
    if len(df_team) == len(df_bg_team):
        return df_team, df_bg_team

    # # logging.info(f'df_team index: {df_team.index}')
    #
    # arr1 = df_team.index.values
    #
    # # logging.info(f'{arr1}')
    #
    # arr1 = np.append(arr1, [0], axis=0)
    #
    # # logging.info(f'{arr1}')
    # # logging.info(f'{df_bg_team.shape[0]}')
    # # logging.info(f'{df_bg_team.index.values}')
    #
    # arr1 = np.array(arr1)
    # arr1 = np.reshape(arr1, (arr1.shape[0], 1))
    #
    # def convert(timestamp):
    #     return str(datetime.fromtimestamp(timestamp))
    #
    # vf_convert = np.vectorize(convert)
    #
    # # logging.info(f'Shape: {np.array(arr1).shape}')
    #
    # arr2 = df_bg_team.index.values
    # arr2 = np.reshape(arr2, (arr2.shape[0], 1))
    #
    # arr1 = vf_convert(arr1)
    # arr2 = vf_convert(arr2)
    #
    # arr_combined = np.append(arr1, arr2, axis=1)

    # logging.info(f'{arr_combined}')

    if len(df_team) < len(df_bg_team):
        logging.info('df_bg_team is more.')
        df_team, df_bg_team = drop_non_matches(df_team, df_bg_team)
    else:
        df_bg_team, df_team = drop_non_matches(df_bg_team, df_team)

    return df_team, df_bg_team


def drop_non_matches(df, df_other):
    no_match = []
    for index, row in df_other.iterrows():
        # logging.info('df...')
        match_found = has_matching_game(index, df)

        if not match_found:
            logging.info(f'Will drop: {str(datetime.fromtimestamp(index))}')
            no_match.append(index)

    # delete no_matches
    # logging.info(f'Will drop: {no_match}')
    # logging.info(f'len: {len(df_other)}')
    df_other = df_other.drop(no_match, axis='index')

    # df = pd.concat([df])

    return df, df_other


def join_with_big_data_ball(df_team_game, df_bd):
    logging.info('About to join with big data ball...')

    df_tg_indexed = df_team_game.set_index(COLUMN_UNIX_DATE).sort_index()
    df_bd_enhanced = df_bd.set_index(COLUMN_UNIX_DATE).sort_index()

    df_bd_enhanced = object_utils.keep_cols(df_bd_enhanced, [COLUMN_NAME_FULL, COLUMN_UNIX_DATE, COLUMN_MONEYLINE, COLUMN_CLOSING_SPREAD, COLUMN_TEAM_REST_DAYS,
                                                             COLUMN_OPENING_SPREAD, COLUMN_OPENING_TOTAL, COLUMN_CLOSING_TOTAL, COLUMN_OEFF, COLUMN_DEFF,
                                                             COLUMN_POS, COLUMN_PACE, COLUMN_SL_1, COLUMN_SL_2, COLUMN_SL_3, COLUMN_SL_4, COLUMN_SL_5])

    df_bd_enhanced = df_bd_enhanced.rename(
        columns={COLUMN_OEFF: f'tot_{COLUMN_OEFF}', COLUMN_DEFF: f'tot_{COLUMN_DEFF}', COLUMN_POS: f'tot_{COLUMN_POS}', COLUMN_PACE: f'tot_{COLUMN_PACE}'})

    team_names = np.unique(df_tg_indexed[COLUMN_NAME_FULL].values)
    team_names = sorted(team_names)

    team_list = []
    for t in team_names:
        # logging.info(f'Joining team: {t}')

        df_team = df_tg_indexed.loc[df_tg_indexed[COLUMN_NAME_FULL] == t]
        df_bg_team = df_bd_enhanced.loc[df_bd_enhanced[COLUMN_NAME_FULL] == t]

        # logging.info(f'Ldft: {len(df_team)}')
        # logging.info(f'Ldbt: {len(df_bg_team)}')

        if len(df_team) != len(df_bg_team):
            continue

        # (df_team, df_bg_team) = delete_games_not_matching(df_team, df_bg_team)
        # logging.info(f'Ldft: {len(df_team)}')
        # logging.info(f'Ldbt: {len(df_bg_team)}')

        df_team = df_team.reset_index(level=COLUMN_UNIX_DATE)
        df_bg_team = df_bg_team.reset_index(level=COLUMN_UNIX_DATE)

        df_bg_miami_ren = df_bg_team.rename(index=str, columns={COLUMN_UNIX_DATE: 'bd_date'})

        df_with_new_date = df_bg_miami_ren.assign(date=df_team[COLUMN_UNIX_DATE].values)
        df_with_new_date[COLUMN_UNIX_DATE] = df_with_new_date[COLUMN_UNIX_DATE].astype('int64')

        df_final = df_with_new_date.drop(['bd_date'], axis=1)

        team_list.append(df_final)

    df_date_corrected = pd.concat(team_list, axis=0)

    df_date_corrected[COLUMN_NAME_FULL] = df_date_corrected[COLUMN_NAME_FULL].astype('str')
    df_tg_indexed[COLUMN_NAME_FULL] = df_tg_indexed[COLUMN_NAME_FULL].astype('str')

    index_cols = [COLUMN_NAME_FULL, COLUMN_UNIX_DATE]
    df_bd_indexed = df_date_corrected.set_index(index_cols).sort_index()
    df_tg_indexed = df_tg_indexed.reset_index(level=COLUMN_UNIX_DATE).set_index(index_cols).sort_index()

    df_joined = df_tg_indexed.join(df_bd_indexed, how='inner')

    return df_joined.reset_index(index_cols)


def get_timezone_diff(origin_city, destination_city):
    gmt_origin = team_location[origin_city]['gmt']
    gmt_destination = team_location[destination_city]['gmt']

    return gmt_destination - gmt_origin


def calculate_distance(city_name1, city_name2):
    from math import sin, cos, sqrt, atan2, radians

    # approximate radius of earth in miles
    R = 3958.8

    lat1 = radians(team_location[city_name1]['n'])
    lon1 = radians(team_location[city_name1]['w'])
    lat2 = radians(team_location[city_name2]['n'])
    lon2 = radians(team_location[city_name2]['w'])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


team_location = {
    "Atlanta Hawks": {"n": 33.7490, "w": -84.3880, "gmt": -4},
    "Boston Celtics": {"n": 42.361145, "w": -71.057083, "gmt": -4},
    "Brooklyn Nets": {'n': 40.650002, 'w': -73.949997, "gmt": -4},
    "New Jersey Nets": {'n': 40.735657, 'w': -74.172363, "gmt": -4},
    "Charlotte Bobcats": {'n': 35.227085, 'w': -80.843124, "gmt": -4},
    "Charlotte Hornets": {'n': 35.227085, 'w': -80.843124, "gmt": -4},
    "Chicago Bulls": {'n': 41.881832, 'w': -87.623177, "gmt": -5},
    "Cleveland Cavaliers": {'n': 41.505493, 'w': -81.681290, "gmt": -4},
    "Dallas Mavericks": {'n': 32.897480, 'w': -97.040443, "gmt": -5},
    "Denver Nuggets": {'n': 39.742043, 'w': -104.991531, "gmt": -6},
    "Detroit Pistons": {'n': 42.331429, 'w': -83.045753, "gmt": -5},
    "Golden State Warriors": {'n': 37.804363, 'w': -122.271111, "gmt": -7},
    "Houston Rockets": {'n': 29.749907, 'w': -95.358421, "gmt": -5},
    "Indiana Pacers": {'n': 39.832081, 'w': -86.145454, "gmt": -5},
    "Los Angeles Clippers": {'n': 34.052235, 'w': -118.243683, "gmt": -7},
    "Los Angeles Lakers": {'n': 34.052235, 'w': -118.243683, "gmt": -7},
    "Memphis Grizzlies": {'n': 35.040031, 'w': -89.981873, "gmt": -5},
    "Miami Heat": {'n': 25.761681, 'w': -80.191788, "gmt": -4},
    "Milwaukee Bucks": {'n': 43.038902, 'w': -87.906471, "gmt": -5},
    "Minnesota Timberwolves": {'n': 44.986656, 'w': -93.258133, "gmt": -5},
    "New Orleans Pelicans": {'n': 29.951065, 'w': -90.071533, "gmt": -5},
    "New Orleans Hornets": {'n': 29.951065, 'w': -90.071533, "gmt": -5},
    "New Orleans/Oklahoma City Hornets": {'n': 29.951065, 'w': -90.071533, "gmt": -5},
    "New York Knicks": {'n': 40.758896, 'w': -73.985130, "gmt": -4},
    "Oklahoma City Thunder": {'n': 35.481918, 'w': -97.508469, "gmt": -5},
    "Orlando Magic": {'n': 28.538336, 'w': -81.379234, "gmt": -4},
    "Philadelphia 76ers": {'n': 39.952583, 'w': -75.165222, "gmt": -4},
    "Phoenix Suns": {'n': 33.448376, 'w': -112.074036, "gmt": -5},
    "Portland Trail Blazers": {'n': 45.512794, 'w': -122.679565, "gmt": -7},
    "Sacramento Kings": {'n': 38.575764, 'w': -121.478851, "gmt": -7},
    "San Antonio Spurs": {'n': 29.424349, 'w': -98.491142, "gmt": -5},
    "Seattle SuperSonics": {'n': 47.608013, 'w': -122.335167, "gmt": -7},
    "Toronto Raptors": {'n': 43.761539, 'w': -79.411079, "gmt": -4},
    "Utah Jazz": {'n': 39.419220, 'w': -111.950684, "gmt": -6},
    "Washington Wizards": {'n': 38.889931, 'w': -77.009003, "gmt": -4}
}

nickname_full_name_map = {
    "76ers": "Philadelphia 76ers",
    "Blazers": "Portland Trail Blazers",
    "Bobcats": "Charlotte Bobcats",
    "Bucks": "Milwaukee Bucks",
    "Bulls": "Chicago Bulls",
    "Cavaliers": "Cleveland Cavaliers",
    "Celtics": "Boston Celtics",
    "Clippers": "Los Angeles Clippers",
    "Grizzlies": "Memphis Grizzlies",
    "Hawks": "Atlanta Hawks",
    "Heat": "Miami Heat",
    "Hornets": "New Orleans Hornets",
    "Jazz": "Utah Jazz",
    "Kings": "Sacramento Kings",
    "Knicks": "New York Knicks",
    "Lakers": "Los Angeles Lakers",
    "Magic": "Orlando Magic",
    "Mavericks": "Dallas Mavericks",
    "Nets": "New Jersey Nets",
    "Nuggets": "Denver Nuggets",
    "Pacers": "Indiana Pacers",
    "Pelicans": "New Orleans Pelicans",
    "Pistons": "Detroit Pistons",
    "Raptors": "Toronto Raptors",
    "Rockets": "Houston Rockets",
    "Sonics": "Seattle Supersonics",
    "Supersonics": "Seattle Supersonics",
    "Spurs": "San Antonio Spurs",
    "Suns": "Phoenix Suns",
    "Thunder": "Oklahoma City Thunder",
    "Timberwolves": "Minnesota Timberwolves",
    "Warriors": "Golden State Warriors",
    "Wizards": "Washington Wizards"
}


def fix_starting_lineup_columns(df):
    return df.rename(index=str,
                     columns={'STARTING LINEUPS': 'SL_1', 'Unnamed: 38': 'SL_2', 'Unnamed: 39': 'SL_3', 'Unnamed: 40': 'SL_4', 'Unnamed: 41': 'SL_5'})


def get_box_score_dataframe(years):
    basename = f'{years}_NBA_Box_Score_Team'
    filename = f'{basename}.csv'
    file_path = os.path.join(config.SPORTS_DIR, 'big_data_ball', filename)
    return pd.read_csv(file_path)
