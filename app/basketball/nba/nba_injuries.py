import logging
import os

import numpy as np
import pandas as pd

from app import ds_util
from app.basketball.nba.big_data_ball import big_data_util
from app.config import config
from app.time_series import nba_team_game_learning

COL_INJ_TEAM_NM_RESOLVED = 'team_name_resolved'
COL_INJ_ACQUIRED = 'Acquired'
COL_INJ_ACQUIRED_RESOLVED = 'Acquired_Resolved'
COL_INJ_RELINQ = 'Relinquished'
COL_INJ_RELINQ_RESOLVED = 'Relinquished_Resolved'
COL_INJ_DATE = 'Date'
COL_INJ_UNIX_DATE = 'unix_date'
COL_INJ_TEAM = 'Team'

PLAYER_UNRECOGNIZED = 'PLAYER_UNRECOGNIZED'


def load_raw_data():
    file_path = os.path.join(config.NBA_DIR, 'nba_injuries_2007-2018.csv')

    return pd.read_csv(file_path)


def load_data():
    df = load_raw_data()

    vf_parse = np.vectorize(convert_to_unix)
    df[COL_INJ_UNIX_DATE] = vf_parse(df[COL_INJ_DATE]).astype('float32')

    vf_convert_name = np.vectorize(convert_to_resolved_name)

    df[COL_INJ_ACQUIRED] = df[COL_INJ_ACQUIRED].apply(lambda x: str(x).strip())
    df[COL_INJ_ACQUIRED_RESOLVED] = vf_convert_name(df[COL_INJ_ACQUIRED])

    df[COL_INJ_RELINQ] = df[COL_INJ_RELINQ].apply(lambda x: str(x).strip())
    df[COL_INJ_RELINQ_RESOLVED] = vf_convert_name(df[COL_INJ_RELINQ])

    vf_convert_team_name = np.vectorize(convert_to_resolved_team_name)
    df[COL_INJ_TEAM_NM_RESOLVED] = vf_convert_team_name(df[COL_INJ_TEAM])

    return df


def get_nba_player_names():
    nba_file_path = os.path.join(config.NBA_DIR, 'joined-2007-2018.csv')
    df_nba = pd.read_csv(nba_file_path)

    all_players = []
    for i in range(14):
        player_column = f'{i}_name'
        if player_column in df_nba.columns:
            df_nba[player_column] = df_nba[player_column].apply(lambda x: str(x).strip())
            players = list(df_nba[player_column].values)
            all_players = all_players + players

    return list(np.unique(all_players))


NBA_PLAYERS = get_nba_player_names()


def find_nba_name_match(last_name):
    matches = []
    for p in NBA_PLAYERS:
        p_stripped = p.strip()
        names = p_stripped.split(' ')
        nba_last_name = names[-1:][0]
        if nba_last_name == last_name:
            matches.append(p_stripped)

    return matches


def convert_to_resolved_name(unresolved_name):
    if unresolved_name is not np.nan:
        names = unresolved_name.split('/')

        for n in names:
            n_stripped = n.strip()
            if n_stripped in NBA_PLAYERS:
                found_player = True
                return n_stripped
            else:
                continue

    return PLAYER_UNRECOGNIZED


def convert_to_resolved_team_name(unresolved_team_name):
    result = 'UNKNOWN_TEAM'
    if unresolved_team_name is not np.nan:
        nickname_map = big_data_util.nickname_full_name_map
        result = nickname_map[unresolved_team_name]

    return result


def convert_to_unix(dateString):
    # print(dateString)
    return ds_util.parse(dateString).timestamp()


def compose_health_state_key(player_name, team_name):
    return f'{player_name}-{team_name}'


def get_injury_state(date, player_name, field_to_set, range_map, team_name):
    key = compose_health_state_key(player_name, team_name)

    health_stat_roll = {'start_injury': None, 'end_injury': None}
    if key in range_map.keys():
        health_stat_roll = range_map[key]

    health_stat_roll[field_to_set] = date

    return key, health_stat_roll


def create_health_states_map(df):
    health_states = {}
    for index, row in df.iterrows():
        player_name_healthy = row[COL_INJ_ACQUIRED_RESOLVED]
        player_name_injured = row[COL_INJ_RELINQ_RESOLVED]
        date = row[COL_INJ_UNIX_DATE]
        team_name = row[COL_INJ_TEAM_NM_RESOLVED]

        key, health_state = get_injury_state(date, player_name_injured, 'start_injury', health_states, team_name)
        health_states[key] = health_state

        key, health_state = get_injury_state(date, player_name_healthy, 'end_injury', health_states, team_name)
        health_states[key] = health_state

    return health_states


def is_injured(health_states, player_name, team_name, game_day_timestamp):
    key = compose_health_state_key(player_name, team_name)

    result = False
    if key in health_states.keys():
        health_status = health_states[key]
        start_inj = health_status['start_injury']
        end_inj = health_status['end_injury']
        if start_inj is not None:
            if start_inj < game_day_timestamp \
                    and (end_inj is None or end_inj > game_day_timestamp):
                logging.info(f's: {start_inj}; e: {end_inj}; actual: {game_day_timestamp}')
                result = True

    return result


def get_player_state_per_game(df_nba, health_states):
    logging.info("About to strip NBA player names...")

    num_players = 12
    for i in range(num_players):
        player_column = f'{i}_name'
        df_nba[player_column] = df_nba[player_column].apply(lambda x: str(x).strip())

    count_injured = 0
    new_data = []
    for index, row in df_nba.iterrows():
        new_row = pd.Series()
        date = index
        team_name = row[nba_team_game_learning.COLUMN_NAME]
        for i in range(num_players):
            player_column = f'{i}_name'
            player_name = row[player_column]
            has_injury = is_injured(health_states, player_name, team_name, date)

            if has_injury is True:
                count_injured += 1

            new_row[f'{i}_injury_state'] = has_injury

        # logging.info(f'New row: {new_row}')
        new_data.append(new_row)

    logging.info(f"Found {count_injured} injury/healthy records.")

    return new_data
