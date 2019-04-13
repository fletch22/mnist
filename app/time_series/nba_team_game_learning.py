import datetime
import logging
import os
import shelve

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.basketball.nba.big_data_ball import big_data_util
from app.basketball.nba.big_data_ball.big_data_util import join_with_big_data_ball, COLUMN_SL_1, COLUMN_SL_2, COLUMN_SL_3, COLUMN_SL_4, COLUMN_SL_5
from app.config import config
from app.services import file_services
from app.time_series import plain_classifiers, nba_pca, dnn
from app.util.object_utils import convert_series_array_to_df

COLUMN_IS_IN_WINDOW = 'is_in_window'

COLUMN_STARTING_MEAN_BIRTHDATE = 'starting_mean_birthdate'

COLUMN_OPEN_VS_CLOSE_SPREAD = 'open_v_close_spread'

COLUMN_MARG_VICT_ATS = 'margin_of_victory_ats'

COLUMN_AWAY_WIN_RATIO = 'away_win_ratio'

COLUMN_AWAY_LOSS_STREAK = 'away_loss_streak'

COLUMN_HOME_WIN_RATIO = 'home_win_ratio'

COLUMN_HOME_LOSS_STREAK = 'home_loss_streak'

COLUMN_HOME_AWAY_DIFF_STREAK = 'home_away_streak_diff'

COLUMN_AWAY_WIN_STREAK = 'away_win_streak'

COLUMN_HOME_WIN_STREAK = 'home_win_streak'

COLUMN_DID_WIN_PREV_GAME = 'did_win_previous_game'

COLUMN_TIMEZONE_TRAVEL_CHANGE = 'timzone_travel_change'

COLUMN_DISTANCE_TO_VENUE = 'distance'

COLUMN_DID_WIN_OPEN_SPREAD = 'did_win_open_spread'
COLUMN_DID_WIN_CLOSE_SPREAD = 'did_win_close_spread'

COLUMN_DATE = 'date'

COLUMN_GAME_CODE = 'code'

COLUMN_TIME = 'time'

COLUMN_LEAGUE = 'league'
COLUMN_GAME_TYPE = 'type'
COLUMN_SEASON = 'season'
COLUMN_NAME = 'name'
COLUMN_OPP_NAME = "opp_name"
COLUMN_OPP_SCORE = 'opp_score'
COLUMN_SCORE = 'score'
COLUMN_AWAY_SCORE = 'away_score'
COLUMN_HOME_SCORE = 'home_score'
COLUMN_RESILIENT_DATE = 'resilient_date'
COLUMN_HOME_NAME = 'home_name'
COLUMN_AWAY_NAME = 'away_name'
COLUMN_DID_WIN = 'did_win'
COLUMN_IS_HOME_GAME = 'is_home_game'
COLUMN_COUNTRY = 'country'
COLUMNS_PREFIX_TMP_KEEP = [COLUMN_AWAY_SCORE, COLUMN_HOME_SCORE, COLUMN_AWAY_NAME, COLUMN_HOME_NAME]
COLUMN_WIN_LABEL = COLUMN_DID_WIN_CLOSE_SPREAD  # COLUMN_DID_WIN_CLOSE_SPREAD #COLUMN_DID_WIN_CLOSE_SPREAD  # COLUMN_DID_WIN_OPEN_SPREAD, COLUMN_DID_WIN

UNKNOWN_VALUE = 'UNKNOWN'
MAX_PLAYERS = 20
POSITIONS = ['PG', 'SG', 'SF', 'PF', 'C']

le_team_name = LabelEncoder()
le_positions = LabelEncoder()


def learn(start_year, end_year, game_limit=None, num_histories=None, num_players=None, train_pct=.7, shelf_path=None, empty_shelves=False):
    X, y, train_X, train_y, test_X, test_y, x_columns = get_train_and_test_data(start_year, end_year, game_limit, num_histories, num_players, train_pct,
                                                                                shelf_path, empty_shelves)

    years = start_year if end_year is None else f'{start_year}-{end_year}'
    score(X, y, train_X, train_y, test_X, test_y, x_columns, years)


def learn_from_df(df, start_year, end_year, num_players, train_pct):
    X, y, train_X, train_y, test_X, test_y, x_columns = slice_and_scale(df, num_players, train_pct)

    years = start_year if end_year is None else f'{start_year}-{end_year}'
    score(X, y, train_X, train_y, test_X, test_y, x_columns, years)


def get_train_and_test_data(start_year, end_year, game_limit, num_histories, num_players, train_pct, shelf_path, empty_shelves):
    if empty_shelves is True:
        file_services.empty_dir(config.SHELVES_DIR)
    dict_shelf = None
    if shelf_path is not None:
        dict_shelf = shelve.open(shelf_path)

    df = get_data(start_year, end_year, num_players)

    df_reduced = drop_unneeded_players(df, num_players)

    df_enhanced = enhance_games(df_reduced, game_limit, num_histories, num_players, dict_shelf)

    dict_shelf.close()

    data_file_suffix = f'{start_year}-{end_year}'
    save_to_csv(df_enhanced, data_file_suffix, num_histories, num_players)

    X, y, train_X, train_y, test_X, test_y, x_columns = slice_and_scale(df_enhanced, num_players, train_pct)

    return X, y, train_X, train_y, test_X, test_y, x_columns


def get_data(start_year, end_year, num_players):
    df_team_game, df_bd = get_raw_data(start_year, end_year)

    df_team_game_enhanced, df_bd_enhanced = enhance_pre_join_dataframes(df_team_game, df_bd)

    df = join_with_big_data_ball(df_team_game_enhanced, df_bd_enhanced)

    df = df.replace('#VALUE!', '0', regex=True)

    df.columns = [c.replace('\n', '') for c in df.columns]

    df[COLUMN_SCORE] = df[COLUMN_SCORE].astype('float32')
    df[COLUMN_OPP_SCORE] = df[COLUMN_OPP_SCORE].astype('float32')
    df[big_data_util.COLUMN_OPENING_SPREAD] = df[big_data_util.COLUMN_OPENING_SPREAD].astype('float32')
    df[big_data_util.COLUMN_CLOSING_SPREAD] = df[big_data_util.COLUMN_CLOSING_SPREAD].astype('float32')

    df[COLUMN_MARG_VICT_ATS] = (df[COLUMN_SCORE] - df[COLUMN_OPP_SCORE]) + df[big_data_util.COLUMN_CLOSING_SPREAD]

    df[COLUMN_DID_WIN_OPEN_SPREAD] = ((df[COLUMN_SCORE] - df[COLUMN_OPP_SCORE]) + df[big_data_util.COLUMN_OPENING_SPREAD]) > 0
    df[COLUMN_DID_WIN_CLOSE_SPREAD] = (df[COLUMN_MARG_VICT_ATS]) > 0

    df[COLUMN_OPEN_VS_CLOSE_SPREAD] = (df[big_data_util.COLUMN_OPENING_SPREAD] - df[big_data_util.COLUMN_CLOSING_SPREAD])

    df_init_clean = df.drop([COLUMN_LEAGUE], axis=1)

    df_g_converted = convert_game_time_to_seconds(df_init_clean, COLUMN_TIME)

    df_g_converted = df_g_converted.set_index(COLUMN_DATE).sort_index()
    logging.info(f'Set index to \'{COLUMN_DATE}\'.')

    return df_g_converted


def enhance_pre_join_dataframes(df_team_game, df_box_score):
    df_team_game[big_data_util.COLUMN_UNIX_DATE] = df_team_game[big_data_util.COLUMN_UNIX_DATE].astype('int64')
    df_bd_enhanced = big_data_util.enhance_for_processing(df_box_score)

    return df_team_game, df_bd_enhanced


def calc_distance(row):
    city_1 = row[COLUMN_NAME]
    city_2 = row[COLUMN_OPP_NAME]
    is_home_game = row[COLUMN_IS_HOME_GAME]

    result = None
    if is_home_game is True:
        result = 0
    else:
        result = big_data_util.calculate_distance(city_1, city_2)

    return result


def add_distance_data(df):
    distance_arr = []

    df = df.set_index(COLUMN_DATE).sort_index()

    count = 0
    ndx = 0
    timezone_diffs = []
    for index, row in df.iterrows():
        date = index

        team_games = df.loc[(df[COLUMN_NAME] == row[COLUMN_NAME]) & (df.index <= date)]

        ndx += 1

        distance = 0
        tz_diff = 0
        if team_games.shape[0] > 1:
            is_current_home_game = row[COLUMN_IS_HOME_GAME]
            current_city_team = row[COLUMN_NAME] if is_current_home_game is True else row[COLUMN_OPP_NAME]

            last_game = team_games.iloc[team_games.shape[0] - 2]
            is_last_home_game = last_game[COLUMN_IS_HOME_GAME]
            last_city_team = last_game[COLUMN_NAME] if is_last_home_game is True else last_game[COLUMN_OPP_NAME]

            distance = int(big_data_util.calculate_distance(last_city_team, current_city_team))
            count += 1

            tz_diff = big_data_util.get_timezone_diff(last_city_team, current_city_team)

        distance_arr.append(distance)
        timezone_diffs.append(tz_diff)

    df[COLUMN_DISTANCE_TO_VENUE] = distance_arr
    df[COLUMN_TIMEZONE_TRAVEL_CHANGE] = timezone_diffs

    df = df.reset_index(level=COLUMN_DATE)

    return df


def get_raw_data(start_year, end_year):
    if end_year is None:
        end_year = start_year

    df_tg_acc = None
    df_bd_acc = None
    for year in range(start_year, end_year + 1):
        logging.info('About to get raw data from file.')
        df_team_game = get_team_game_dataframe(year)
        if df_tg_acc is None:
            df_tg_acc = df_team_game
        else:
            df_tg_acc = pd.concat([df_tg_acc, df_team_game], axis=0)

        df_bd = big_data_util.get_box_score_dataframe(year)
        if df_bd is None:
            df_bd_acc = df_bd
        else:
            df_bd_acc = pd.concat([df_bd_acc, df_bd], axis=0)

    return df_tg_acc, df_bd_acc


def get_team_game_dataframe(years):
    basename = f'games_{years}'
    filename = f'tg_{basename}.csv'
    file_path = os.path.join(config.TEAM_GAME_DIR, filename)
    df_team_game = pd.read_csv(file_path)
    return df_team_game


def drop_unneeded_players(df, num_players):
    df_result = df

    drop_columns = []
    if num_players is not None:
        for i in range(num_players, MAX_PLAYERS):
            columns = [c for c in df.columns if c.startswith(f'{i}_')]
            drop_columns += columns

    if len(drop_columns) > 0:
        df_result = df.drop(drop_columns, axis=1)

    logging.info("Dropped players we don't need.")

    return df_result


def score(X, y, train_X, train_y, test_X, test_y, x_columns, years):
    logging.info('Scoring data ...')

    # pca = PCA(n_components=500)
    # train_X = pca.fit_transform(train_X)
    # test_X = pca.transform(test_X)

    # logging.info(f'train_X shape: {train_X.shape}')
    # logging.info(f'test_X shape: {test_X.shape}')
    dnn.learn(200, train_X, train_y, test_X, test_y)
    # stratified_kfold(batch_size, create_baseline, train_X, train_y)
    plain_classifiers.score_plain_classifiers(years, train_X, train_y, test_X, test_y)
    # nba_pca.show_pca(train_X, train_y, test_X, test_y)
    # logits_features_score(X, y, x_columns)
    # plain_classifiers.logits_ridge_selection(train_X, train_y, x_columns)


def diff_positions(df):
    for p in POSITIONS:
        suffix = f'_starting_{p}'
        df[f'diff{suffix}'] = (df[f'home{suffix}'] - df[f'away{suffix}'])

    return df


def slice_and_scale(df, num_players, train_pct):
    logging.info('Slicing and scaling data ...')

    windows = []
    for index, row in df.iterrows():
        date = row['resilient_date']
        within_window = False
        logging.info(f'Month: {datetime.datetime.fromtimestamp(date).month}')
        if datetime.datetime.fromtimestamp(date).month > 10 or datetime.datetime.fromtimestamp(date).month < 2:
            within_window = True
        windows.append(within_window)

    df[COLUMN_IS_IN_WINDOW] = np.array(windows)

    df = df.loc[(df[COLUMN_IS_IN_WINDOW] == True)]

    logging.info(f'head: {df.head(20)}')

    df_encoded = label_encode_columns(df, num_players)

    X, y, train_X, train_y, test_X, test_y, x_columns = split_train_test(df_encoded, num_players, train_pct)

    standard_scaler = StandardScaler()
    train_X = standard_scaler.fit_transform(train_X)
    test_X = standard_scaler.transform(test_X)

    return X, y, train_X, train_y, test_X, test_y, x_columns


def drop_columns_from_df(df, columns_to_drop):
    col_that_exist = [c for c in columns_to_drop if c in df.columns]
    return df.drop(col_that_exist, axis=1)


def split_train_test(df, num_players, train_pct=.7):
    logging.info('About to split train and test...')
    np.set_printoptions(threshold=np.nan)
    pd.set_option('display.max_colwidth', -1)

    y = df[COLUMN_WIN_LABEL].values

    # df = drop_columns_from_df(df, [c for c in df.columns if c.startswith('home_mean') or c.startswith('away_mean')])
    #
    remaining_tell_cols = [x for x in get_telling_columns(num_players) if x in df.columns]

    # mp_cols = [c for c in remaining_tell_cols if '_MP' in c]
    # for c in mp_cols:
    #     df[c].loc[df[c] > 0] = -1
    # remaining_tell_cols = [c for c in remaining_tell_cols if '_MP' not in c]

    # drop_columns = remaining_tell_cols + [big_data_util.COLUMN_OPENING_SPREAD, big_data_util.COLUMN_OPENING_TOTAL, big_data_util.COLUMN_CLOSING_TOTAL, 'code',
    #                                       'country', 'season'] + [c for c in df.columns if '_home_mean_' in c]
    #
    # df_X_naked = drop_columns_from_df(df, remaining_tell_cols)



    # keep_columns = [COLUMN_IS_HOME_GAME, COLUMN_MONEYLINE, COLUMN_TEAM_REST_DAYS, COLUMN_NAME, COLUMN_OPP_NAME, COLUMN_HOME_WIN_STREAK, COLUMN_HOME_LOSS_STREAK,
    #                 COLUMN_AWAY_WIN_STREAK, COLUMN_AWAY_LOSS_STREAK, COLUMN_TIMEZONE_TRAVEL_CHANGE, 'type', COLUMN_DISTANCE_TO_VENUE]
    # keep_columns = [COLUMN_IS_HOME_GAME, COLUMN_MONEYLINE, COLUMN_TEAM_REST_DAYS, COLUMN_NAME, COLUMN_OPP_NAME, COLUMN_HOME_WIN_STREAK, COLUMN_HOME_LOSS_STREAK,
    #                 COLUMN_AWAY_WIN_STREAK, COLUMN_AWAY_LOSS_STREAK, COLUMN_TIMEZONE_TRAVEL_CHANGE, 'type', COLUMN_DISTANCE_TO_VENUE,
    #                 f'mean_diffs_{COLUMN_MARG_VICT_ATS}', f'mean_diffs_{COLUMN_DID_WIN}', f'mean_diffs_{COLUMN_OPEN_VS_CLOSE_SPREAD}',
    #                 big_data_util.COLUMN_OPENING_SPREAD, big_data_util.COLUMN_OPENING_TOTAL, big_data_util.COLUMN_CLOSING_TOTAL]
    # drop_columns = [COLUMN_WIN_LABEL] + [c for c in df.columns if c not in keep_columns]
    # df_X_naked = drop_columns_from_df(df, drop_columns)

    plain_mean = [c for c in df.columns if 'home_mean' in c or 'away_mean' in c]
    # starting_pos = [c for c in df.columns if 'home_starting' in c or 'away_starting' in c]

    drop_columns = [COLUMN_WIN_LABEL] + remaining_tell_cols + plain_mean # + starting_pos

    df_X_naked = drop_columns_from_df(df, drop_columns)

    # logging.info(df_X_naked['0_MP'].head())

    for c in df_X_naked.columns:
        logging.info(f'c: {c}')

    X = df_X_naked.values

    logging.info('\n')
    for c in df_X_naked.columns:
        logging.info(f'c: {c}')

    train_num = np.math.floor(df.shape[0] * train_pct)
    df_X_raw = df.iloc[:train_num]

    train_y = df_X_raw[COLUMN_WIN_LABEL].values
    # df_train_X = df_X_raw.drop(drop_columns, axis=1)
    df_train_X = drop_columns_from_df(df_X_raw, drop_columns)

    logging.info(f'Train cols: \n {[c for c in df_train_X.columns]}')

    df_X_test = df.iloc[train_num:]

    test_y = df_X_test[COLUMN_WIN_LABEL].values

    # df_winnowed = df_X_test.drop(drop_columns, axis=1)
    df_winnowed = drop_columns_from_df(df_X_test, drop_columns)
    test_X = df_winnowed.values

    logging.info(f'Train Data: {df_train_X.head()}')

    logging.info(f'Number of training samples: {df_train_X.shape[0]}')
    logging.info(f'Number of test samples: {test_X.shape[0]}')

    return X, y, df_train_X.values, train_y, test_X, test_y, df_train_X.columns

def save_to_csv(df_collected, data_file_suffix, num_histories, num_players):
    file_path = os.path.join(config.SESSION_DIR, 'df_collected.csv')

    message = f'''Saving df_collected to {file_path} using with attributes: date_file_suffix:{data_file_suffix}; num_games:{df_collected.shape[0]}; 
    num_histories={num_histories}; num_players:{num_players}; columns:{df_collected.columns};'''
    logging.info(message)

    os.makedirs(config.SESSION_DIR, exist_ok=True)

    meta_file_path = os.path.join(config.SESSION_DIR, 'df_collected_meta.txt')
    with open(meta_file_path, "w") as f:
        f.write(message)
    df_collected.to_csv(file_path, sep=',', encoding='utf-8')


def enhance_games(df, game_limit, num_histories, num_players, dict_shelf):
    logging.info('Enhancing games...')
    all_enhanced_games = []
    # for team_name in teams:
    game_count = 0
    df_time_windowed = None
    last_date = None
    for index, row in df.iterrows():
        game_count += 1
        if game_limit is not None and game_limit <= game_count:
            break

        date = index

        row[COLUMN_RESILIENT_DATE] = date

        if date != last_date:
            df_time_windowed = get_windowed_records(df, date, -250)
            last_date = date

        series_enhanced = row

        home_team_name_display = row[COLUMN_NAME]
        home_team_name_encoded = home_team_name_display
        away_team_name_display = row[COLUMN_OPP_NAME]
        away_team_name_encoded = away_team_name_display

        diffs_series_enhanced, is_no_data = get_tm_tot_diffs_with_shelve(df_time_windowed, home_team_name_encoded, away_team_name_encoded, date, num_histories,
                                                                         dict_shelf)
        if is_no_data:
            continue

        player_home_series_enhanced = get_player_rollups_with_shelve(df_time_windowed, row, home_team_name_encoded, date, num_histories, 'home', num_players,
                                                                     dict_shelf)
        player_away_series_enhanced = get_player_rollups_with_shelve(df_time_windowed, row, away_team_name_encoded, date, num_histories, 'away', num_players,
                                                                     dict_shelf)

        series_enhanced = pd.concat([row, diffs_series_enhanced, player_home_series_enhanced, player_away_series_enhanced])

        # for c in series_enhanced.index:
        #     logging.info(c)

        # logging.info(f"home_mean_games_score: {series_enhanced['home_mean_games_score']}")
        # logging.info(f"away_mean_games_score: {series_enhanced['away_mean_games_score']}")

        series_enhanced['diff_mean_games_score'] = (series_enhanced['home_mean_games_score']) - (series_enhanced['away_mean_games_score'])

        assert (series_enhanced[COLUMN_RESILIENT_DATE] == date)

        all_enhanced_games.append(series_enhanced)

    df_collected = convert_series_array_to_df(all_enhanced_games)

    df_collected = df_collected.set_index('resilient_date').sort_index()

    df_enhanced = df_collected.fillna(0)

    logging.info('Enhanced games.')

    return df_enhanced


def add_mean_games_score(away_or_home, series, row):
    positions_columns = [f'{away_or_home}_starting_{p}' for p in POSITIONS]

    starting_lineup_names = [row[big_data_util.COLUMN_SL_1], row[big_data_util.COLUMN_SL_2], row[big_data_util.COLUMN_SL_3], row[big_data_util.COLUMN_SL_4],
                             row[big_data_util.COLUMN_SL_5]]
    add_col = pd.Series(starting_lineup_names, index=positions_columns)

    num_players = 14

    team_games_scores = []
    for i in range(num_players):
        prefix = f'{away_or_home}_mean_{i}_'
        if f'{prefix}PTS' in series.index:
            PTS = series[f'{prefix}PTS']
            FG = series[f'{prefix}FG']
            FGA = series[f'{prefix}FGA']
            FT = series[f'{prefix}FT']
            FTA = series[f'{prefix}FTA']
            ORB = series[f'{prefix}ORB']
            DRB = series[f'{prefix}DRB']
            STL = series[f'{prefix}STL']
            AST = series[f'{prefix}AST']
            BLK = series[f'{prefix}BLK']
            PF = series[f'{prefix}PF']
            TOV = series[f'{prefix}TOV']
            game_score = PTS + 0.4 * FG - 0.7 * FGA - 0.4 * (FTA - FT) + 0.7 * ORB + 0.3 * DRB + STL + 0.7 * AST + 0.7 * BLK - 0.4 * PF - TOV

            series[f'{away_or_home}_{i}_mean_games_score'] = float(game_score)
            team_games_scores.append(game_score)

    series[f'{away_or_home}_mean_games_score'] = float(np.mean(team_games_scores))

    return series.append(add_col)


def get_position_name_map(team_game: pd.Series):
    columns = team_game.index

    map = {}
    for i in range(16):
        col_player_name = f'{i}_name'
        if col_player_name in columns:
            play_name = team_game[col_player_name]
            map[play_name] = team_game[f'{i}_position']

    return map


def get_player_rollups_with_shelve(df_time_windowed, row, team_name_encoded, date, num_histories, home_or_away, num_players, shelf):
    player_shelf_key = f"get_player_rollups_with_shelve::{team_name_encoded}::{date}::{num_histories}::{home_or_away}::{num_players}"
    if shelf is not None and player_shelf_key in shelf.keys():
        player_home_series_enhanced = shelf[player_shelf_key]
    else:
        player_home_series_enhanced = get_player_rollups(df_time_windowed, row, team_name_encoded, date, num_histories, home_or_away, num_players)
        if shelf is not None:
            shelf[player_shelf_key] = player_home_series_enhanced
    return player_home_series_enhanced


def get_tm_tot_diffs_with_shelve(df_time_windowed, home_team_name_encoded, away_team_name_encoded, date, num_histories, shelf):
    diff_shelf_key = f"get_team_tot_diffs_shelf::{home_team_name_encoded}::{away_team_name_encoded}::{date}::{num_histories}"
    if shelf is not None and diff_shelf_key in shelf.keys():
        diffs_series_enhanced, is_no_data = shelf[diff_shelf_key]
    else:
        logging.info(f'Key not found {diff_shelf_key}.')
        diffs_series_enhanced, is_no_data = get_team_tot_diffs(df_time_windowed, home_team_name_encoded, away_team_name_encoded, date, num_histories)
        if shelf is not None:
            shelf[diff_shelf_key] = diffs_series_enhanced, is_no_data
    return diffs_series_enhanced, is_no_data


def get_player_rollups(df, row, team_name, date_game, num_past_games, home_or_away, num_top_players):
    df = df.reset_index(level=['date']).set_index('date').sort_index()

    df_team_games = df.loc[(df[COLUMN_NAME] == team_name) & (df.index < date_game)]

    num_games_found = len(df_team_games)
    num_to_take = num_past_games if num_games_found >= num_past_games else num_games_found
    df_team_games = df_team_games.iloc[:num_to_take]

    mean_rollup = pd.Series()
    for i in range(num_top_players):
        prefix = f'{i}_'
        player_cols = [c for c in df_team_games.columns if prefix in c]

        for col in player_cols:
            if df_team_games[col].dtype != 'object':
                mean_rollup[f'{home_or_away}_mean_{col}'] = df_team_games[col].mean()

    with_starting_lineup_series = add_mean_games_score(home_or_away, mean_rollup, row)

    return with_starting_lineup_series


def get_team_tot_diffs(df, home_team_name_encoded, away_team_name_encoded, date, num_histories):
    home_team_tots, home_win_streak, home_loss_streak, home_win_ratio = get_team_rollups(df, home_team_name_encoded, date, num_histories, 'home')
    away_team_tots, away_win_streak, away_loss_streak, away_win_ratio = get_team_rollups(df, away_team_name_encoded, date, num_histories, 'away')

    is_no_data = home_team_tots.dropna().empty is True or away_team_tots.dropna().empty is True

    tot_diffs = pd.Series()
    if is_no_data is False:
        token_home_prefix = 'mean_home_tot_'
        token_away_prefix = 'mean_away_tot_'
        suffixes = [x[len(token_home_prefix):] for x in home_team_tots.index if x.startswith(token_home_prefix)]

        for suff in suffixes:
            column_home = token_home_prefix + suff
            home_mean = home_team_tots[column_home]
            away_mean = away_team_tots[token_away_prefix + suff]
            tot_diffs[f'mean_diffs_{suff}'] = home_mean - away_mean

    # num_wins = (df.loc[df[COLUMN_WIN_LABEL] == True]).shape[0]
    # win_ratio = num_wins / df.shape[0]
    tot_diffs[COLUMN_HOME_WIN_STREAK] = home_win_streak
    tot_diffs[COLUMN_AWAY_WIN_STREAK] = away_win_streak
    tot_diffs[COLUMN_HOME_AWAY_DIFF_STREAK] = home_win_streak - away_win_streak
    tot_diffs[COLUMN_HOME_LOSS_STREAK] = home_loss_streak
    tot_diffs[COLUMN_HOME_WIN_RATIO] = home_win_ratio
    tot_diffs[COLUMN_AWAY_LOSS_STREAK] = away_loss_streak
    tot_diffs[COLUMN_AWAY_WIN_RATIO] = away_win_ratio

    return tot_diffs, is_no_data


def get_team_rollups(df, team_name, date_game, num_past_games, home_or_away):
    df = df.reset_index(level=['date']).set_index('date').sort_index()

    df_team_games = df[(df[COLUMN_NAME] == team_name) & (df.index < date_game)]

    num_games_found = len(df_team_games)

    # logging.info(f'Found {num_games_found} games.')

    num_to_take = num_past_games if num_games_found >= num_past_games else num_games_found
    df_team_games = df_team_games.iloc[:num_to_take]

    df_team_games[f'tot_{COLUMN_MARG_VICT_ATS}'] = df_team_games[COLUMN_MARG_VICT_ATS].copy()
    df_team_games[f'tot_{COLUMN_DID_WIN}'] = df_team_games[COLUMN_DID_WIN].copy()
    df_team_games[f'tot_{COLUMN_OPEN_VS_CLOSE_SPREAD}'] = df_team_games[COLUMN_OPEN_VS_CLOSE_SPREAD].copy()

    game_totals_cols = [c for c in df_team_games.columns if f'tot_' in c]

    # logging.info(f'Columns: {game_totals_cols}')
    # raise Exception('foo')

    win_streak = 0
    loss_streak = 0
    win_ratio = 0
    new_series = pd.Series()
    if len(df_team_games) > 0:
        for col in game_totals_cols:
            new_series[f'mean_{home_or_away}_{col}'] = df_team_games[col].mean()

        win_streak, loss_streak, win_ratio = get_winning_streak(df_team_games)

    return new_series, win_streak, loss_streak, win_ratio


def get_winning_streak(df_team_games):
    num_wins = (df_team_games.loc[df_team_games[COLUMN_WIN_LABEL] == True]).shape[0]
    win_ratio = num_wins / df_team_games.shape[0]

    win_streak = get_streak(df_team_games, True)
    loss_streak = get_streak(df_team_games, False)

    return win_streak, loss_streak, win_ratio


def get_streak(df_team_games, is_wins=True):
    df_team_games = df_team_games.iloc[::-1]  # Reverse
    count = 0
    for index, row in df_team_games.iterrows():
        if row[COLUMN_WIN_LABEL] == is_wins:
            count += 1
        else:
            break
    return count


def label_encode_columns(df, num_players):
    logging.info('Encoding columns ...')
    le_player_name = LabelEncoder()
    le_colleges = LabelEncoder()
    le_country_origins = LabelEncoder()
    le_game_type = LabelEncoder()
    le_season = LabelEncoder()
    le_country = LabelEncoder()
    le_code = LabelEncoder()
    le_team_rest_days = LabelEncoder()

    team_names = np.concatenate((df[COLUMN_NAME].values, df[COLUMN_OPP_NAME].values), axis=0)

    le_team_name.fit(team_names)
    le_transform_df_col(COLUMN_NAME, df, le_team_name)
    le_transform_df_col(COLUMN_OPP_NAME, df, le_team_name)

    df[COLUMN_GAME_TYPE] = le_game_type.fit_transform(df[COLUMN_GAME_TYPE])
    df[COLUMN_SEASON] = le_season.fit_transform(df[COLUMN_SEASON])
    df[COLUMN_GAME_CODE] = le_code.fit_transform(df[COLUMN_GAME_CODE])
    df[COLUMN_COUNTRY] = le_country.fit_transform(df[COLUMN_COUNTRY])
    df[big_data_util.COLUMN_TEAM_REST_DAYS] = le_team_rest_days.fit_transform(df[big_data_util.COLUMN_TEAM_REST_DAYS])

    player_names = []
    positions = []
    colleges = []
    country_origins = []

    player_names = fix_append_col_values(df, COLUMN_SL_1, player_names, UNKNOWN_VALUE)
    player_names = fix_append_col_values(df, COLUMN_SL_2, player_names, UNKNOWN_VALUE)
    player_names = fix_append_col_values(df, COLUMN_SL_3, player_names, UNKNOWN_VALUE)
    player_names = fix_append_col_values(df, COLUMN_SL_4, player_names, UNKNOWN_VALUE)
    player_names = fix_append_col_values(df, COLUMN_SL_5, player_names, UNKNOWN_VALUE)

    for i in range(num_players):
        df = df.replace({f'{i}_name': {0: ''}})
        player_names = fix_append_col_values(df, f'{i}_name', player_names, UNKNOWN_VALUE)

        df = df.replace({f'{i}_position': {0: ''}})
        positions = fix_append_col_values(df, f'{i}_position', positions, 'SF')

        df = df.replace({f'{i}_college': {0: ''}})
        colleges = fix_append_col_values(df, f'{i}_college', colleges, UNKNOWN_VALUE)

        df = df.replace({f'{i}_blank': {0: ''}})
        country_origins = fix_append_col_values(df, f'{i}_blank', country_origins, UNKNOWN_VALUE)

    player_names = fix_append_starting_position(df, 'home', player_names)
    player_names = fix_append_starting_position(df, 'away', player_names)

    player_names = np.unique(player_names)
    le_player_name.fit(player_names)

    positions = np.unique(positions)
    le_positions.fit(positions)

    colleges = np.unique(colleges)
    le_colleges.fit(colleges)

    country_origins = np.unique(country_origins)
    le_country_origins.fit(country_origins)

    le_transform_df_col(COLUMN_SL_1, df, le_player_name)
    le_transform_df_col(COLUMN_SL_2, df, le_player_name)
    le_transform_df_col(COLUMN_SL_3, df, le_player_name)
    le_transform_df_col(COLUMN_SL_4, df, le_player_name)
    le_transform_df_col(COLUMN_SL_5, df, le_player_name)

    for i in range(num_players):
        le_transform_df_col(f'{i}_name', df, le_player_name)

        le_transform_df_col(f'{i}_position', df, le_positions)

        le_transform_df_col(f'{i}_college', df, le_colleges)

        le_transform_df_col(f'{i}_blank', df, le_country_origins)

    for p in POSITIONS:
        amble = '_starting_'
        le_transform_df_col(f'home{amble}{p}', df, le_player_name)
        le_transform_df_col(f'away{amble}{p}', df, le_player_name)

    logging.info('Label encoded columns.')

    return df


def fix_append_starting_position(df, home_or_away, player_names):
    for p in POSITIONS:
        column = f'{home_or_away}_starting_{p}'
        player_names = fix_append_col_values(df, column, player_names, UNKNOWN_VALUE)
    return player_names


def le_transform_df_col(column_name, df, le):
    column_values = df[column_name].values
    df[column_name] = le.transform(column_values)


def fix_append_col_values(df, column_name, accumulator, replacement_value):
    if column_name in df.columns:
        df[column_name] = df[column_name].fillna(replacement_value)
        col_values = df[column_name].values
        accumulator = np.append(accumulator, col_values, axis=0)

    return accumulator


def get_telling_columns(num_players):
    tot_columns_suffix = ['FT',
                          'twoPA',
                          'FG',
                          'DRB',
                          'ORB_pct',
                          'AST',
                          'threePAr',
                          'PF',
                          'FGA',
                          'DRBr',
                          'twoP',
                          'ORBr',
                          'TOV_pct',
                          'AST_pct',
                          'FTAr',
                          'FIC',
                          'eFG_pct',
                          'FG_pct',
                          'twoPAr',
                          'plus_minus',
                          'USG_pct',
                          'DRtg',
                          'twoP_pct',
                          'DRB_pct',
                          'ORtg',
                          'TRB_pct',
                          'ORB',
                          'threeP',
                          'TOV',
                          'STL_TOV',
                          'TSA',
                          'AST_TOV',
                          'threePA',
                          'BLK_pct',
                          'FT_pct',
                          'PTS',
                          'HOB',
                          'STL',
                          'TRB',
                          'FTA',
                          'BLK',
                          'FTr',
                          'TS_pct',
                          'FT_FGA',
                          'threeP_pct',
                          'STL_pct',
                          big_data_util.COLUMN_OEFF,
                          big_data_util.COLUMN_DEFF,
                          big_data_util.COLUMN_POS,
                          big_data_util.COLUMN_PACE
                          ]
    home_telling_columns = [f'tot_{col}' for col in tot_columns_suffix[:]]
    player_columns_suffix = ['STL_pct',
                             'blank',
                             'FT',
                             'weight',
                             'threeP',
                             'TOV',
                             'STL_TOV',
                             'TSA',
                             'twoPA',
                             'college',
                             'FG',
                             'threePA',
                             'DRB',
                             'ORB_pct',
                             'BLK_pct',
                             'AST_TOV',
                             'position',
                             'AST',
                             'FT_pct',
                             'threePAr',
                             'PF',
                             'PTS',
                             'FGA',
                             'DRBr',
                             'ORBr',
                             'twoP',
                             'STL',
                             'TRB',
                             'TOV_pct',
                             'AST_pct',
                             'FTAr',
                             'FTA',
                             'FIC',
                             'eFG_pct',
                             'BLK',
                             'birth_date',
                             'FG_pct',
                             'twoPAr',
                             'FTr',
                             'plus_minus',
                             'name',
                             'USG_pct',
                             'DRB_pct',
                             'TS_pct',
                             'experience',
                             'height',
                             'twoP_pct',
                             'MP',
                             'DRtg',
                             'ORtg',
                             'TRB_pct',
                             'FT_FGA',
                             'ORB',
                             'threeP_pct',
                             'HOB']
    away_player_telling_columns = []
    home_player_telling_columns = []

    for i in range(num_players):
        home_player_telling_columns += [f'{i}_{col}' for col in player_columns_suffix[:]]

    telling_columns = [COLUMN_MARG_VICT_ATS, COLUMN_OPEN_VS_CLOSE_SPREAD, COLUMN_DID_WIN, COLUMN_SCORE, COLUMN_OPP_SCORE, COLUMN_DID_WIN_OPEN_SPREAD,
                       COLUMN_DID_WIN_CLOSE_SPREAD] \
                      + home_telling_columns + away_player_telling_columns + home_player_telling_columns

    logging.debug(f'Telling columns: {telling_columns}')

    return telling_columns


def get_team_columns(columns, prefix):
    return [c for c in columns if c.startswith(prefix) or c in COLUMNS_PREFIX_TMP_KEEP]


def pull_out_team_from_games(df, prefix_team_1, prefix_team_2):
    team_1_columns = get_team_columns(df.columns, prefix_team_1)
    team_2_columns = get_team_columns(df.columns, prefix_team_2)

    df[COLUMN_DID_WIN] = (df[f'{prefix_team_1}score'] - df[f'{prefix_team_2}score'] > 0)

    df_team_1 = df[team_1_columns]
    df_team_2 = df[team_2_columns]

    df_team_1 = swap_team_columns(df_team_1, prefix_team_1, prefix_team_2)
    df_team_2 = swap_team_columns(df_team_2, prefix_team_2, prefix_team_1)

    all_team_columns = team_1_columns + team_2_columns + COLUMNS_PREFIX_TMP_KEEP
    df_team_naked = df.drop(all_team_columns, axis=1)

    df_team_1_ren = rename_columns(df_team_1, prefix_team_1)
    df_team_2_ren = rename_columns(df_team_2, prefix_team_2)

    is_home = False
    if prefix_team_1 == 'home_':
        is_home = True

    df_team_1_ren[COLUMN_IS_HOME_GAME] = is_home
    df_team_2_ren[COLUMN_IS_HOME_GAME] = not is_home

    df_team_naked = df_team_naked.reset_index()

    df_team_1_ren = df_team_1_ren.reset_index()
    df_team_1_combined = pd.concat([df_team_1_ren, df_team_naked], axis=1, ignore_index=False)

    df_team_2_ren = df_team_2_ren.reset_index()
    df_team_2_combined = pd.concat([df_team_2_ren, df_team_naked], axis=1, ignore_index=False)

    df_team_2_combined[COLUMN_DID_WIN] = ~df_team_1_combined[COLUMN_DID_WIN]

    df_both = pd.concat([df_team_1_combined, df_team_2_combined], axis=0)

    return add_distance_data(df_both)


def swap_team_columns(df, prefix_team, prefix_opp_team):
    df[COLUMN_SCORE] = df[f'{prefix_team}score']
    df[COLUMN_NAME] = df[f'{prefix_team}name']
    df[COLUMN_OPP_SCORE] = df[f'{prefix_opp_team}score']
    df[COLUMN_OPP_NAME] = df[f'{prefix_opp_team}name']
    df = df.drop(COLUMNS_PREFIX_TMP_KEEP, axis=1)
    return df


def rename_columns(df, prefix):
    ren_columns = {}
    for col in df.columns:
        ren_columns[col] = rename_column(col, prefix)

    df_team_1_ren = df.rename(index=str, columns=ren_columns)

    return df_team_1_ren


def rename_column(original_column, prefix):
    result = original_column
    if original_column.startswith(prefix):
        result = original_column[len(prefix):]

    return result


def convert_to_seconds(str_time):
    result = 0
    if str_time is not None and len(str_time) > 0:
        elements = str_time.split(':')

        hours = int(elements[0]) * 60 * 60
        minutes = int(elements[1]) * 60
        seconds = int(elements[2])

        result = hours + minutes + seconds

    return result


VF_CONVERT_GAME_TIME = np.vectorize(convert_to_seconds)


def convert_game_time_to_seconds(df, column_name):
    df[column_name] = VF_CONVERT_GAME_TIME(df[column_name])

    logging.info('Converted game time to seconds.')

    return df


def get_windowed_records(df, date, days):
    rear_window = datetime.datetime.fromtimestamp(date) + datetime.timedelta(days=days)

    return df.query(f'date > {rear_window.timestamp()} & date <= {date}')
