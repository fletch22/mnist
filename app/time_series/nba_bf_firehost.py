import logging
import os

import numpy as np
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler

from app.config import config
from app.time_series import nba_bf_firehost_features, plain_classifiers
from app.time_series.prep import series_to_supervised
from app.util import object_utils

RANDOM_SEED = 42
time_scaler = MinMaxScaler(feature_range=(0, 1))

le_team_name = LabelEncoder()

# data_file_suffix = '2011'
data_file_suffix = '2002-2012'

UNKNOWN_VALUE = 'UNKNOWN'
COLUMN_HOME_DID_WIN = 'home_did_win'

def learn(game_limit=None):
    df = get_dataframe()

    df[COLUMN_HOME_DID_WIN] = ((df['home_score'] - df['away_score']) > 0)

    # df = balance_data(df)

    num_teams = 30
    num_histories = 14
    num_players = 10
    num_best_features = -15

    telling_columns = get_columns(num_players)
    telling_columns.append(COLUMN_HOME_DID_WIN)

    df = df[['away_name', 'home_name'] + telling_columns]

    teams = df['home_name']
    logging.info(f'Num teams: {len(teams)}')

    label_encode_columns(df, num_players)

    df_collected = enhance_games(df, game_limit, num_histories, num_players)

    # logging.info(f'cols: {df_collected.columns}')
    if num_best_features > 0:
        df_collected = keep_best_columns(df_collected, num_best_features)

    save_to_csv(df_collected, num_histories, num_players)

    slice_and_score(df_collected, telling_columns)


def learn_from_preprocessed_file(file_path, num_players, columns_to_keep=None):
    telling_columns = get_columns(num_players)
    telling_columns.append(COLUMN_HOME_DID_WIN)

    df = pd.read_csv(file_path, delimiter=',')

    df = balance_data(df)

    if columns_to_keep is not None:
        columns_to_drop = [x for x in df.columns if x not in columns_to_keep]

        logging.info(f'tot: {len(df.columns)}; c2d: {len(columns_to_drop)}')

        columns_to_drop = list(set(telling_columns + columns_to_drop))
    else:
        columns_to_drop = telling_columns

    slice_and_score(df, columns_to_drop)


def save_to_csv(df_collected, num_histories, num_players):
    file_path = os.path.join(config.SESSION_DIR, 'df_collected.csv')

    message = f'''Saving df_collected to {file_path} using 
    with attributes: date_file_suffix:{data_file_suffix}; num_games:{df_collected.shape[0]};
    num_histories={num_histories}; num_players:{num_players}; columns:{df_collected.columns}; '''

    logging.info(message)

    os.makedirs(config.SESSION_DIR, exist_ok=True)

    meta_file_path = os.path.join(config.SESSION_DIR, 'df_collected_meta.txt')
    with open(meta_file_path, "w") as f:
        f.write(message)
    df_collected.to_csv(file_path, sep=',', encoding='utf-8')


def slice_and_score(df_collected, columns_to_drop):
    X, y, train_X, train_y, test_X, test_y, x_columns = split_train_test(df_collected, columns_to_drop)

    standard_scaler = StandardScaler()
    train_X = standard_scaler.fit_transform(train_X)
    # train_X = standard_scaler.transform(train_X)

    test_X = standard_scaler.transform(test_X)
    # test_X = standard_scaler.transform(test_X)

    # logging.info(f'train_X shape: {train_X.shape}')
    # logging.info(f'test_X shape: {test_X.shape}')
    # dnn(200, train_X, train_y, test_X, test_y)
    # stratified_kfold(batch_size, create_baseline, train_X, train_y)
    plain_classifiers.score_plain_classifiers(train_X, train_y, test_X, test_y)
    # nba_pca.show_pca(train_X, train_y, test_X, test_y)
    # logits_features_score(X, y, x_columns)
    # logits_ridge_selection(X, y, x_columns)


def label_encode_columns(df, num_players):
    le_player_name = LabelEncoder()
    le_positions = LabelEncoder()
    le_colleges = LabelEncoder()
    le_country_origins = LabelEncoder()

    names = []
    positions = []
    colleges = []
    country_origins = []
    for i in range(num_players):
        names = fix_append_col_values(df, f'away_{i}_name', names, UNKNOWN_VALUE)
        names = fix_append_col_values(df, f'home_{i}_name', names, UNKNOWN_VALUE)

        positions = fix_append_col_values(df, f'away_{i}_position', positions, 'SF')
        positions = fix_append_col_values(df, f'home_{i}_position', positions, 'SF')

        colleges = fix_append_col_values(df, f'away_{i}_college', colleges, UNKNOWN_VALUE)
        colleges = fix_append_col_values(df, f'home_{i}_college', colleges, UNKNOWN_VALUE)

        country_origins = fix_append_col_values(df, f'away_{i}_blank', country_origins, UNKNOWN_VALUE)
        country_origins = fix_append_col_values(df, f'home_{i}_blank', country_origins, UNKNOWN_VALUE)

    names = np.unique(names)
    le_player_name.fit(names)

    positions = np.unique(positions)
    le_positions.fit(positions)

    colleges = np.unique(colleges)
    le_colleges.fit(colleges)

    country_origins = np.unique(country_origins)
    le_country_origins.fit(country_origins)

    for i in range(num_players):
        le_transform_df_col(f'away_{i}_name', df, le_player_name)
        le_transform_df_col(f'home_{i}_name', df, le_player_name)

        le_transform_df_col(f'away_{i}_position', df, le_positions)
        le_transform_df_col(f'home_{i}_position', df, le_positions)

        le_transform_df_col(f'away_{i}_college', df, le_colleges)
        le_transform_df_col(f'home_{i}_college', df, le_colleges)

        le_transform_df_col(f'away_{i}_blank', df, le_country_origins)
        le_transform_df_col(f'home_{i}_blank', df, le_country_origins)


def balance_data(df):
    df_wins = df[df[COLUMN_HOME_DID_WIN] == 1]
    df_losses = df[df[COLUMN_HOME_DID_WIN] == 0]

    diff = df_wins.shape[0] - df_losses.shape[0]
    df_has_more = df_wins if diff > 0 else df_losses
    df_has_less = df_wins if diff < 0 else df_losses

    df_has_less = df_has_less.sample(frac=1)
    df_has_less = pd.concat([df_has_less, df_has_less[:diff]], axis=0)

    df = pd.concat([df_has_more, df_has_less, df_has_less[:diff]], axis=0)

    # df = pd.concat([df_wins[:df_losses.shape[0]], df_losses], axis=0)

    return df.sort_index()


def keep_best_columns(df, num_top_cols):
    columns = get_col_names_from_best_features(num_top_cols)

    columns += ['home_score', 'away_score', 'away_name', 'home_name', COLUMN_HOME_DID_WIN]

    drop_cols = [c for c in df.columns if c not in columns]

    return df.drop(drop_cols, axis=1)


def get_col_names_from_best_features(num_top_cols):
    features = nba_bf_firehost_features.features
    num_top_cols = len(features) if num_top_cols > len(features) else num_top_cols
    columns = [tup[1] for tup in features[:num_top_cols]]
    return columns


def logits_features_score(X, y, columns):
    X_scaled = StandardScaler().fit_transform(X)
    clf = LogisticRegression(penalty='l1', C=0.1)
    clf.fit(X_scaled, y)
    zero_feat = []
    nonzero_feat = []
    num_features = X.shape[1]
    # type(clf.coef_)
    for i in range(num_features):
        coef = clf.coef_[0, i]
        if coef == 0:
            zero_feat.append(columns[i])
        else:
            nonzero_feat.append((abs(coef), columns[i]))
    print('Features that have coeffcient of 0 are: ', zero_feat)
    print('Features that have non-zero coefficients are:')
    print(sorted(nonzero_feat, reverse=True))


def le_transform_df_col(column_name, df, le_player_name):
    column_values = df[column_name].values
    df[column_name] = le_player_name.transform(column_values)


def fix_append_col_values(df, column_name, accumulator, replacement_value):
    df[column_name] = df[column_name].fillna(replacement_value)
    col_values = df[column_name].values
    accumulator = np.append(accumulator, col_values, axis=0)
    return accumulator


def get_columns(num_players):
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
                          'STL_pct'
                          ]
    away_telling_columns = [f'away_tot_{col}' for col in tot_columns_suffix[:]]
    home_telling_columns = [f'home_tot_{col}' for col in tot_columns_suffix[:]]
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
        away_player_telling_columns += [f'away_{i}_{col}' for col in player_columns_suffix[:]]
        home_player_telling_columns += [f'home_{i}_{col}' for col in player_columns_suffix[:]]
    # 3: 0.6656580937972768; 6: 0.6656580937972768
    # 'away_tot_FT', 'away_tot_twoPA', 'away_tot_FG'
    telling_columns = ['home_score', 'away_score'] + away_telling_columns + home_telling_columns + away_player_telling_columns + home_player_telling_columns
    logging.debug(f'Telling columns: {telling_columns}')

    return telling_columns


def enhance_games(df, game_limit, num_histories, num_players):
    all_enhanced_games = []
    # for team_name in teams:
    game_count = 0
    for index, row in df.iterrows():
        game_count += 1
        if game_limit is not None and game_limit <= game_count:
            break

        team_name_encoded = int(row["home_name"])
        date = index
        row['resilient_date'] = date

        logging.debug(f'date: {date}')
        team_name = list(le_team_name.inverse_transform([team_name_encoded]))[0]
        logging.info(f'Getting previous games for team {team_name} on {object_utils.get_display_date(date)} ({date})')

        series_enhanced = row
        series_enhanced, is_no_data = get_team_tot_diffs(df, row, team_name_encoded, date, num_histories)

        if is_no_data:
            continue

        series_enhanced = get_player_rollups(df, series_enhanced, team_name_encoded, date, num_histories, 'away', num_players)
        series_enhanced = get_player_rollups(df, series_enhanced, team_name_encoded, date, num_histories, 'home', num_players)

        assert (series_enhanced['resilient_date'] == date)

        all_enhanced_games.append(series_enhanced)

    df_collected = convert_series_array_to_df(all_enhanced_games)
    return df_collected.fillna(0)


def split_train_test(df_collected, columns_to_drop):
    np.set_printoptions(threshold=np.nan)
    pd.set_option('display.max_colwidth', -1)

    y = df_collected[COLUMN_HOME_DID_WIN].values

    remaining_tell_cols = [x for x in columns_to_drop if x in df_collected.columns]
    df_X_naked = df_collected.drop(remaining_tell_cols, axis=1)
    x_columns = df_X_naked.columns
    X = df_X_naked.values

    train_prop = np.math.floor(len(df_collected) * .7)
    df_X_raw = df_collected.iloc[:train_prop]
    # df_X_raw = df_X_raw.sample(frac=1)

    train_y = df_X_raw[COLUMN_HOME_DID_WIN].values
    df_train_X = df_X_raw.drop(remaining_tell_cols, axis=1)

    # train_prop + 200
    df_X_test = df_collected.iloc[train_prop:]

    # df_print = df_X_test[['resilient_date', 'away_score', 'home_score']]
    # logging.info(f'df_X_test: {df_print.values}')
    # logging.info(f'df_X_test: {df_X_test.head()}')

    test_y = df_X_test[COLUMN_HOME_DID_WIN].values

    df_winnowed = df_X_test.drop(remaining_tell_cols, axis=1)
    test_X = df_winnowed.values

    logging.info(f'Len test_X: {test_X.shape[0]}')

    logging.info(f'Len df_train_X: {df_train_X.shape[0]}')

    return X, y, df_train_X.values, train_y, test_X, test_y, x_columns


def convert_series_array_to_df(all_enhanced_games):
    logging.info(f'Columns: {all_enhanced_games[0].index}')
    df_collected = pd.DataFrame(columns=all_enhanced_games[0].index)
    for x in all_enhanced_games:
        df_collected = df_collected.append(x)

    df_collected = df_collected.set_index('resilient_date').sort_index()

    return df_collected


def get_team_tot_diffs(df, row, team_name_encoded, date, num_histories):
    home_team_tots = get_team_rollups(df, team_name_encoded, date, num_histories, 'home')
    away_team_tots = get_team_rollups(df, team_name_encoded, date, num_histories, 'away')

    is_no_data = home_team_tots.dropna().empty is True or away_team_tots.dropna().empty is True

    if is_no_data is False:
        token_home_prefix = 'mean_home_tot_'
        token_away_prefix = 'mean_away_tot_'
        suffixes = [x[len(token_home_prefix):] for x in home_team_tots.index if x.startswith(token_home_prefix)]

        for suff in suffixes:
            home_mean = home_team_tots[token_home_prefix + suff]
            away_mean = away_team_tots[token_away_prefix + suff]
            row[f'mean_diffs_{suff}'] = home_mean - away_mean

    return row, is_no_data


def get_series_prepped(df, le_team_names, num_histories):
    df_all_enhanced_games = None
    # for team_name in teams:
    for index, row in df.iterrows():
        team_name_encoded = int(row["home_name"])
        date = index

        logging.debug(f'date: {date}')
        team_name = list(le_team_names.inverse_transform([team_name_encoded]))[0]
        logging.info(f'Getting previous games for team {team_name} on {object_utils.get_display_date(date)} ({date})')

        df_current_team_records = df.loc[(df['home_name'] == team_name_encoded) & (df.index <= date)]

        # logging.debug(f'df_current: {len(df_current_team_records)}')
        # logging.debug(f'Num team records: {df_current_team_records.shape[0]}')
        # logging.debug(f'df_single_team.shape: {df_current_team_records.shape}')

        if len(df_current_team_records) < num_histories + 1:
            continue

        df_current_team_records = df_current_team_records.reset_index(level=['date']).set_index('date').sort_index(ascending=False)
        df_current_team_records = df_current_team_records.iloc[:num_histories + 1]

        df_current_team_records = df_current_team_records.sort_index(ascending=True)

        # for i in range(len(df_current_team_records)):
        # logging.debug(df_current_team_records.index[i])

        assert (df_current_team_records.index[-1] == date)
        # logging.debug(f'team: {df_current_team_records.head()}')
        # logging.info(f'Num df_current_team_records: {len(df_current_team_records)}')

        df_enhanced_game = prep_for_time_series(df_current_team_records, num_histories)

        assert (df_enhanced_game.index[0] == date)

        assert (len(df_enhanced_game) <= 1)
        if len(df_enhanced_game) == 1:
            assert (df_enhanced_game.index[0] == date)

            # logging.info(f'{df_enhanced_game.head()}')
            # return

            # logging.info(f'date1: {df_enhanced_game["date(t-1)"]}')
            # logging.info(f'date2: {df_enhanced_game["date(t-2)"]}')
            # logging.info(f'date3: {df_enhanced_game["date(t-3)"]}')
            # logging.info(f'date4: {df_enhanced_game["date(t-4)"]}')

        # logging.info(f'Num games: {len(df_enhanced_game)}')

        if df_all_enhanced_games is None:
            df_all_enhanced_games = df_enhanced_game
        else:
            df_all_enhanced_games = pd.concat([df_all_enhanced_games, df_enhanced_game])

    df_all_enhanced_games = df_all_enhanced_games.reset_index(level=['date']).set_index('date').sort_index()

    return


def get_team_rollups(df, team_name, date_game, num_past_games, home_or_away):
    df = df.reset_index(level=['date']).set_index('date').sort_index()

    # df_team_games = df.loc[(df[f'{home_or_away}_name'] == team_name) & (df.index < date_game)]
    df_team_games = df[(df[f'{home_or_away}_name'] == team_name) & (df.index < date_game)]

    num_games_found = len(df_team_games)
    num_to_take = num_past_games if num_games_found >= num_past_games else num_games_found
    df_team_games = df_team_games.iloc[:num_to_take]

    game_totals_cols = [c for c in df_team_games.columns if f'{home_or_away}_tot_' in c]

    new_series = pd.Series()

    if len(df_team_games) > 0:
        for col in game_totals_cols:
            new_series[f'mean_{col}'] = df_team_games[col].mean()

    return new_series


def get_player_rollups(df, series_current_game, team_name, date_game, num_past_games, home_or_away, num_top_players=1):
    df = df.reset_index(level=['date']).set_index('date').sort_index()

    df_team_games = df.loc[(df[f'{home_or_away}_name'] == team_name) & (df.index < date_game)]

    num_games_found = len(df_team_games)
    num_to_take = num_past_games if num_games_found >= num_past_games else num_games_found
    df_team_games = df_team_games.iloc[:num_to_take]

    # logging.info(f'Num top players: {num_top_players}')

    for i in range(num_top_players):

        prefix = f'{home_or_away}_{i}_'
        # logging.info(f'Looking for : {prefix}')
        player_cols = [c for c in df_team_games.columns if prefix in c]

        for col in player_cols:
            if df_team_games[col].dtype != 'object':
                series_current_game[f'mean_{col}'] = df_team_games[col].mean()

    return series_current_game


def stratified_kfold(batch_size, create_baseline, train_X, train_y):
    estimator = KerasClassifier(build_fn=create_baseline, epochs=10, batch_size=batch_size, verbose=1)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    results = cross_val_score(estimator, train_X, train_y, cv=kfold)
    print("Results: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))


def get_train_test_split(X, y, train_size=.7):
    proportion = np.math.floor(len(X) * train_size)
    train_X = X[:proportion]
    train_Y = y[:proportion]
    test_X = X[proportion:]
    test_y = y[proportion:]

    return train_X, train_Y, test_X, test_y


def drop_columns(df, dropping_columns):
    return df.drop(dropping_columns, axis=1)


def fit_label_encoder(df, column_name):
    le = LabelEncoder()
    if column_name in df.columns:
        df[column_name] = le.fit_transform(df[column_name])

    return df, le


def field_encoder(df):
    # df['away_name'] = df['away_name'].astype(str)
    # df['home_name'] = df['home_name'].astype(str)
    data = np.concatenate((df['away_name'].values, df['home_name'].values), axis=0)

    data_unique = np.unique(data)

    logging.debug(f'data.shape: {data_unique.shape}')
    logging.debug(f'data: {data_unique[0:len(data_unique)]}')

    le_team_name.fit(data_unique)

    df['away_name'] = le_team_name.transform(df['away_name'])
    df['home_name'] = le_team_name.transform(df['home_name'])

    return df


def prep_for_time_series(df, num_histories):
    # logging.debug(f'Len Cols: {len(df.columns)}')
    # logging.debug(f'Columns: {df.columns}')
    # logging.info(f'df date index 1: {df.index[0]}')

    df = df.reset_index(level=['date'])
    # logging.info(f'df date index 2: {df["date"].iloc[0]}')

    # index_loc = df.columns.get_loc('date')
    # logging.info(f'Index loc: {index_loc}')
    # logging.info(f'df len: {len(df)}')

    data = df.values
    # logging.debug(f'df date index 3: {data[0, 0]}')

    # data = np.argsort(data, axis=index_loc)

    # logging.debug(f'df date index 4: {data[0, 0]}')
    # return

    reframed = series_to_supervised(data, columns=df.columns, n_in=num_histories, n_out=1, skip_rows=num_histories)

    # logging.info(f'reframed Cols: {len(reframed)}')

    reframed = reframed.set_index('date').sort_index(ascending=False)

    # logging.info(f'reframed 2 Cols: {len(reframed)}')

    if reframed.shape[0] > 0:
        # logging.info(f'df date index 5: {reframed.index[0]}')
        reframed = reframed.iloc[:1]

    # logging.info(f'len reframed: {reframed.shape}')

    return reframed


def calc_return(val_acc, amount_bet, number_bets):
    raw_result = ((val_acc * number_bets) * (amount_bet * 1.91)) - (((1 - val_acc) * number_bets) * amount_bet)
    tot_invest = amount_bet * number_bets
    roi = (raw_result - (tot_invest)) / tot_invest
    return round(roi, 4)


def get_dataframe():
    nba_path = os.path.join(config.DATA_FOLDER_PATH, 'sports', f'games_{data_file_suffix}.csv')

    df = pd.read_csv(nba_path)

    df = field_encoder(df)
    # df = df.fillna(0)

    return df.set_index('date').sort_index()


def get_team_name(team_name_encoded):
    return list(le_team_name.inverse_transform([team_name_encoded]))[0]
