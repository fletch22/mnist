import datetime
import logging

import pandas as pd

UNSET = -1.234567890

spelled_numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']


def replace_illegal(key):
    result = key.replace("%", "_pct")
    result = result.replace("/", "_")
    result = result.replace("+", "plus")
    result = result.replace("-", "minus")

    for i in range(10):
        result = result.replace(f'{i}', f'{spelled_numbers[i]}')

    if key == "":
        result = "blank"

    return result


def fix_illegal_keys(dict_thing):
    new_dict = {}
    for key in dict_thing.keys():
        value = dict_thing[key]
        new_key = replace_illegal(key)
        if type(value) is dict:
            result_dict = fix_illegal_keys(value)
            new_dict[new_key] = result_dict
        else:
            new_dict[new_key] = value
    return new_dict


def get_attrs_from_obj(obj):
    return [a for a in dir(obj) if not a.startswith('__') and not callable(getattr(obj, a))]


def to_dict(instance):
    attrs = get_attrs_from_obj(instance)

    d = {}
    for a in attrs:
        d[a] = instance.__getattribute__(a)

    return d


def get_display_date(date):
    return datetime.datetime.utcfromtimestamp(date).strftime("%Y-%m-%d")


def convert_series_array_to_df(series_array):
    df = pd.DataFrame(columns=series_array[0].index)
    for x in series_array:
        df = df.append(x, ignore_index=True)

    return df


def keep_cols(df, cols_to_keep):
    cols_to_drop = [c for c in df.columns if c not in cols_to_keep]

    return df.drop(cols_to_drop, axis=1)
