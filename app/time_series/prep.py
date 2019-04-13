from pandas import DataFrame, concat
import logging


def series_to_supervised(data, columns=None, n_in=1, n_out=1, dropnan=True, skip_rows=9):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        # names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        names += [f'{c}(t-{i})' for c in columns]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            # names += [f'{columns[j]}%d(t)' % (j + 1) for j in range(n_vars)]
            names += [f'{columns[j]}' for j in range(n_vars)]
        else:
            # names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
            names += [f'{columns[j]}(t+{i}' for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names

    logging.debug(f'agg shape: {agg.shape}')

    # drop rows with NaN values
    if dropnan:
        # agg.dropna(inplace=True)
        agg = agg[skip_rows:]
    return agg
