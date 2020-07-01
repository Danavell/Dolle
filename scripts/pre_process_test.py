import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


def load_data(agg_path, sensor_path):

    agg_cols_to_use = [
        'JOBNUM', 'Date', 'Non Duplicate 0102', '0103 Pace',
        '0104 Alarm Time', '0105 Alarm Time', '0106 Alarm Time',
        'Label', '0101 Group', '0103 ID',
    ]

    original = pd.read_csv(
        agg_path, sep=',', usecols=agg_cols_to_use,
        parse_dates=['Date'], infer_datetime_format=True
    )
    agg = original.copy()
    agg = agg.drop(['0101 Group', '0103 ID', 'Date'], axis=1)

    sensor_data = pd.read_csv(sensor_path, sep=',', parse_dates=['Date'], infer_datetime_format=True)
    sensor_data['0101 Group'].fillna(0, inplace=True)
    condition = sensor_data['0101 Group'] > 0
    deac_dates = sensor_data[condition][['0101 Group', 'Date']].copy().dropna()

    deac_times = pd.merge(
        left=original[['Date', '0103 ID', '0101 Group']],
        right=deac_dates,
        left_on='0101 Group',
        right_on='0101 Group',
        how='left',
        suffixes=('', '_deac')
    )
    return agg, deac_times


def shift_deac_dates(deac_times, catch):
    for i in range(1, catch + 1):
        deac_times[f'{i}'] = deac_times['Date_deac'].shift(-i)
    return deac_times


def _to_numpy(agg, label):
    return agg[label] \
        .to_numpy() \
        .reshape(-1, 1)


def _split_apply(x, col, func, *args):
    for group in np.unique(x[:, col]):
        j = x[x[:, col] == group]
        j = func(j, *args)
        o = np.concatenate([o, j]) \
            if 'o' in locals() else j
    return o


def _add_distance_label(x, catch):
    col = x.shape[1]-1
    idx = np.array(np.where(x[:, col] == 1))
    o = np.zeros((x.shape[0], 1))
    if idx.shape[1] > 0:
        for i in range(1, catch+1):
            p_i = idx - i
            mask = p_i[p_i >= 0]
            o[mask] = i
    t = np.concatenate((x, o), axis=1)
    return t


def _pad_stack_sequences(x, rows, columns, three_d=True):
    x = np.concatenate(
        [x[:, 1:], np.zeros([rows-1, x.shape[1]-1])]
    )
    eval_comp = [
        f'x[:-{rows - 1}]'
        if i == 0
        else f'x[{rows - 1}:]' if i == rows - 1
        else f'x[{i}:-{rows - i - 1}]'
        for i in range(rows)
    ]
    eval_str = f"np.hstack(({', '.join(eval_comp)}))"
    t = eval(eval_str)
    return t.reshape(-1, rows, columns) if three_d else t


def _index_X(X, idx):
    length = len(X.shape)
    if length == 3:
        return X[idx, :, :]
    elif length == 2:
        return X[idx, :]
    else:
        raise ValueError('Unsupported number of dimensions passed')


def _split_labels(X, y, i):
    idx = np.where(y == i)[0]
    return _index_X(X, idx), idx


def _concat_y_orig_indices(dy, idx):
    data = np.concatenate(
        [dy.reshape(-1, 1), idx.reshape(-1, 1)], axis=1
    )
    data = pd.DataFrame(data, columns=['pred', 'idx'])

    test_idx = pd.DataFrame(columns=['pred', 'idx'])
    data_tuple = (test_idx, data)
    return pd.concat(data_tuple, axis=0)


def _train_test_deacs(x, d_y, i):
    d_idx = np.where(d_y == i)[0]
    d = d_idx.size
    idx = np.random.choice(
        d_idx, d // 5, replace=False
    )
    deacs = _index_X(x, idx)
    dX_train = np.delete(deacs, idx, axis=0)
    dX_test = _index_X(x, idx)
    dy_train = np.full((dX_train.shape[0], 1), i)
    dy_test = np.full((dX_test.shape[0], 1), i)

    test_idx = _concat_y_orig_indices(dy_test, idx)

    return dX_train, dX_test, dy_train, dy_test, test_idx


def _train_test_non_deacs(X):
    nd = X.shape[0]
    idx = np.random.choice(
        nd, nd // 5, replace=False
    )
    ndX_train = np.delete(X, idx, axis=0)
    ndX_test = _index_X(X, idx)
    dy_train = np.zeros(
        ndX_train.shape[0]
    ).reshape(-1, 1)
    dy_test = np.zeros(
        ndX_test.shape[0]
    ).reshape(-1, 1)
    return ndX_train, ndX_test, dy_train, dy_test


def shuffle_data(y_test, t_idx):
    t_idx.reset_index(drop=True, inplace=True)
    condition = y_test == 0
    zeros_length = len(y_test[condition])

    t_idx.index += zeros_length
    z = pd.DataFrame(
        y_test, columns=['y']
    )
    z['orig_indices'] = np.nan
    z.loc[t_idx.index, 'orig_indices'] = t_idx['idx']

    shuffled = pd.DataFrame(z)\
                 .sample(frac=1, replace=False)
    idx = shuffled.index.to_numpy()
    return shuffled, idx


def _upsample(ndX, dX, i, t_idx=None):
    nds, ds = ndX.shape[0], dX.shape[0]
    diff = nds // ds
    remainder = nds - diff * ds
    X = np.tile(dX, (diff, 1, 1))
    idx = np.random.choice(
        ds, remainder, replace=False
    )
    shape = len(X.shape)
    if shape == 3:
        dX = dX[idx, :, :]
    elif shape == 2:
        dX = dX[idx, :]
    else:
        raise ValueError()

    X = np.concatenate((X, dX), axis=0)
    y = np.full((X.shape[0], 1), i)
    if isinstance(t_idx, pd.DataFrame):
        t_idx = t_idx.to_numpy()
        q = np.tile(t_idx, (diff, 1))
        t_idx = np.concatenate((q, t_idx[idx, :]), axis=0)
        t_idx = pd.DataFrame(t_idx, columns=['pred', 'idx'])
    return X, y, t_idx


def _upsample_no_tts(x, d_y, catch):
    nd, _ = _split_labels(x, d_y, 0)
    for i in range(1, catch + 1):
        d, idx = _split_labels(x, d_y, i)
        i_s = np.full((idx.shape[0], 1), i)
        idx = np.concatenate(
            (i_s, idx.reshape(-1, 1)),
            axis=1
        )
        idx = pd.DataFrame(idx)
        x_i, y_i, idx = _upsample(nd, d, i, idx)
        if 'X' in locals() and 'y' in locals():
            X = np.concatenate((X, x_i), axis=0)
            y = np.concatenate((y, y_i), axis=0)
        else:
            X = np.concatenate((nd, x_i), axis=0)
            y = np.concatenate(
                (np.zeros(nd.shape[0]).reshape(-1, 1), y_i),
                axis=0
            )

        if 't_idx' in locals():
            t_idx = pd.concat([t_idx, idx], axis=0)
        else:
            t_idx = idx

    shuffled, idx = shuffle_data(y, t_idx)
    y = y[idx]
    X = X[idx, :, :]
    shuffled.reset_index(drop=True, inplace=True)
    return X, y, shuffled


def _train_test_split(x, d_y, catch, balance_test=True):
    nd, _ = _split_labels(x, d_y, 0)
    ndX_train, ndX_test, ndy_train, ndy_test = \
        _train_test_non_deacs(nd)
    for i in range(1, catch + 1):
        dX_train, dX_test, dy_train, dy_test, test_idx = \
            _train_test_deacs(x, d_y, i)
        xtr, ytr, _ = _upsample(ndX_train, dX_train, i)

        if 'X_train' in locals() and 'y_train' in locals():
            X_train = np.concatenate((X_train, xtr))
            y_train = np.concatenate((y_train, ytr))
        else:
            X_train = np.concatenate((ndX_train, xtr))
            y_train = np.concatenate((ndy_train, ytr))

        if balance_test:
            dX_test, dy_test, test_idx = _upsample(
                ndX_test, dX_test, i, test_idx
            )

        if 'X_test' in locals() and 'y_test' in locals():
            X_test = np.concatenate((X_test, dX_test))
            y_test = np.concatenate((y_test, dy_test))
        else:
            X_test = np.concatenate((ndX_test, dX_test))
            y_test = np.concatenate((ndy_test, dy_test))

        if 'y_test_idx' in locals():
            y_test_idx = pd.concat([y_test_idx, test_idx], axis=0)
        else:
            y_test_idx = test_idx

    shuffled, idx = shuffle_data(y_test, y_test_idx)

    y_test = y_test[idx]
    X_test = X_test[idx, :, :]
    shuffled.reset_index(drop=True, inplace=True)
    return X_train, y_train, X_test, y_test, shuffled


def prep_data(agg, deac_times, catch, rows, columns, scaler=None, three_d=True):
    deac_times = shift_deac_dates(deac_times, catch)

    x = agg.to_numpy()
    x = _split_apply(x, 0, _add_distance_label, catch)
    d_y = x[:, -1].reshape(-1, 1)
    x = x[:, :-1]

    if not isinstance(scaler, MinMaxScaler):
        scaler = MinMaxScaler()

    scaler.fit(x)
    x = scaler.transform(x)

    x = _split_apply(
        x, 0, _pad_stack_sequences, rows, columns, three_d
    )
    return x, d_y, deac_times, scaler


def flatten_idxmax_y(y_pred, y_test):
    y_pred_pd = pd.DataFrame(y_pred, index=y_test.index)
    y_pred_1D = y_pred_pd.idxmax(axis=1)
    y_test_1D = pd.DataFrame(y_test).idxmax(axis=1)
    return y_pred_1D, y_test_1D


def calc_time_until_deac(catch, test_idx, y_pred_1D, deac_times):
    q = pd.merge(
        left=test_idx,
        right=pd.DataFrame(y_pred_1D),
        left_index=True,
        right_index=True,
        how='left'
    )
    q = q.dropna()
    q.drop_duplicates(keep='first', inplace=True)
    q.set_index('orig_indices', inplace=True)
    columns = {
        'y': 'y_true',
        0: 'y_pred'
    }
    q.rename(columns=columns, inplace=True)
    time_matrix = pd.merge(
        left=deac_times,
        right=q,
        left_index=True,
        right_index=True,
        how='left'
    )

    for i in range(1, catch + 1):
        condition = (time_matrix['y_true'] == i) & \
                    (time_matrix['y_pred'] == i)
        a = time_matrix[condition]
        t = (a[f'{i}'] - a['Date']).dt.total_seconds()
        t = pd.DataFrame(t, columns=['prediction time'])
        if 'prediction time' in q.columns:
            q.loc[t.index, 'prediction time'] = t['prediction time']
        else:
            q = pd.merge(
                left=q,
                right=t,
                left_index=True,
                right_index=True,
                how='left'
            )

    q = q[['y_true', 'prediction time']].sort_index().dropna()
    return q.to_numpy(), time_matrix


def pre_process_no_tts(agg_path, sensor_path, catch=2, rows=3,
                       columns=6, scaler=None, balance_classes=True,
                       three_d=True
                       ):
    agg, deac_times = load_data(agg_path, sensor_path)
    deac_times = shift_deac_dates(deac_times, catch)

    x, d_y, deac_times, scaler = prep_data(
        agg, deac_times, catch, rows, columns, scaler, three_d
    )

    if balance_classes:
        x, d_y, test_idx = _upsample_no_tts(x, d_y, catch)
    else:
        test_idx = np.concatenate(
            (d_y.reshape(-1, 1), np.arange(d_y.shape[0]).reshape(-1, 1)),
            axis=1
        )
        test_idx = pd.DataFrame(test_idx, columns=['y', 'orig_indices'])

    y = pd.get_dummies(d_y.reshape(-1))

    meta = dict()
    meta['scaler'] = scaler
    meta['deac_times'] = deac_times
    meta['test_idx'] = test_idx
    return x, y, meta


def pre_process(agg_path, sensor_path, catch=2, rows=3,
                columns=6, balance_test=True, scaler=None,
                three_d=True
                ):
    agg, deac_times = load_data(agg_path, sensor_path)
    deac_times = shift_deac_dates(deac_times, catch)

    x, d_y, deac_times, scaler = prep_data(
        agg, deac_times, catch, rows, columns, scaler, three_d
    )
    X_train, y_train, X_test, y_test, test_idx = \
        _train_test_split(x, d_y, catch, balance_test=balance_test
    )

    y_train = pd.get_dummies(y_train.reshape(-1))
    y_test = pd.get_dummies(y_test.reshape(-1))

    meta = dict()
    meta['scaler'] = scaler
    meta['deac_times'] = deac_times
    meta['test_idx'] = test_idx
    return X_train, y_train, X_test, y_test, meta


# model = Sequential()
# model.add(LSTM(
#     units=3, input_shape=(rows, columns), return_sequences=True, dropout=0.2, recurrent_dropout=0.2
# ))
# model.add(LSTM(3, return_sequences=False))
# model.add(Dense(catch + 1, activation='sigmoid'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
# mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
# reduce_lr_loss = ReduceLROnPlateau(
#     monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min'
# )
# history = model.fit(
#     X_train, y_train, batch_size=32, epochs=100,
#     validation_data=(X_test, y_test), shuffle=False,
#     callbacks=[
#         earlyStopping, mcp_save, reduce_lr_loss
#     ],
#     class_weight={0: 1, 1: 0.2, 2: 0.2}
# )