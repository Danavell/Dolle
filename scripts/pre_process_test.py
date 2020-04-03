import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler


def load_data():
    aggregate_path = r'/home/james//Documents/DolleProject/dolle_csvs/28-02-16 to 2018-12-19' \
                     r'/MLAgg0103 1405: 1 SW, 3 CF, no overlaps/SW-3D-3F-3B-12T.csv'

    agg_cols_to_use = [
        'JOBNUM', 'Date', 'Non Duplicate 0102', '0103 Pace',
        '0104 Alarm Time', '0105 Alarm Time', '0106 Alarm Time',
        'Label', '0101 Group', '0103 ID',
    ]

    original = pd.read_csv(
        aggregate_path, sep=',', usecols=agg_cols_to_use,
        parse_dates=['Date'], infer_datetime_format=True
    )
    agg = original.copy()
    agg = agg.drop(['0101 Group', '0103 ID', 'Date'], axis=1)

    sensor_path = r'/home/james//Documents/DolleProject/dolle_csvs/28-02-16 to 2018-12-19' \
                  r'/MLAgg0103 1405: 1 SW, 3 CF, no overlaps/sensor_data.csv'

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


def _pad_dstack_sequences(x, rows, columns):
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
    eval_str = f"np.hstack(({', '.join(eval_comp)}))" \
                 f".reshape(-1, {rows}, {columns})"
    return eval(eval_str)


def _split_labels(X, y, i):
    idx = np.where(y == i)[0]
    return X[idx, :, :]


def _train_test_deacs(x, d_y, i):
    d_idx = np.where(d_y == i)[0]
    d = d_idx.size
    idx = np.random.choice(
        d_idx, d // 5, replace=False
    )

    deacs = x[d_idx, :, :]
    dX_train = np.delete(deacs, idx, axis=0)
    dX_test = x[idx, :, :]
    dy_train = np.full((dX_train.shape[0], 1), i)
    dy_test = np.full((dX_test.shape[0], 1), i)

    data = np.concatenate(
        [dy_test.reshape(-1, 1), idx.reshape(-1, 1)], axis=1
    )
    data = pd.DataFrame(data, columns=['pred', 'idx'])

    test_idx = pd.DataFrame(columns=['pred', 'idx'])
    data_tuple = (test_idx, data)
    test_idx = pd.concat(data_tuple, axis=0)

    return dX_train, dX_test, dy_train, dy_test, test_idx


def _train_test_non_deacs(X):
    nd = X.shape[0]
    idx = np.random.choice(
        nd, nd // 5, replace=False
    )
    ndX_train = np.delete(X, idx, axis=0)
    ndX_test = X[idx, :, :]
    dy_train = np.zeros(
        ndX_train.shape[0]
    ).reshape(-1, 1)
    dy_test = np.zeros(
        ndX_test.shape[0]
    ).reshape(-1, 1)
    return ndX_train, ndX_test, dy_train, dy_test


def _upsample(x, ndX, dX, i):
    nds, ds = ndX.shape[0], dX.shape[0]
    diff = nds // ds
    remainder = nds - diff * ds
    X = np.tile(dX, (diff, 1, 1))
    idx = np.random.choice(
        nds, remainder, replace=False
    )
    X = np.concatenate((X, x[idx, :, :]), axis=0)
    y = np.full((X.shape[0], 1), i)
    return X, y


def _train_test_split(x, d_y, catch, balance_test=True):
    nd = _split_labels(x, d_y, 0)
    ndX_train, ndX_test, ndy_train, ndy_test = \
        _train_test_non_deacs(nd)
    for i in range(1, catch + 1):
        dX_train, dX_test, dy_train, dy_test, test_idx = \
            _train_test_deacs(x, d_y, i)
        xtr, ytr = _upsample(x, ndX_train, dX_train, i)

        if 'X_train' in locals() and 'y_train' in locals():
            X_train = np.concatenate((X_train, xtr))
            y_train = np.concatenate((y_train, ytr))
        else:
            X_train = np.concatenate((ndX_train, xtr))
            y_train = np.concatenate((ndy_train, ytr))

        if balance_test:
            dX_test, dy_test = _upsample(x, ndX_test, dX_test, i)

        if 'X_test' in locals() and 'y_test' in locals():
            X_test = np.concatenate((X_test, dX_test))
            y_test = np.concatenate((y_test, dy_test))
        else:
            X_test = np.concatenate((ndX_test, dX_test))
            y_test = np.concatenate((ndy_test, dy_test))

    shuffled = pd.DataFrame(y_test)\
                 .sample(frac=1, replace=False)
    idx = shuffled.index.to_numpy()

    shuffled.reset_index(drop=False, inplace=True)
    condition = shuffled.iloc[:, 1] > 0
    shuffled_deacs = shuffled[condition].copy().dropna()
    shuffled_deacs.reset_index(drop=False, inplace=True)

    test_idx.reset_index(drop=True, inplace=True)
    zeros = y_test[~condition]
    test_idx['orig test indices'] = test_idx.index + len(zeros)

    q = pd.merge(
        left=shuffled_deacs,
        right=test_idx,
        left_on='index',
        right_on='orig test indices',
        how='left'
    )

    q.set_index('level_0', inplace=True, drop=True)
    test_idx =q[['pred', 'idx']]
    y_test = y_test[idx]
    X_test = X_test[idx, :, :]

    return X_train, y_train, X_test, y_test, q


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
    ).iloc[:, :-1].set_index('idx', drop=True)

    time_matrix = pd.merge(
        left=deac_times,
        right=q,
        left_index=True,
        right_index=True,
        how='left'
    )

    for i in range(1, catch + 1):
        condition = time_matrix['pred'] == i
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

    q.sort_index(inplace=True)
    return q.iloc[1:, :].to_numpy()


def pre_process(catch=2, rows=3, columns=6, balance_test=True):
    agg, deac_times = load_data()
    deac_times = shift_deac_dates(deac_times, catch)

    x = agg.to_numpy()
    x = _split_apply(x, 0, _add_distance_label, catch)
    d_y = x[:, -1].reshape(-1, 1)

    x = x[:, :-1]
    _scaler = MinMaxScaler()
    _scaler.fit(x)
    x = _scaler.transform(x)

    x = _split_apply(
        x, 0, _pad_dstack_sequences, rows, columns
    )
    X_train, y_train, X_test, y_test, test_idx = \
        _train_test_split(x, d_y, catch, balance_test=balance_test
    )

    y_train = pd.get_dummies(y_train.reshape(-1))
    y_test = pd.get_dummies(y_test.reshape(-1))

    meta = dict()
    meta['scaler'] = _scaler
    meta['deac_times'] = deac_times
    meta['test_idx'] = test_idx
    return X_train, y_train, X_test, y_test, meta


from machine_learning.MLModels import DolleLSTM

X_train, y_train, X_test, y_test, meta = pre_process()
deac_times = meta['deac_times']
test_idx = meta['test_idx']

model = DolleLSTM()
history = model.fit(
    X_train, y_train, X_test, y_test, epochs=50, class_weights={0: 1, 1: 0.2, 2: 0.2}
)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

y_pred = model.predict(X_test)

y_pred_1D, y_test_1D = flatten_idxmax_y(y_pred, y_test)
ConfusionMatrix = confusion_matrix(y_test_1D, y_pred_1D)
c = ConfusionMatrix.reshape(
    1, ConfusionMatrix.shape[0], ConfusionMatrix.shape[1]
)
times = calc_time_until_deac(2, test_idx, y_pred_1D, deac_times)

np.save(r'/home/james/Documents/DolleProject/Dolle/scripts', c)


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