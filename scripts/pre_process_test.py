import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

import machine_learning.PreProcess.Utilities as ml_util

from machine_learning import MLModels as m


aggregate_path = r'/home/james//Documents/DolleProject/dolle_csvs/28-02-16 to 2018-12-19' \
                 r'/MLAgg0103 1405: 1 SW, 3 CF, no overlaps/SW-3D-3F-3B-12T.csv'

# agg_cols_to_use = [
#     'JOBNUM', 'Non Duplicate 0102', '0103 Pace',
#     '0104 Alarm Time', '0105 Alarm Time', '0106 Alarm Time', 'Label'
# ]

agg_cols_to_use = [
    'JOBNUM', 'Non Duplicate 0102', '0103 Pace',
    '0104 Alarm Time', '0105 Alarm Time', '0106 Alarm Time', 'Label'
]

agg = pd.read_csv(aggregate_path, sep=',', usecols=agg_cols_to_use)
catch = 3


def add_distance_label(data, catch):
    data['True Label'] = 0
    condition = data['Label'] == 1
    deacs = data[condition].copy()
    for index, _ in deacs.iterrows():
        try:
            count = 1
            for i in range(index - 1, index - catch - 2, -1):
                data.loc[i, 'True Label'] = count
                count += 1
        except:
            pass
    return data


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


def _pad_dstack_sequences(x, rows):
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


def _split_apply(x, col, func, *args):
    for group in np.unique(x[:, col]):
        j = x[x[:, col] == group]
        j = func(j, *args)
        o = np.concatenate([o, j]) \
            if 'o' in locals() else j
    return o


rows = 4
columns = 6

x = agg.to_numpy()
a = _split_apply(x, 0, _add_distance_label, catch)
d_y = x[:, -1]
y = x[:, -2]

x = _split_apply(
    a[:, :-1], 0, _pad_dstack_sequences, rows
)

idx = np.where(d_y == 0)
non_deacs = x[idx, :, :]


agg_2 = agg.groupby('JOBNUM') \
    .apply(add_distance_label, catch) \
    .reset_index(drop=True)

agg_2 = agg_2.dropna().reset_index(drop=True)
o_label = agg_2['Label']
d_label = agg_2['True Label']
agg_2.drop('Label', axis=1, inplace=True)

    # if count == 0:
    #     o = j
    #     count += 1
    # else:
    #     o = np.concatenate([o, j])
    #
    #
    # o = j if o not in locals() else j
    # # output = np.concatenate([o, d]) if o in locals() else d



# agg_4 = agg_2.groupby('JOBNUM').apply(_hstack, rows, columns)
# t = agg_3.values
#
# a = np.array([-1, 4, 6])
# for elem in t:
#     a = np.concatenate((a, elem), axis=0)
