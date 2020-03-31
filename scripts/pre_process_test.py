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


def _hstack(data, rows, columns):
    if 'JOBNUM' in data.columns:
        data.drop('JOBNUM', axis=1, inplace=True)
    a = np.concatenate(
        [data.to_numpy(), np.zeros([rows - 1, columns])]
    )
    eval_comp = [
        f'a[:-{rows - 1}]'
        if i == 0
        else f'a[{rows - 1}:]' if i == rows - 1
        else f'a[{i}:-{rows - i - 1}]'
        for i in range(rows)
    ]
    eval_str = f"np.hstack(({', '.join(eval_comp)}))" \
                 f".reshape(-1, {rows}, {columns})"
    t = eval(eval_str)
    return t


rows = 4
agg_2 = agg.groupby('JOBNUM')\
           .apply(add_distance_label, catch)\
           .reset_index(drop=True)
agg_2.drop('Label', axis=1, inplace=True)
agg_2 = agg_2.dropna().reset_index(drop=True)
agg_3 = agg_2.groupby('JOBNUM').apply(_hstack, rows, 6)
t = agg_3.values

a = np.array([-1, 4, 6])
for elem in t:
    a = np.concatenate((a, elem), axis=0)
