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


agg_2 = agg.groupby('JOBNUM').apply(add_distance_label, catch)