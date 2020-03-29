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
    'Non Duplicate 0102', '0103 Pace',
    '0104 Alarm Time', '0105 Alarm Time', '0106 Alarm Time', 'Label'
]


agg = pd.read_csv(aggregate_path, sep=',', usecols=agg_cols_to_use)

num_rows = 6
skip = 1
catch = 3

first_row, skip = ml_util.adjust_inputs_catch(num_rows, skip, catch)
agg = ml_util.flatten(agg, first_row, skip, True)
agg, _ = ml_util.scale(agg)
agg = ml_util.add_labels(skip, catch, agg)

labels = agg['True Label'].value_counts()
t = agg.loc[agg['True Label'] == 0, :].copy()
for i in range(1, labels.size):
    condition = agg['True Label'] == i
    a = agg.loc[condition, :].copy()
    scale = labels[0] // labels[i]
    diff = labels[0] - scale * len(a)
    b = pd.concat([a] * scale)
    b = pd.concat([b, a.sample(diff)])
    t = pd.concat([t, b])

t.reset_index(drop=True, inplace=True)
train = t.groupby('True Label')\
         .apply(lambda x: x.sample(frac=0.8, replace=False))

train.index = train.index.get_level_values(1)
train.sort_index(inplace=True)

condition = t.index.isin(train.index)
test = t.loc[~condition, :].copy()

X_train, y_train = train.iloc[:, :-2], pd.get_dummies(train.iloc[:, -1])
X_train = X_train.iloc[:, :6*catch]

X_test, y_test = test.iloc[:, :-2], pd.get_dummies(test.iloc[:, -1])
X_test = X_test.iloc[:, :6*catch]

model = m.DolleNeural1D()

class_weights = {0: 1, 1: 0.7, 2: 0.7, 3: 0.7}
# class_weights = {i: 1 for i in range(catch + 1)}
history = model.fit(X_train, y_train, X_test=X_test, y_test=y_test, class_weights=class_weights)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

y_pred = model.predict(X_test)

# model.save(r'/home/james/Documents/model.hdf5')
y_pred_pd = pd.DataFrame(y_pred, index=y_test.index)

y_pred_1D = y_pred_pd.idxmax(axis=1)
y_test_1D = pd.DataFrame(y_test).idxmax(axis=1)

ConfusionMatrix = confusion_matrix(y_test_1D, y_pred_1D)


import numpy as np

a = [[[0, 1]],
     [[2, 3]],
     [[4, 5]],
     [[6, 7]],
     [[8, 9]],
     [[10, 11]]
     ]

b = np.hstack((a[:-2], a[1:-1], a[2:]))

c = agg.to_numpy()
z = np.hstack((c[2:], c[1:-1], c[:-2]))\
      .reshape(-1, 3, 6)

