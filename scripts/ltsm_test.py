# import numpy as np
# import pandas as pd
#
# from MachineLearning import MLModels as m
# from MachineLearning.PreProcess.PreProcess import CatchMultiCategoryTTS, process_data, CatchSingleBlobNoTTS, NoCatchNoTTS
# from MachineLearning.STATS import ml_stats, confused
# from MachineLearning.PreProcess.Utilities import flatten
# from sklearn.metrics import confusion_matrix
# from sklearn.preprocessing import StandardScaler
#
# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Dropout
# from tensorflow.compat.v1.keras.utils import to_categorical
# import matplotlib.pyplot as plt
#
# from sklearn.metrics import confusion_matrix
#
# aggregate_path = r'/home/james/Documents/Development/Dolle/csvs/2018-02-16 -> ' \
#                  r'2018-12-19/MLAgg/agg_all_three.csv'
#
# agg_cols_to_use = [
#     'JOBNUM', 'Non Duplicate 0102', 'Sum 0102 Jam >= 20', '0103 Pace',
#     '0104 Alarm Time', '0105 Alarm Time', '0106 Alarm Time', 'Label'
# ]
#
# catch = 4
# skip = 1
# agg = pd.read_csv(aggregate_path, sep=';', usecols=agg_cols_to_use)
# agg = flatten(agg, catch + skip, 0, True)
# y = agg.loc[:, 'Label']
# X = agg.iloc[:, :(len(agg_cols_to_use) - 1) * (catch + 1)]
#
# scaler = StandardScaler()
# scaled = scaler.fit_transform(X)
# a = pd.DataFrame(scaled, index=agg.index, columns=agg.columns[:(len(agg_cols_to_use) - 1) * (catch + 1)])
#
# data = pd.concat([a, y], axis=1)
# data = data.sort_index()
# train = data.sample(frac=0.8, axis=0)
# test = data.loc[~data.index.isin(train.index), :]
# X_train, y_train = train.iloc[:, :-1].to_numpy(), train.iloc[:, -1].to_numpy()
# X_test, y_test = test.iloc[:, :-1].to_numpy(), test.iloc[:, -1].to_numpy()
# y_test, y_train = to_categorical(y_test), to_categorical(y_train)
#
# X_train = X_train.reshape(X_train.shape[0], 5, 7)
# X_test = X_test.reshape(X_test.shape[0], 5, 7)
#
# # design network
# model = Sequential()
# model.add(LSTM(35, input_shape=(X_train.shape[1], X_train.shape[2]), dropout=0.2, recurrent_dropout=0.2))
# # model.add(Dropout(0.2, input_shape=(18,)))
# # model.add(Dense(18, activation='relu'))
# # model.add(Dropout(0.2, input_shape=(18,)))
# # model.add(Dense(9, activation='relu'))
# # model.add(Dropout(0.2, input_shape=(18,)))
# model.add(Dense(4, activation='relu'))
# model.add(Dropout(0.2, input_shape=(18,)))
# model.add(Dense(2, activation='softmax'))
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# # fit network
# class_weight = {0: 1, 1: 15}
# history = model.fit(X_train, y_train, epochs=50, batch_size=50,
#                     validation_data=(X_test, y_test), verbose=2,
#                     shuffle=False, class_weight=class_weight)
# # plot history
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# plt.show()
#
# y_pred = model.predict(X_test)
# y_pred_1D = pd.DataFrame(y_pred).idxmax(axis=1)
# y_test_1D = pd.DataFrame(y_test).idxmax(axis=1)
#
# conf = confusion_matrix(y_test_1D, y_pred_1D)
#
#

import pandas as pd

from MachineLearning import MLModels as m
from MachineLearning.PreProcess.PreProcess import CatchMultiCategoryTTS, process_data, CatchSingleBlobNoTTS, NoCatchNoTTS
from MachineLearning.STATS import ml_stats, confused
from MachineLearning.PreProcess.Utilities import flatten
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from tensorflow.compat.v1.keras.utils import to_categorical
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

aggregate_path = r'/home/james/Documents/Development/Dolle/csvs/2018-02-16 -> ' \
                 r'2018-12-19/MLAgg/agg_all_three.csv'

agg_cols_to_use = [
    'JOBNUM', 'Non Duplicate 0102', 'Sum 0102 Jam >= 20', '0103 Pace',
    '0104 Alarm Time', '0105 Alarm Time', '0106 Alarm Time', 'Label'
]


def slice_data(data, catch, num_columns):
    data = data.drop('JOBNUM', axis=1)
    X = data.iloc[::catch, :num_columns]
    y = data.iloc[::catch, num_columns:]
    condition = y.columns.str.contains('Label')
    y = y.loc[:, condition].max(axis=1)
    return pd.concat([X, y], axis=1)


def slice_data_overlapping(data, num_columns):
    data = data.drop('JOBNUM', axis=1)
    X = data.iloc[:, :num_columns]
    y = data.iloc[:, num_columns:]
    condition = y.columns.str.contains('Label')
    y = y.loc[:, condition].max(axis=1)
    return pd.concat([X, y], axis=1)

# import numpy as np
# import pandas as pd
#
# from MachineLearning import MLModels as m
# from MachineLearning.PreProcess.PreProcess import CatchMultiCategoryTTS, process_data, CatchSingleBlobNoTTS, NoCatchNoTTS
# from MachineLearning.STATS import ml_stats, confused
# from MachineLearning.PreProcess.Utilities import flatten
# from sklearn.metrics import confusion_matrix
# from sklearn.preprocessing import StandardScaler
#
# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Dropout
# from tensorflow.compat.v1.keras.utils import to_categorical
# import matplotlib.pyplot as plt
#
# from sklearn.metrics import confusion_matrix
#
# aggregate_path = r'/home/james/Documents/Development/Dolle/csvs/2018-02-16 -> ' \
#                  r'2018-12-19/MLAgg/agg_all_three.csv'
#
# agg_cols_to_use = [
#     'JOBNUM', 'Non Duplicate 0102', 'Sum 0102 Jam >= 20', '0103 Pace',
#     '0104 Alarm Time', '0105 Alarm Time', '0106 Alarm Time', 'Label'
# ]
#
# catch = 4
# skip = 1
# agg = pd.read_csv(aggregate_path, sep=';', usecols=agg_cols_to_use)
# agg = flatten(agg, catch + skip, 0, True)
# y = agg.loc[:, 'Label']
# X = agg.iloc[:, :(len(agg_cols_to_use) - 1) * (catch + 1)]
#
# scaler = StandardScaler()
# scaled = scaler.fit_transform(X)
# a = pd.DataFrame(scaled, index=agg.index, columns=agg.columns[:(len(agg_cols_to_use) - 1) * (catch + 1)])
#
# data = pd.concat([a, y], axis=1)
# data = data.sort_index()
# train = data.sample(frac=0.8, axis=0)
# test = data.loc[~data.index.isin(train.index), :]
# X_train, y_train = train.iloc[:, :-1].to_numpy(), train.iloc[:, -1].to_numpy()
# X_test, y_test = test.iloc[:, :-1].to_numpy(), test.iloc[:, -1].to_numpy()
# y_test, y_train = to_categorical(y_test), to_categorical(y_train)
#
# X_train = X_train.reshape(X_train.shape[0], 5, 7)
# X_test = X_test.reshape(X_test.shape[0], 5, 7)
#
# # design network
# model = Sequential()
# model.add(LSTM(35, input_shape=(X_train.shape[1], X_train.shape[2]), dropout=0.2, recurrent_dropout=0.2))
# # model.add(Dropout(0.2, input_shape=(18,)))
# # model.add(Dense(18, activation='relu'))
# # model.add(Dropout(0.2, input_shape=(18,)))
# # model.add(Dense(9, activation='relu'))
# # model.add(Dropout(0.2, input_shape=(18,)))
# model.add(Dense(4, activation='relu'))
# model.add(Dropout(0.2, input_shape=(18,)))
# model.add(Dense(2, activation='softmax'))
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# # fit network
# class_weight = {0: 1, 1: 15}
# history = model.fit(X_train, y_train, epochs=50, batch_size=50,
#                     validation_data=(X_test, y_test), verbose=2,
#                     shuffle=False, class_weight=class_weight)
# # plot history
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# plt.show()
#
# y_pred = model.predict(X_test)
# y_pred_1D = pd.DataFrame(y_pred).idxmax(axis=1)
# y_test_1D = pd.DataFrame(y_test).idxmax(axis=1)
#
# conf = confusion_matrix(y_test_1D, y_pred_1D)
#
#

import pandas as pd

from MachineLearning import MLModels as m
from MachineLearning.PreProcess.PreProcess import CatchMultiCategoryTTS, process_data, CatchSingleBlobNoTTS, NoCatchNoTTS
from MachineLearning.STATS import ml_stats, confused
from MachineLearning.PreProcess.Utilities import flatten
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from tensorflow.compat.v1.keras.utils import to_categorical
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

aggregate_path = r'/home/james/Documents/Development/Dolle/csvs/2018-02-16 -> ' \
                 r'2018-12-19/MLAgg/agg_all_three.csv'

agg_cols_to_use = [
    'JOBNUM', 'Non Duplicate 0102', 'Sum 0102 Jam >= 20', '0103 Pace',
    '0104 Alarm Time', '0105 Alarm Time', '0106 Alarm Time', 'Label'
]


def slice_data(data, catch, num_columns):
    data = data.drop('JOBNUM', axis=1)
    X = data.iloc[::catch, :num_columns]
    y = data.iloc[::catch, num_columns:]
    condition = y.columns.str.contains('Label')
    y = y.loc[:, condition].max(axis=1)
    return pd.concat([X, y], axis=1)


def slice_data_overlapping(data, num_columns):
    data = data.drop('JOBNUM', axis=1)
    X = data.iloc[:, :num_columns]
    y = data.iloc[:, num_columns:]
    condition = y.columns.str.contains('Label')
    y = y.loc[:, condition].max(axis=1)
    return pd.concat([X, y], axis=1)


catch = 4
skip = 1
agg = pd.read_csv(aggregate_path, sep=';', usecols=agg_cols_to_use)
flattened = flatten(agg, catch + skip, 0, True)
flattened.loc[:, 'JOBNUM'] = agg.loc[:, 'JOBNUM']
groupby = flattened.groupby('JOBNUM')
flattened = groupby.apply(slice_data_overlapping, 35)

y = flattened.iloc[:, -1]
X = flattened.iloc[:, :-1]

scaler = MinMaxScaler()
scaled = scaler.fit_transform(X)
a = pd.DataFrame(scaled, index=X.index, columns=X.columns)

data = pd.concat([a, y], axis=1)
data = data.sort_index()

train = data.sample(frac=0.8, axis=0)
test = data.loc[~data.index.isin(train.index), :]
X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1].to_numpy()
X_train = X_train.loc[:, X_train.columns[::-1]].to_numpy()
X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1].to_numpy()
X_test = X_test.loc[:, X_test.columns[::-1]].to_numpy()

y_test, y_train = to_categorical(y_test), to_categorical(y_train)

X_train = X_train.reshape(X_train.shape[0], 5, 7)
X_test = X_test.reshape(X_test.shape[0], 5, 7)

# design network
model = Sequential()
model.add(LSTM(7, input_shape=(
    X_train.shape[1], X_train.shape[2]), dropout=0.2, recurrent_dropout=0.2)
)
model.add(Dropout(0.2, input_shape=(18,)))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.2, input_shape=(2,)))
model.add(Dense(2, activation='softmax'))

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min'
)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# fit network
class_weight = {0: 1, 1: 5}
history = model.fit(
    X_train, y_train, epochs=25, batch_size=5,
    validation_data=(X_test, y_test), verbose=2,
    shuffle=False, callbacks=[
        earlyStopping, mcp_save, reduce_lr_loss
    ],
    class_weight=class_weight
)

# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

y_pred = model.predict(X_test)
y_pred_1D = pd.DataFrame(y_pred).idxmax(axis=1)
y_test_1D = pd.DataFrame(y_test).idxmax(axis=1)

conf = confusion_matrix(y_test_1D, y_pred_1D)

model.load_weights(filepath=r'/home/james/Documents/Development/Dolle/.mdl_wts.hdf5')

y_pred = model.predict(X_test)
y_pred_1D = pd.DataFrame(y_pred).idxmax(axis=1)
y_test_1D = pd.DataFrame(y_test).idxmax(axis=1)

conf_2 = confusion_matrix(y_test_1D, y_pred_1D)
#
#
# catch = 6
# skip = 2
# agg = pd.read_csv(aggregate_path, sep=';', usecols=agg_cols_to_use)
# flattened = flatten(agg, catch + skip, 0, True)
# flattened.loc[:, 'JOBNUM'] = agg.loc[:, 'JOBNUM']
# groupby = flattened.groupby('JOBNUM')
# flattened = groupby.apply(slice_data_overlapping, 35)
#
# y = flattened.iloc[:, -1]
# X = flattened.iloc[:, :-1]
#
# scaler = MinMaxScaler()
# scaled = scaler.fit_transform(X)
# a = pd.DataFrame(scaled, index=X.index, columns=X.columns)
#
# data = pd.concat([a, y], axis=1)
# data = data.sort_index()
# train = data.sample(frac=0.8, axis=0)
# test = data.loc[~data.index.isin(train.index), :]
# X_train, y_train = train.iloc[:, :-1].to_numpy(), train.iloc[:, -1].to_numpy()
# X_test, y_test = test.iloc[:, :-1].to_numpy(), test.iloc[:, -1].to_numpy()
# y_test, y_train = to_categorical(y_test), to_categorical(y_train)
#
# X_train = X_train.reshape(X_train.shape[0], 5, 7)
# X_test = X_test.reshape(X_test.shape[0], 5, 7)
#
# # design network
# model = Sequential()
# model.add(LSTM(7, input_shape=(
#     X_train.shape[1], X_train.shape[2]), dropout=0.2, recurrent_dropout=0.2)
# )
# model.add(Dropout(0.2, input_shape=(18,)))
# model.add(Dense(4, activation='relu'))
# model.add(Dropout(0.2, input_shape=(2,)))
# model.add(Dense(2, activation='softmax'))
#
# earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
# mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
# reduce_lr_loss = ReduceLROnPlateau(
#     monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min'
# )
#
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# # fit network
# class_weight = {0: 1, 1: 5}
# history = model.fit(
#     X_train, y_train, epochs=25, batch_size=5,
#     validation_data=(X_test, y_test), verbose=2,
#     shuffle=False, callbacks=[
#         earlyStopping, mcp_save, reduce_lr_loss
#     ],
#     class_weight=class_weight
# )
#
# # plot history
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# plt.show()
#
# y_pred = model.predict(X_test)
# y_pred_1D = pd.DataFrame(y_pred).idxmax(axis=1)
# y_test_1D = pd.DataFrame(y_test).idxmax(axis=1)
#
# conf = confusion_matrix(y_test_1D, y_pred_1D)
#
# model.load_weights(filepath=r'/home/james/Documents/Development/Dolle/.mdl_wts.hdf5')
#
# y_pred = model.predict(X_test)
# y_pred_1D = pd.DataFrame(y_pred).idxmax(axis=1)
# y_test_1D = pd.DataFrame(y_test).idxmax(axis=1)
#
# conf_2 = confusion_matrix(y_test_1D, y_pred_1D)
