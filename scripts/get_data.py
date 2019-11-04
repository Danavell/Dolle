import numpy as np
import pandas as pd

from machine_learning import MLModels as m
from machine_learning.PreProcess.PreProcess import process_data
from machine_learning.STATS import ml_stats, confused

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

aggregate_path = r'/home/james/Documents/Development/dolle_csvs/01-01-18 to 01-01-19/' \
                 r'MLAgg0103 1405: 1 SW, 3 CF, no overlaps/SW-3D-3F-3B-12T.csv'

agg_cols_to_use = [
    'JOBNUM', 'Non Duplicate 0102', 'Sum 0102 Jam >= 20', '0103 Pace',
    '0104 Alarm Time', '0105 Alarm Time', '0106 Alarm Time', 'Label'
]

sensor_path = r'/home/james/Documents/Development/dolle_csvs/01-01-18 to 01-01-19/' \
              r'MLAgg0103 1405: 1 SW, 3 CF, no overlaps/sensor_data.csv'

agg = pd.read_csv(aggregate_path, sep=';', usecols=agg_cols_to_use)
sensor_data = pd.read_csv(sensor_path, sep=';', parse_dates=['Date'], infer_datetime_format=True)

num_rows = 6
skip = 1
catch = 3
method = 'multi'

X_train, y_train, X_test, y_test, meta = process_data(
    agg, num_rows=num_rows, skip=skip, catch=catch, split=0.8, method=method
)

model = m.DolleNeural1D()
# , 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1
class_weights = {0: 1, 1: 0.4, 2: 0.1, 3: 0.1, }
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

ConfusionMatrix = confused(y_test, y_pred, meta)
ConfusionMatrix_2 = confusion_matrix(y_test_1D, y_pred_1D)
total_num_deactivations, non_unique_total_seen, total_seen, time_deltas, data, dr = ml_stats(
    y_test, y_pred, sensor_data, meta
)

percentile_ticks = [
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95
]

percentiles = np.percentile(
    time_deltas, percentile_ticks
)

indices = [
    '5%', '10%', '15%', '20%', '25%',
    '30%', '35%', '40%', '45%', '50%',
    '55%', '60%', '65%', '70%', '75%',
    '80%', '85%', '90%', '95%',
]
percentiles = pd.DataFrame(percentiles.T, index=indices, columns=['Time (s)'])


output_dict = dict()
for i in range(1, catch + 1):
    condition = dr.loc[:, 'label'] == i
    output_dict[i] = len(dr.loc[condition, :].index)


y_pred = model.predict_best_weights(X_test, r'/home/james/Documents/Development/Dolle/machine_learning/mdl_wts.hdf5')

# model.save(r'/home/james/Documents/model.hdf5')
y_pred_pd = pd.DataFrame(y_pred, index=y_test.index)

y_pred_1D = y_pred_pd.idxmax(axis=1)
y_test_1D = pd.DataFrame(y_test).idxmax(axis=1)


ConfusionMatrix_3 = confused(y_test, y_pred, meta)
ConfusionMatrix_4 = confusion_matrix(y_test_1D, y_pred_1D)
total_num_deactivations_2, non_unique_total_seen_2, total_seen_2, time_deltas_2, data_2, dr_2 = ml_stats(
    y_test, y_pred, sensor_data, meta
)

percentiles_2 = np.percentile(
    time_deltas, percentile_ticks
)

percentiles_2 = pd.DataFrame(percentiles_2.T, index=indices, columns=['Time (s)'])

output_dict_2 = dict()
for i in range(1, catch + 1):
    condition = dr_2.loc[:, 'label'] == i
    output_dict_2[i] = len(dr_2.loc[condition, :].index)


# import numpy as np
# import pandas as pd
#
# from machine_learning import MLModels as m
# from machine_learning.PreProcess.PreProcess import process_data
# from machine_learning.STATS import ml_stats, confused
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
# sensor_path = r'/home/james/Documents/Development/Dolle/csvs/2018-02-16 -> ' \
#                  r'2018-12-19/MLAgg/sensor_data.csv'
#
# agg = pd.read_csv(aggregate_path, sep=';', usecols=agg_cols_to_use)
# sensor_data = pd.read_csv(sensor_path, sep=';', parse_dates=['Date'], infer_datetime_format=True)
#
# num_rows = 6
# skip = 1
# catch = 1
# method = 'multi'
#
# X_train, y_train, X_test, y_test, meta = process_data(
#     agg, num_rows=num_rows, skip=skip, catch=catch, split=0.8, method=method, start_deac=100
# )
#
# class_weights = {0: 1, 1: 0.7}
# model, y_pred = m.create_model_fit_predict(X_train, y_train, X_test, y_test, class_weights)
# y_pred_single_pd = pd.DataFrame(y_pred, index=y_test.index)
# y_pred_single_1D = y_pred_single_pd.idxmax(axis=1)
# y_test_single_1D = pd.DataFrame(y_test).idxmax(axis=1)
#
# catch = 6
# class_weights = {0: 1, 1: 0.6, 2: 0.3, 3: 0.3, 4: 0.3, 5: 0.3, 6: 0.3}
# X_train_multi, y_train_multi, X_test_multi, y_test_multi, meta_multi = process_data(
#     agg, num_rows=num_rows, skip=skip, catch=catch, split=0.8, method=method, start_deac=92
# )
# model_multi, y_pred_multi = m.create_model_fit_predict(
#     X_train_multi, y_train_multi, X_test_multi, y_test_multi, class_weights
# )
# y_pred_multi_pd = pd.DataFrame(y_pred_multi, index=y_test_multi.index)
# y_pred_multi_1D = y_pred_multi_pd.idxmax(axis=1)
# y_test_multi_1D = pd.DataFrame(y_test_multi).idxmax(axis=1)
#
# condition = y_pred_multi_1D.index.isin(y_pred_single_1D.index)
# y_pred_multi_1D_resized = y_pred_multi_1D[condition]
#
# condition = y_pred_single_1D.index.isin(y_pred_multi_1D.index)
# y_pred_single_1D_resized = y_pred_single_1D[condition]
#
# condition = y_test_single_1D.index.isin(y_pred_single_1D_resized.index)
# y_test_single_1D_resized = y_test_single_1D[condition]
#
# merged = pd.concat([y_pred_single_1D_resized, y_pred_multi_1D_resized], axis=1)
# condition = (merged.iloc[:, 0] == 1) & (merged.iloc[:, 1] == 1)
# indices = merged.loc[condition].index
#
# output = pd.Series(index=merged.index)
# output.loc[indices] = 1
# output = output.fillna(0)
#
# ConfusionMatrixSingle = confusion_matrix(y_test_single_1D, y_pred_single_1D)
# ConfusionMatrix = confusion_matrix(output, y_pred_single_1D_resized)
#
# total_num_deactivations, non_unique_total_seen, total_seen, time_deltas, data, dr = ml_stats(
#     y_test, y_pred, sensor_data, meta
# )
#
# percentile_ticks = [
#         5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95
#     ]
#
# percentiles = np.percentile(
#     time_deltas, percentile_ticks
# )
#
# indices = [
#     '5%', '10%', '15%', '20%', '25%',
#     '30%', '35%', '40%', '45%', '50%',
#     '55%', '60%', '65%', '70%', '75%',
#     '80%', '85%', '90%', '95%',
# ]
# percentiles = pd.DataFrame(percentiles.T, index=indices, columns=['Time (s)'])
#
#
# output_dict = dict()
# for i in range(1, catch + 1):
#     condition = dr.loc[:, 'label'] == i
#     output_dict[i] = len(dr.loc[condition, :].index)
#
#
#
#
# # y_pred = model.predict_best_weights(X_test, r'/home/james/Documents/Development/Dolle/machine_learning/mdl_wts.hdf5')
#
# # model.save(r'/home/james/Documents/model.hdf5')
# y_pred_pd = pd.DataFrame(y_pred, index=y_test.index)
#
# y_pred_1D = y_pred_pd.idxmax(axis=1)
# y_test_1D = pd.DataFrame(y_test).idxmax(axis=1)
#
#
# ConfusionMatrix_3 = confused(y_test, y_pred, meta)
# ConfusionMatrix_4 = confusion_matrix(y_test_1D, y_pred_1D)
# total_num_deactivations_2, non_unique_total_seen_2, total_seen_2, time_deltas_2, data_2, dr_2 = ml_stats(
#     y_test, y_pred, sensor_data, meta
# )
#
# percentiles_2 = np.percentile(
#     time_deltas, percentile_ticks
# )
#
# percentiles_2 = pd.DataFrame(percentiles_2.T, index=indices, columns=['Time (s)'])
#
# output_dict_2 = dict()
# for i in range(1, catch + 1):
#     condition = dr_2.loc[:, 'label'] == i
#     output_dict_2[i] = len(dr_2.loc[condition, :].index)
