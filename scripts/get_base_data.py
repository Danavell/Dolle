import os

import numpy as np
import pandas as pd

import pre_processing.STATS_aggregate_0102.aggregates as ast_0102
import pre_processing.STATS_aggregate_0102.business_intelligence as BI

from pre_processing.utils import BaseDataFactory
from pre_processing.aggregate_0102 import aggregates as ag

BaseDataFactory.get_ladder_codes()
# old_folder = '2020-01-06 04:44:58 - 2020-01-21 18:33:13'

# old_data = BaseDataFactory.factory(
#     1, old_folder, fix_duplicates=False, save=True
# )

new_folder = '28-02-16 to 2018-12-19'
new_data = BaseDataFactory.factory(
    8, new_folder, fix_duplicates=True, save=True
)

# a = data['sensor_data']
#
# condition = (a['0101 Duration'] == 0) & (a['Non Duplicate 0101'] == 1)
# b = a[condition]
#
# path = r'/home/james/Development/DolleProject/dolle_csvs/' \
#        r'28-02-16 to 2018-12-19/Stats 1405: 0102, 1 SW, 2 CF, ' \
#        r'no overlaps/'
#
# p_sensor_data = pd.read_csv(
#     os.path.join(path, r'sensor_data.csv'), sep=','
# )
# agg = pd.read_csv(
#     os.path.join(path, r'all 3 products/all 3 products.csv'), sep=','
# )
#
# func = ag.sum_non_deac_pace_ins_larger_than_n
# agg = ag.deacs_roll(agg, func, 25, n_rows_back=30)
#
# agg, \
# deacs_count, \
# non_deacs_count, \
# total_spikes, \
# num_spikes_deacs, \
# total_num_spikes_deacs, \
# percentage_spikes_before_deacs = BI.pie_chart_num_spikes_count(agg)
#
# deacs, p_sensor_data = ast_0102.returns_deacs_strings_before_or_after_ladder_out(
#     agg, p_sensor_data
# )
#
# condition = (deacs['strings until next ladder'] == -1) & \
#             (deacs['0103 non_unique ID'] > 0) & \
#             (deacs['0102 ID'] > 0)
#
# a = deacs[condition]
#
# zero_strings_until_next = deacs[deacs['strings until next ladder'] == 0]
#
#
# condition = deacs['0102 ID'] > 0
# no_zero_deacs = deacs[condition]
#
# # def roll_forward_return_unique_series(data, func, n, n_rows_forward=30, n_sum=0):
# #     """
# #     Groups on JOBNUM and for each deactivation looks back
# #     a maximum of n_rows_back and sums the number of pace-ins
# #     longer than n
# #     """
# #     groupby = data.groupby('JOBNUM')
# #     data = groupby.apply(func, n, n_rows_forward, n_sum)\
# #                   .reset_index(drop=True)
# #     return data.unique()
# #
# #
# # def return_all_unique_deac_ids_in_n_future_rows(data, n, n_rows_forward, n_sum):
# #     """
# #     Iterates through each pace >= n ID in each JOBNUM and calculates how many
# #     pace >= n occured n_rows_back
# #     """
# #
# #     n_rows_forward += 1
# #     condition = (data[f'0102 Pace >= {n} ID'] >= 1) & \
# #                 (data[f'0102 Sum Pace >= {n}'] >= n_sum)
# #     ids = data.loc[condition, :]
# #
# #     for index, row in ids.iterrows():
# #         """
# #         check whether there are less than n_rows_back before the
# #         0102 pace >= n ID
# #         """
# #         last_index = data.index[-1]
# #         if index + n_rows_forward <= last_index:
# #             sliced = data.loc[index:index + n_rows_forward, '0101 ID']
# #         else:
# #             sliced = data.loc[index:last_index + 1, '0101 ID']
# #
# #         if 'unique_ids' in locals():
# #             unique_ids = pd.concat([unique_ids, sliced], axis=0, sort=False)
# #         else:
# #             unique_ids = sliced.copy()
# #     try:
# #         return unique_ids
# #     except:
# #         pass
# #
# #
# # def return_all_unique_non_deac_ids_in_n_future_rows(data, n, n_rows_forward, n_sum):
# #     """
# #     Iterates through each pace >= n ID in each JOBNUM and calculates how many
# #     pace >= n occured n_rows_back
# #     """
# #
# #     n_rows_forward += 1
# #     condition = (data[f'0102 Pace >= {n} ID'] >= 1) & \
# #                 (data['Label'] == 0) & \
# #                 (data[f'0102 Sum Pace >= {n}'] >= n_sum)
# #     ids = data.loc[condition, :]
# #
# #     for index, row in ids.iterrows():
# #         """
# #         check whether there are less than n_rows_back before the
# #         0102 pace >= n ID
# #         """
# #         last_index = data.index[-1]
# #         if index + n_rows_forward <= last_index:
# #             sliced = data.loc[index:index + n_rows_forward, :]
# #         else:
# #             sliced = data.loc[index:last_index + 1, :]
# #
# #         sliced = sliced.loc[condition, '0102 Pace >= 25 ID']
# #
# #         if 'unique_ids' in locals():
# #             unique_ids = pd.concat([unique_ids, sliced], axis=0, sort=False)
# #         else:
# #             unique_ids = sliced.copy()
# #     try:
# #         return unique_ids
# #     except:
# #         pass
#
#
# # deacs = agg[agg['Label'] == 1].copy()
# # deacs['0101 ID'] = np.arange(len(deacs.index))
# # agg['0101 ID'] = deacs['0101 ID']
# # agg['0101 ID'].fillna(0, inplace=True)
#
# # n_sum = 0
# # deac_ids = roll_forward_return_unique_series(
# #     data=agg,
# #     func=return_all_unique_deac_ids_in_n_future_rows,
# #     n=25,
# #     n_rows_forward=30,
# #     n_sum=n_sum
# # )
# # deacs = len(deac_ids)
# #
# # non_deac_ids = roll_forward_return_unique_series(
# #     data=agg,
# #     func=return_all_unique_non_deac_ids_in_n_future_rows,
# #     n=25,
# #     n_rows_forward=30,
# #     n_sum=n_sum
# # )
# # non_deacs = len(non_deac_ids)
#
# condition = (agg['0102 ID'] > 0) & \
#             (agg['Label'] == 1)
#
# mid_deacs = agg[condition]
#
#
# data = deacs_roll(
#     data=agg,
#     func=return_all_n_rows_before_every_deactivation,
#     n=25,
#     n_rows_back=30)
#
# data.dropna(inplace=True)
#
# percs = [
#     np.percentile(data, 25),
#     np.percentile(data, 50),
#     np.percentile(data, 75),
# ]
#
# deacs = agg[(agg['Label'] == 1) & (agg['0102 Pace >= 25 Count'] > 0)]
# example = agg.loc[46427:46457, '0102 Pace']
# example.reset_index(drop=True, inplace=True)
# example.index = example.index + 1
#
# import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')
#
# plt.plot(example)
# plt.xlabel('Strings')
# plt.ylabel('String In Pace')
# plt.title('String In Pace for 30 strings before a deactivation')
# plt.show()
# a = deacs.copy()
#
# a['strings plus/minus'] = a['strings since last ladder'].copy()
# condition = a['strings until next ladder'] <= a['strings since last ladder']
# indices = a[condition].index
# a.loc[indices, 'strings plus/mins'] = a.loc[indices, 'strings until next ladder']
#
# condition = (a['Non Duplicate 0101'] == 1) & \
#             (a['0102 ID'] > 0) & \
#             (a['0103 ID'] > 0) & \
#             (a['0103 non_unique ID'] > 1)
# a = a.loc[condition, :]
#
# c = a.groupby('strings plus/minus').count().iloc[:, 1]
# c = c.reset_index()
# c.columns = ['Proximity', 'Number Of Deactivations']
# k = c.corr()
# c.iloc[7, 1] = np.sum(c.iloc[7:, 1])
# c = c.iloc[:8, :]
# c.iloc[7, 0] = 'Greater than 6'
#
# c.iloc[0, 0] = 'After ladder exits but before next string enters'
# c.iloc[1, 0] = 'After last string enters before next ladder exits'
# for i in range(2, 7):
#     c.iloc[i, 0] = f'+/- {i-1} string'
#
#
# #
# # b = p_sensor_data.copy()
# # condition = (b['0102 ID'] < 0)
# #
# # b = b.loc[condition, 'Non Duplicate 0101'].sum()
# #
# # condition = deacs['strings since last ladder'] == -1
# # t = deacs[condition]
# # p_sensor_data['0103 Cumcount'] = ut.make_column_cumcount_ne_zero(
# #        p_sensor_data, 'Non Duplicate 0103', 'JOBNUM', 'JOBNUM', n=2, fill='ffill'
# # )
# # p_sensor_data['0103 Cumcount'] = p_sensor_data['0103 Cumcount'].fillna(1)
# #
# # condition = (p_sensor_data['strings until next ladder'] == 1) & (p_sensor_data['0103 non_unique ID'] > 0)
# # data = p_sensor_data[condition]
# # np.sum(deacs['Non Duplicate 0101'])
# #
# # condition = p_sensor_data['0103 Cumcount'] == 1
# # columns = ['Date', 'JOBNUM'] + [f'Non Duplicate 010{X}' for X in range(1, 7)] + ['0103 Cumcount']
# # data = p_sensor_data.loc[condition, columns]
# #
# # np.sum(data['Non Duplicate 0101'])
# #
# # c = deacs.groupby('strings until next ladder').count().iloc[:, 1]
# # c = c.reset_index()
# # c.columns = ['strings until next ladder', 'Number Of Deactivations']
# # k = c.corr()
# # c.iloc[7, 1] = np.sum(c.iloc[7:, 1])
# # c = c.iloc[:8, :]
# # c.iloc[7, 0] = 'Greater than 7'
# #
# # a = sensor_data[['Date', 'Non Duplicate 0101', 'Non Duplicate 0102', 'Non Duplicate 0103']]
# # a.columns = ['Timestamp', 'Deactivation', 'String In', 'Ladder Out']
# # b = pd.DataFrame(
# #     np.percentile(deacs['strings until next ladder'].dropna(), [i for i in range(0, 110, 10)]),
# #     [f'{i}%' for i in range(0, 110, 10)]
# # )
#
#
#
# #
# #
# # import matplotlib.pyplot as plt
# #
# # b = pd.DataFrame(
# #     np.percentile(agg['Time Delta'].dropna(), [i for i in range(0, 110, 10)]),
# #     [f'{i}%' for i in range(0, 110, 10)]
# # )
# #
# # b.iloc[0, 0] = 1
# # b.iloc[1, 0] = 1
# #
# # plt.plot(np.log(b), label='Time delta between last string and deactivation')
# # plt.axhline(
# #     y=np.log10(8), color='r', linestyle='-', label='Median String In Pace'
# # )
# # plt.ylabel('log 10 time difference')
# # plt.xlabel('Percentage of Deactivations')
# # plt.legend(loc='upper left')
# # plt.show()
# #
# # deacs, sensor_data = ast_0102.returns_deacs_strings_before_or_after_ladder_out(
# #     agg, p_sensor_data
# # )
# #
# # a = sensor_data[['Date', 'Non Duplicate 0101', 'Non Duplicate 0102', 'Non Duplicate 0103']]
# # a.columns = ['Timestamp', 'Deactivation', 'String In', 'Ladder Out']
# # a['Timestamp'] = pd.to_datetime(a['Timestamp'])
# # a['Timestamp'] = a['Timestamp'].dt.time
# # a = agg[['0102 ID', 'Time Delta']].dropna()
# # deacs = deacs.merge(right=a, left_on='0102 ID', right_on='0102 ID', how='left')
# # single_deacs = deacs[deacs.groupby('0102 ID').cumcount() == 0]
# #
# # import matplotlib.pyplot as plt
# # import matplotlib.style as style
# # style.use('ggplot')
# #
# # generator = range(0, 100, 10)
# # p_ticks = [i for i in generator]
# # percentiles = [f'{i}%' for i in generator]
# # b = pd.DataFrame(
# #     np.percentile(single_deacs['Time Delta'].dropna(), p_ticks),
# #     percentiles
# # )
# # b.iloc[0, 0] = 1
# # c = agg.loc[(agg['0102 Pace'] > 25) & (agg['Label'] == 0), '0102 Pace']
# # c = pd.DataFrame(
# #     np.percentile(c, p_ticks),
# #     percentiles
# # )
# # plt.plot(b, color='b')
# # plt.plot(c, color='r')
# # plt.xlabel('Percentage of data points')
# # plt.ylabel('Time (s)')
# # plt.show()
# #
# # agg[agg['0102 Pace'] > 8].groupby('0102 Pace').count()
# #
# # a = deacs[deacs['strings until next ladder'] == 0]
# # a['0102 ID clone'] = a['0102 ID']
# # c = a.groupby('0102 ID').count()
# # c['Time Delta'] = c['Time Delta'] - 1
# # np.sum(c['Time Delta'])
# # a = sensor_data[
# #     [
# #         'JOBNUM', 'Non Duplicate 0101', 'Non Duplicate 0102', 'Non Duplicate 0103',
# #     ]
# # ]
# #
# # a.columns = ['JOBNUM', 'Deactivation', 'String In', 'String Out']
# #
# #
# # agg = agg.merge(
# #     right=non_deacs[['0102 ID', 'ND 0 time delta']].reset_index(),
# #     left_on='0102 ID',
# #     right_on='0102 ID'
# # )
# #
# # a.columns = ['Deactivation', 'String In', 'String Out']
# #
# # a['strings plus/minus'] = a['strings since last ladder'].copy()
# # condition = a['strings until next ladder'] <= a['strings since last ladder']
# # indices = a[condition].index
# # a.loc[indices, 'strings plus/minus'] = a.loc[indices, 'strings until next ladder']
# #
# # # a.columns = ['Off', 'String In', 'Ladder Out', 'strings until next ladder', 'strings since last ladder']
# #
# # c = a.groupby('strings plus/minus').count().iloc[:, 1]
# # c = c.reset_index()
# # c.columns = ['strings plus/minus', 'Number Of Deactivations']
# # k = c.corr()
# # c.iloc[7, 1] = np.sum(c.iloc[7:, 1])
# # c = c.iloc[:8, :]
# # c.iloc[7, 0] = 'Greater than 7'
# #
# # c = a.groupby('strings since next ladder').count().iloc[:, 1]
# # c = c.reset_index()
# # c = c.iloc[:6, :]
# # c.iloc[:, 0] = c.iloc[:, 0].astype(float)
# # corr = c.corr()
# # c = c.iloc[:7, :]
# # c.iloc[7, 0] = 'Greater than 7'
# # c.iloc[6, 1] = 32
# # c.columns = ['String Since Last Ladder', 'Number of Deactivations']
# # d = a.groupby('strings until next ladder').count().iloc[:7, 0]
# # d = d.reset_index()
# # d = d.corr()
# # e = a.groupby('strings until next ladder').count().iloc[7:, 0].sum()
# # d = d.reset_index()
# # d.iloc[7, 0] = 7
# # d['strings until next ladder'] = d['strings until next ladder'].astype(float)
# #
# #
# #
# # corr = d.corr()
# # d.iloc[7, 1] = 232
# # e = c.copy()
# # e['Number of Deactivations'] = (e['Number of Deactivations'] / np.sum(e['Number of Deactivations'])) * 100
# # e.columns = ['Strings Since Last Ladder', 'Percentage of Deactivations']
# #
# # d = pd.concat([d, e], axis=0)
# # e = pd.Series([7, 232]).T
# # d.iloc[0, 7] = 7
# #
# #
# # def _calc_percentile(data):
# #     return pd.DataFrame(
# #         np.percentile(data, [i for i in range(1, 100)]),
# #         [f'{i}%' for i in range(1, 100)]
# #     )
# #
# #
# # def calc_percentiles(frames):
# #     """
# #     Calculates the percentiles from 0 to 99 for dataframe passed in the frames dict
# #     """
# #     output = pd.concat([_calc_percentile(frames[key]) for key in frames.keys()], axis=1)
# #     output.columns = [key for key in frames.keys()]
# #     return output
# #
# #
# # b = pd.DataFrame(
# #     np.percentile(deacs['rows until end'], [i for i in range(1, 100)]),
# #     [f'{i}%' for i in range(1, 100)]
# # )
# #
# # d = agg[(agg['0102 Pace'] >= 25) &
# #         (agg['Label'] == 0) &
# #         (agg['0103 non_unique ID'] > 1)]
# #
# # q = pd.DataFrame(
# #     np.percentile(d['rows until end'], [i for i in range(1, 100)]),
# #     [f'{i}%' for i in range(1, 100)]
# # )
#
# # condition = (deacs['Non Duplicate 0101'] == 1) & (deacs['rows until end'] > 5)
# # e = deacs[condition]
#
# # z = pd.DataFrame(
# #     np.percentile(e['rows since start'], [i for i in range(1, 100)]),
# #     [f'{i}%' for i in range(1, 100)]
# # )
# #
# # indices = agg['0102 ID'].isin(ids)
# # t = agg.merge(
# #     right=a,
# #     left_on='0102 ID',
# #     right_on='0102 ID',
# #     how='left'
# # )
# #
# # #
# # #
# # # from utils.sensor_data import data_preparation as sd
# # # from utils.work_table import data_preparation as wt
# # # from utils.load_data import api
# # #
# # # # work_table = api.get_erp()
# # # work_table = pd.read_csv(r'/home/james/work_table.csv', sep=',')
# # # work_table = api.prep_erp(work_table)
# # #
# # # wt_cleaner = wt.WorkTableCleaner(
# # #     work_table, stats=False,
# # #     remove_overlaps=wt.remove_all_overlaps,
# # #     ladder_filter=wt.filter_main_SW_and_2_CF_1405_auto_input
# # # )
# # # wt_prep = wt.PrepareWorkTable(None, False, wt_cleaner)
# # # work_table = wt_prep.prep_work_table()
# # #
# # #
# # # # data = api.get_sensor_data()
# # # data = pd.read_csv(r'/home/james/sensor_data.csv', sep=',')
# # #
# # # data['Date'] = pd.to_datetime(data['timestamp'])
# # # data = sd.filter_sensor_data(work_table, data)
# # # data['port'] = data['port'].astype(str)
# # # data['value'] = data['value'].astype(int)
# # # output = pd.DataFrame()
# # # output[['Date', 'JOBNUM']] = data[['Date', 'JOBNUM']]
# # #
# # # for x in range(1, 7):
# # #     values = data.loc[data['port'] == f'10{x}.0', 'value']
# # #     output.loc[values.index, f'Indgang 010{x}'] = values
# # #     output[f'Indgang 010{x}'] = output\
# # #         .groupby('JOBNUM')[f'Indgang 010{x}']\
# # #         .fillna(method='ffill')
# # #
# # # output['Indgang 0101'] = output['Indgang 0101'].fillna(1)
# # #
# # # indices = list()
# # # jobrefs = list()
# # # for x in range(1, 7):
# # #     temp = data.copy()
# # #     temp.loc[:, 'Duplicate index'] = data.index
# # #     temp = temp.loc[temp['port'] == f'102.0', :]
# # #     temp['prev_value'] = temp['value'].shift(1)
# # #     indices += temp.loc[
# # #         temp['value'] == temp['prev_value'],
# # #         'Duplicate index'
# # #     ].tolist()
# # #     jobrefs += temp.loc[
# # #         temp['value'] == temp['prev_value'],
# # #         'JOBREF'
# # #     ].tolist()
# # #
# # # data['prev_value'] = temp['prev_value']
# # # contaminated = data.loc[indices, :]
# # # uncontaminated_jobrefs = work_table.loc[~work_table['JOBREF'].isin(set(jobrefs)), :]
#
