import os

import pandas as pd

import numpy as np
from pre_processing.utils import BaseDataFactory

import pre_processing.config_settings as cs
import pre_processing.utils as ut
import utils.sensor_data.feature_extraction as fsd

from pre_processing.aggregate_0102 import aggregates as a_0102
from pre_processing.aggregate_0103 import aggregates as a_0103
from pre_processing.STATS_aggregate_0102 import aggregates as ast_0102
from utils.sensor_data import data_preparation as sd
from utils.utils import Machine1405, make_column_arange


BaseDataFactory.get_ladder_codes()
# # # for i in range(1, 6):
BaseDataFactory.factory(4, '01-01-18 to 01-01-19', fix_duplicates=True)

# BaseDataFactory.factory(2, '01-01-18 to 01-01-19', fix_duplicates=True)


# path = r'/home/james/Documents/Development/dolle_csvs/01-01-18 to 01-01-19/' \
#        r'Stats 1405 All ladders/agg_stats.csv'
#
# stats = pd.read_csv(path, sep=';').drop(0, axis=0)
# stats['Product'] = stats['Product'].str.replace('/', '-')
# products = pd.Series([
#     'CF-3D-3F-2B-12T', 'CF-3D-4F-3B-12T', 'CF-3D-4F-4B-12T', 'SW-3D-3F-3B-12T'
# ])
# output = stats.loc[stats['Product'].isin(products), ['Product', 'Job Length(s)']]
#
#
# CF_3D_3F_2B_12T_path = os.path.join(base_path, f'{products[0]}.csv')
# CF_3D_3F_2B_12T = None
#
# CF_3D_3F_3B_12T_path = os.path.join(base_path, f'{products[1]}.csv')
# CF_3D_3F_3B_12T = None
#
# CF_3D_4F_4B_12T_path = os.path.join(base_path, f'{products[2]}.csv')
# CF_3D_4F_4B_12T = None
#

base_path = r'/home/james/Documents/Development/dolle_csvs/01-01-18 to 01-01-19/' \
            r'Stats 1405: 0102, 1 SW, 2 CF, no overlaps'

path = os.path.join(base_path, 'all 3 products.csv')
agg = pd.read_csv(path, sep=';')
ast_0102.confusion_matrix(agg)
#
# #
# # machine = Machine1405()
# # columns = machine.data_generation_columns
# # sensor_data = fsd.create_non_duplicates(sensor_data)
# # sensor_data = fsd.calculate_pace(sensor_data, columns)
# # sensor_data['0102 ID'] = make_column_arange(
# #     sensor_data, 'Non Duplicate 0102', fillna_groupby_col='JOBNUM'
# # )
# # sensor_data['0103 ID'] = make_column_arange(
# #     sensor_data, 'Non Duplicate 0103', fillna_groupby_col='JOBNUM'
# # )
# # sensor_data.loc[:, '0103 non_unique ID'] = fsd.make_ID(sensor_data, 3)
# # sensor_data = sd.get_dummies_concat(sensor_data)
# #
# # reg_ex = r'^[A-Z]{2}[-][1-9][A-Z][-][1-9][A-Z][-][1-9][A-Z][-][1-9]{2}[A-Z]$'
# # drop_first_rows = False
# # aggs = ut.make_aggregates(
# #     sensor_data, reg_ex, '0102 ID', cs.agg_funcs_0102, drop_first_rows,
# # )
# #
# # agg = aggs['CF-3D-4F-4B-12T']
# # condition = sensor_data.loc[:, 'CF-3D-4F-4B-12T'] == 1
# # data = sensor_data.loc[condition, :].copy()
# # agg.loc[:, 'rows until end'] = agg.groupby('0103 ID').cumcount(ascending=False) + 1
# # agg.loc[:, 'rows since start'] = agg.groupby('0103 ID').cumcount() + 1
# #
# # no_deacs = agg.loc[agg['Non Duplicate 0101'] == 0, :]
# # no_deacs.loc[agg['Non Duplicate 0101'] == 0, 'Label'] = 0
# # deacs_sd = sensor_data.loc[
# #     (sensor_data['Non Duplicate 0101'] == 1) & (sensor_data['0102 ID'].isin(agg['0102 ID'])),
# #     ['Date', '0102 ID']
# # ]
# #
# # """
# # Separate 0102 IDs with 1 and more than 1 deactivation. Slightly different functions
# # will have to be applied to both. Therefore, it makes sense to separate them as
# # early as possible
# # """
# # single_deacs = ast_0102.calc_t_delta_and_merge(deacs_sd, agg, agg['Non Duplicate 0101'] == 1)
# # multi_deacs = ast_0102.calc_t_delta_and_merge(deacs_sd, agg, agg['Non Duplicate 0101'] > 1, multi=True)
# #
# # agg_2 = pd.concat([
# #     single_deacs,
# #     no_deacs,
# #     multi_deacs.loc[multi_deacs.groupby('0102 ID').cumcount() + 1 == 1, :]
# # ], axis=0, sort=False)
# #
# # agg_2 = agg_2.sort_values(['0102 ID', 'Date']).reset_index(drop=True)
# #
# # percentiles = ast_0102.calc_percentiles(agg_2)
# #
# # agg_2.loc[agg_2.loc[:, 'Label'] > 1, 'Label'] = 1
# #
# # true_pos = len(agg_2.loc[agg_2.loc[:, 'Label'] == 0].index)
# # false_neg = len(agg_2.loc[(agg_2.loc[:, '0102 Pace'] >= 25) & (agg_2.loc[:, 'Label'] == 0), :].index)
# # true_neg = len(agg_2.loc[(agg_2.loc[:, 'Time Delta'] >= 25) & (agg_2.loc[:, 'Label'] == 1)].index)
# # false_pos = len(agg_2.loc[agg_2.loc[:, 'Time Delta'] < 25].index)
# #
# # import numpy as np
# # confusion = np.array([
# #     [true_pos, false_neg],
# #     [false_pos, true_neg]
# # ])
# #
# # a = percentiles[['Non-Deactivations: 0102 Pace']]
# # a['Label'] = 0
# # a.columns = ['Time', 'Label']
# #
# # b = percentiles[['Deactivations: time delta']]
# # b['Label'] = 1
# # b.columns = ['Time', 'Label']
# # corr = pd.concat([a, b], axis=0, sort=False).corr()
# #
# # # multi_deacs = agg.loc[agg.loc[: 'Label'] > 1, :]
# # #
# # #
# # #
# # # all_deacs, aggs_singles, aggs_multis = ast_0102.add_unique_deactivations_to_0102_IDs(
# # #     agg, time_delta_aggs
# # # )
