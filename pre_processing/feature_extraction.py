import re

import numpy as np
import pandas as pd

import pre_processing.utils as ut
import pre_processing.config_settings as cs

from pre_processing.aggregate_0102 import aggregates as a_0102
from pre_processing.aggregate_0103 import aggregates as a_0103
from pre_processing.STATS_aggregate_0102 import aggregates as ast_0102
from utils.sensor_data import data_preparation as sd
from utils.sensor_data import feature_extraction as fsd
from utils.STATS import STATS as st
from utils.utils import make_column_arange


class BaseData1405FeatureExtractor:
    def __init__(self, _):
        self.data = dict()
        self._category = None

    def feature_extraction(self, work_table, sensor_data, _, __):
        self.data['sensor_data'] = sensor_data
        self.data['work_table'] = work_table


class BaseDataAdapter:
    def __init__(self, work_table, sensor_data):
        self.work_table = work_table
        self.sensor_data = sensor_data


class StatsFeatureExtractor:
    def __init__(self):
        self.data = dict()

    def feature_extraction(self, work_table, sensor_data, machine, _):
        columns = machine.data_generation_columns
        sensor_data = fsd.create_non_duplicates(sensor_data)
        sensor_data = fsd.calculate_pace(sensor_data, columns)

        columns = machine.generate_statistics
        stats = st.generate_statistics(sensor_data.copy(), work_table, columns)
        agg_dict = {
            'Product': 'first',
            'Job Length(s)': 'sum',
            'Strings per Ladder': ['median', 'mean'],
            '0103 Count Vs Expected': ['median', 'mean'],
            '0102 Pace median(s)': 'median',
            '0102 Pace avg(s)': 'mean',
            '0103 Pace median(s)': 'median',
            '0103 Pace avg(s)': 'mean'
        }
        self.data['sensor_data'] = sensor_data
        self.data['work_table'] = work_table
        self.data['stats'] = stats
        self.data['agg_stats'] = stats.groupby('Product').agg(agg_dict)


class StatsFeatureExtractor0102Agg:
    def __init__(self):
        self.data = dict()
        self.stats = None

    def feature_extraction(self, work_table, sensor_data, machine, meta):
        base = MLFeatureExtractor0102()
        base.feature_extraction(work_table, sensor_data, machine, meta)
        aggs = base.data
        sensor_data = aggs.pop('sensor_data')
        work_table = aggs.pop('work_table')

        products = []
        for key in aggs.keys():
            if not re.match(r'^all (\d) products$', key):
                condition = sensor_data.loc[:, key] == 1
                data = sensor_data.loc[condition, :].copy()
                products.append(key)
                total_duration = self.stats.loc[self.stats['Product'] == key, ['Job Length(s)']]\
                    .astype(float)\
                    .squeeze()
            else:
                data = sensor_data.copy()
                total_duration = self.stats.loc[self.stats['Product'].isin(products), ['Job Length(s)']]\
                    .astype(float)\
                    .squeeze()
                total_duration = np.sum(total_duration)

            agg = aggs[key].copy()
            agg.loc[:, 'rows until end'] = agg.groupby('0103 ID').cumcount(ascending=False) + 1
            agg.loc[:, 'rows since start'] = agg.groupby('0103 ID').cumcount() + 1

            no_deacs = agg.loc[agg['Non Duplicate 0101'] == 0, :]
            deacs_sd = data.loc[
                (data['Non Duplicate 0101'] == 1) & (data['0102 ID'].isin(agg['0102 ID'])),
                ['Date', '0102 ID']
            ]

            """
            Separate 0102 IDs with 1 and more than 1 deactivation. Slightly different functions
            will have to be applied to both. Therefore, it makes sense to separate them as 
            early as possible
            """
            single_deacs = ast_0102.calc_t_delta_and_merge(deacs_sd, agg, agg['Non Duplicate 0101'] == 1)
            multi_deacs = ast_0102.calc_t_delta_and_merge(
                deacs_sd, agg, agg['Non Duplicate 0101'] > 1, multi=True
            )

            agg = pd.concat([
                single_deacs,
                no_deacs,
                multi_deacs.loc[multi_deacs.groupby('0102 ID').cumcount() + 1 == 1, :]
            ], axis=0, sort=False)
            agg = agg.sort_values(['0102 ID', 'Date']).reset_index(drop=True)

            frames = {
                'ND: 0102 Pace': agg.loc[agg.loc[:, 'Label'] == 0, '0102 Pace'],
                'D: time delta': agg.loc[agg.loc[:, 'Label'] == 1, 'Time Delta'],
                'ND: until end >= 25': agg.loc[agg['0102 Pace'] >= 25, 'rows until end'],
                'D: until end': agg.loc[agg.loc[:, 'Label'] == 1, 'rows until end'],
                'ND: since start >= 25': agg.loc[agg['0102 Pace'] >= 25, 'rows since start'],
                'D: since start': agg.loc[agg.loc[:, 'Label'] == 1, 'rows since start'],
            }
            percentiles = ut.calc_percentiles(frames)

            self.data[key] = dict()
            self.data[key][key] = agg
            self.data[key][f'{key} percentiles'] = percentiles

            conf = ast_0102.confusion_matrix(agg, train=False)
            total_alarms = conf[0, 1] + conf[1, 1]
            avg_time_per_alarm = total_duration / total_alarms

            print(key)
            print('---------------')
            print(ast_0102.confusion_matrix(agg))
            print('---------------')
            print(f'precision: {conf[1, 1] / (conf[0, 1] + conf[1, 1])}')
            print(f'recall: {conf[1, 1] / (conf[1, 0] + conf[1, 1])}')
            print(f'average time between alarms: {avg_time_per_alarm}')
            print(f'correlation deacs 0102 Pace: {ast_0102.corr(percentiles)}')
            print('---------------')
            print('\n')

        self.data['sensor_data'] = sensor_data
        self.data['work_table'] = work_table


class MLFeatureExtractor0102:
    def __init__(self):
        self.data = None

    def feature_extraction(self, work_table, sensor_data, machine, meta):
        drop_first_rows = a_0102.drop_first_rows if meta.get('drop_first') else False

        columns = machine.data_generation_columns
        sensor_data = fsd.create_non_duplicates(sensor_data)
        sensor_data = fsd.calculate_pace(sensor_data, columns)
        sensor_data['0102 ID'] = make_column_arange(
            sensor_data, 'Non Duplicate 0102', fillna_groupby_col='JOBNUM'
        )
        sensor_data['0103 ID'] = make_column_arange(
            sensor_data, 'Non Duplicate 0103', fillna_groupby_col='JOBNUM'
        )
        sensor_data['Indgang 0101 time'] = fsd.calc_error_time(
            sensor_data, 'Indgang 0101', groupby_cols=['JOBNUM', '0102 ID']
        )

        sensor_data.loc[:, '0103 non_unique ID'] = fsd.make_ID(sensor_data, 3)
        sensor_data = sd.get_dummies_concat(sensor_data)

        self.data = ut.make_aggregates(
            sensor_data, cs.product_col_reg_ex, '0102 ID', cs.agg_funcs_0102, drop_first_rows
        )
        self.data['sensor_data'] = sensor_data
        self.data['work_table'] = work_table


class StatsFeatureExtractor0103Agg:
    def __init__(self):
        self.data = dict()

    def feature_extraction(self, work_table, sensor_data, machine, meta):
        base = MLFeatureExtractor0103()
        base.feature_extraction(work_table, sensor_data, machine, meta)

        sensor_data = base.data.pop('sensor_data')
        aggs = base.data
        for key in aggs.keys():
            if not re.match(r'^all (\d) products$', key):
                condition = sensor_data.loc[:, key] == 1
                data = sensor_data.loc[condition, :].copy()
            else:
                data = sensor_data.copy()

            agg = base.data[key].copy()

        self.data['work_table'] = base.data['work_table']


class MLFeatureExtractor0103:
    def __init__(self):
        self.data = None

    def feature_extraction(self, work_table, sensor_data, machine, meta):
        jam = meta.get('jam')
        drop_first_rows = a_0103.drop_first_rows if meta.get('drop_first') else False

        columns = machine.data_generation_columns
        sensor_data = fsd.create_non_duplicates(sensor_data)
        sensor_data = fsd.sensor_groupings(sensor_data)
        sensor_data = fsd.calculate_pace(sensor_data, columns)

        sensor_data['0103 ID'] = make_column_arange(
            sensor_data, 'Non Duplicate 0103', fillna_groupby_col='JOBNUM'
        )
        funcs = cs.base_agg_funcs_0103
        if jam:
            num = 20
            sensor_data = a_0103.make_n_length_jam_durations(sensor_data, num)
            for i in range(num, 1, -1):
                funcs[f'Sum 0102 Jam >= {i}'] = 'sum'

        sensor_data = sd.get_dummies_concat(sensor_data)
        self.data = ut.make_aggregates(
            sensor_data, cs.product_col_reg_ex, '0103 ID', funcs, drop_first_rows
        )
        self.data['sensor_data'] = sensor_data
        self.data['work_table'] = work_table
