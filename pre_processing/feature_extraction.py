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
from utils.utils import make_column_arange_gte


regex = r'^all (\d) products$'


class BaseData1405FeatureExtractor:
    def __init__(self, _):
        self.data = dict()
        self._category = None

    def feature_extraction(self, work_table, sensor_data, machine, meta):
        self.data['sensor_data'] = sensor_data
        self.data['work_table'] = work_table


class BaseDataAdapter:
    def __init__(self, work_table, sensor_data):
        self.work_table = work_table
        self.sensor_data = sensor_data


class StatsFeatureExtractor:
    def __init__(self):
        self.data = dict()

    def feature_extraction(self, work_table, sensor_data, machine, meta):
        columns = machine.data_generation_columns
        sensor_data = fsd.create_non_duplicates(sensor_data)
        sensor_data = fsd.calculate_pace(sensor_data, columns)

        columns = machine.generate_statistics
        stats = st.generate_statistics(
            sensor_data.copy(), work_table, columns
        )
        agg_dict = {
            'Product': 'first',
            'Job Length(s)': 'sum',
            'No. Deactivations': 'sum',
            'Down Time(s)': ['sum', 'median', 'mean', 'std', 'max', 'min'],
            'Strings per Ladder': ['median', 'mean'],
            '0103 Count Vs Expected': ['median', 'mean'],
            '0102 Pace median(s)': 'median',
            '0102 Pace avg(s)': 'mean',
            '0103 Pace median(s)': 'median',
            '0103 Pace avg(s)': 'mean',
        }
        stats = stats.sort_values(by='Start Time')
        self.data['sensor_data'] = sensor_data
        self.data['work_table'] = work_table
        self.data['stats'] = stats
        agg_stats = stats.groupby('Product').agg(agg_dict)
        temp = agg_stats[('Job Length(s)', 'sum')] - agg_stats[('Down Time(s)', 'sum')]
        agg_stats[('Time Between Deactivations', 'mean')] = temp / agg_stats[('No. Deactivations', 'sum')]
        self.data['agg_stats'] = agg_stats
        deac_summary = agg_stats[
            [('Product', 'first'),
             ('Job Length(s)', 'sum'),
             ('No. Deactivations', 'sum'),
             ('Time Between Deactivations', 'mean'),
             ('Down Time(s)', 'sum'),
             ('Down Time(s)', 'median'),
             ('Down Time(s)', 'max'),
             ('Down Time(s)', 'min')]
        ]
        self.data['deactivations_summary'] = deac_summary


class StatsFeatureExtractor0102Agg:
    def __init__(self):
        self.data = dict()
        self.stats = None

    def feature_extraction(self, work_table, sensor_data, machine, meta):
        n = meta.get('n', 25)
        base = MLFeatureExtractor0102()
        base.feature_extraction(work_table, sensor_data, machine, meta)
        aggs = base.data
        sensor_data = aggs.pop('sensor_data')
        work_table = aggs.pop('work_table')
        output = None

        for key in aggs.keys():
            if not re.match(regex, key):
                total_duration = self.stats\
                    .loc[self.stats['Product'] == key, ['Job Length(s)']]\
                    .astype(float)\
                    .squeeze()
                p_sensor_data = sensor_data[sensor_data['NAME'] == key].copy()
            else:
                products = [
                    key for key in aggs.keys() if not re.match(regex, key)
                ]
                total_duration = self.stats\
                    .loc[self.stats['Product'].isin(products), ['Job Length(s)']]\
                    .astype(float)\
                    .squeeze()
                total_duration = np.sum(total_duration)
                p_sensor_data = sensor_data.copy()

            agg = aggs[key].copy()
            agg.loc[:, 'strings since last ladder'] = agg.groupby('0103 ID')\
                .cumcount() + 1
            agg.loc[:, 'strings until next ladder'] = agg.groupby('0103 ID')\
                .cumcount(ascending=False) + 1

            deacs, p_sensor_data = ast_0102\
                .returns_deacs_strings_before_or_after_ladder_out(
                agg, p_sensor_data
            )

            frames = {
                'ND: 0102 Pace': agg.loc[
                    agg.loc[:, 'Label'] == 0, '0102 Pace'
                ],
                'D: time delta': agg.loc[
                    agg.loc[:, 'Label'] == 1, 'Time Delta'
                ],
                'ND: sum hi pace 30 r': agg.loc[
                    (agg['Label'] == 0) &
                    (agg['0102 Pace'] >= n), f'0102 Sum Pace >= {n}'],
                'D: sum hi pace 30 r': agg.loc[
                    (agg['Label'] == 1) &
                    (agg['0102 Pace'] >= n), f'0102 Sum Pace >= {n}'],
                'ND: Time since 0103 hi': agg.loc[
                    (agg['Label'] == 0) &
                    (agg['0102 Pace'] >= n), 'Time Since Last 0103'
                ],
                'D: Time since 0103': agg.loc[
                    agg['Label'] == 1, 'Time Since Last 0103'],
                f'ND: until end >= {n}': agg.loc[
                    (agg['0102 Pace'] >= n) &
                    (agg['Label'] == 0) &
                    (agg['0103 non_unique ID'] > 1), 'strings until next ladder'
                ],
                'D: until end': deacs['strings until next ladder'],
                f'ND: since start >= {n}': agg.loc[
                    (agg['0102 Pace'] >= n) &
                    (agg['Label'] == 0) &
                    (agg['0103 non_unique ID'] > 1), 'strings since last ladder'
                ],
                'D: since start': deacs['strings since last ladder'],
            }
            percentiles = ut.calc_percentiles(frames)

            self.data[key] = dict()
            self.data[key][key] = agg
            self.data[key][f'{key} percentiles'] = percentiles

            # frames = [output, p_sensor_data]
            # output = pd.concat(frames) if output else p_sensor_data

            conf = ast_0102.confusion_matrix(agg, train=False)
            total_alarms = conf[0, 1] + conf[1, 1]
            avg_time_per_alarm = total_duration / total_alarms

            print(key)
            print('---------------')
            print(ast_0102.confusion_matrix(agg, n=n))
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
        self.data = dict()

    def feature_extraction(self, work_table, sensor_data, machine, meta):
        drop = a_0102.drop_first_rows if meta.get('drop_first') else False
        n = meta.get('n', 25)

        columns = machine.data_generation_columns
        sensor_data = fsd.create_non_duplicates(sensor_data)
        # sensor_data = fsd.calculate_pace(sensor_data, columns)
        sensor_data['0102 ID'] = make_column_arange_gte(
            sensor_data, 'Non Duplicate 0102', fillna_groupby_col='JOBNUM'
        )
        sensor_data['0103 ID'] = make_column_arange_gte(
            sensor_data, 'Non Duplicate 0103', fillna_groupby_col='JOBNUM'
        )
        for i in range(2, 4):
            sensor_data[f'010{i} ID'] = sensor_data[f'010{i} ID'].fillna(-1)
            sensor_data[f'prev_010{i} ID'] = sensor_data\
                .groupby('JOBNUM')[f'010{i} ID']\
                .shift(1)

        sensor_data = ut.give_unique_010n_ids_end_jobnum(sensor_data, 2)
        sensor_data = ut.give_unique_010n_ids_end_jobnum(sensor_data)
        sensor_data = a_0102.calc_time_delta_last_ladder_out(sensor_data)

        sensor_data['Indgang 0101 time'] = fsd.calc_error_time(
            sensor_data, 'Indgang 0101', groupby_cols=['JOBNUM', '0102 ID']
        )

        sensor_data.loc[:, '0103 non_unique ID'] = fsd.make_ID(
            sensor_data, 3
        )
        sensor_data = sd.get_dummies_concat(sensor_data)
        aggs = ut.make_aggregates(
            sensor_data, cs.product_col_reg_ex, '0102 ID',
            cs.agg_funcs_0102, drop
        )
        for key in aggs.keys():
            if not re.match(regex, key):
                condition = sensor_data.loc[:, key] == 1
                data = sensor_data.loc[condition, :].copy()
            else:
                data = sensor_data.copy()

            agg = aggs[key].copy()
            agg = ut.calc_time_delta_between_deac_and_010n(data, agg, 2)

            """
            Creates unique IDs for all aggregates that have an 0102 pace 
            >= n. Also creates a column with the value of 1 at the same row. 
            This intended for summing the number of rows n_num_rows before 
            each unique 0102 pace >= n ID
            """
            indices = agg[agg['0102 Pace'] >= n].index
            agg.loc[indices, f'0102 Pace >= {n} ID'] = np.arange(
                1, len(indices) + 1
            )
            agg[f'0102 Pace >= {n} ID'] = agg[f'0102 Pace >= {n} ID']\
                .fillna(0)\
                .astype(int)
            agg[f'0102 Pace >= {n} Count'] = 0
            agg.loc[indices, f'0102 Pace >= {n} Count'] = 1
            func = a_0102.sum_num_pace_ins_larger_than_n
            agg = a_0102.deacs_roll(agg, func, n)

            self.data[key] = agg

        self.data['sensor_data'] = sensor_data
        self.data['work_table'] = work_table


class StatsFeatureExtractor0103Agg:
    def __init__(self):
        self.data = dict()

    def feature_extraction(self, work_table, sensor_data, machine, meta):
        base = MLFeatureExtractor0103()
        base.feature_extraction(work_table, sensor_data, machine, meta)

        sensor_data = base.data.pop('sensor_data')
        work_table = base.data.pop('work_table')
        aggs = base.data
        for key in aggs.keys():
            if not re.match(regex, key):
                condition = sensor_data.loc[:, key] == 1
                data = sensor_data.loc[condition, :].copy()
            else:
                data = sensor_data.copy()

            agg = base.data[key].copy()
            agg = ut.calc_time_delta_between_deac_and_010n(data, agg, 3)

            frames = {
                'ND: 0103 Pace': agg.loc[
                    agg.loc[:, 'Label'] == 0, '0103 Pace'
                ],
                'D: time delta': agg.loc[
                    agg.loc[:, 'Label'] == 1, 'Time Delta'
                ],
            }
            percentiles = ut.calc_percentiles(frames)
            self.data[key] = dict()
            self.data[key][key] = agg
            self.data[key][f'{key} percentiles'] = percentiles

        self.data['sensor_data'] = sensor_data
        self.data['work_table'] = work_table


class MLFeatureExtractor0103:
    def __init__(self):
        self.data = None

    def feature_extraction(self, work_table, sensor_data, machine, meta):
        # jam = meta.get('jam', 20)
        drop = a_0103.drop_first_rows if meta.get('drop_first') else False

        columns = machine.data_generation_columns
        sensor_data = fsd.create_non_duplicates(sensor_data)
        sensor_data = fsd.sensor_groupings(sensor_data)
        sensor_data = fsd.calculate_pace(sensor_data, columns)

        sensor_data['0103 ID'] = make_column_arange_gte(
            sensor_data, 'Non Duplicate 0103', fillna_groupby_col='JOBNUM'
        )
        
        label = 'Downtime Label'
        sensor_data[label] = 0
        condition = (sensor_data['0103 ID'] > 0) & (sensor_data['0103 Pace'] > 90)
        sensor_data.loc[condition, label] = 1
        
        condition = sensor_data['Non Duplicate 0101'] == 1
        deacs = sensor_data[condition][['Non Duplicate 0101']].copy()
        deacs['0101 Group'] = np.arange(1, len(deacs.index) + 1)
        sensor_data.drop('0101 Group', axis=1)
        sensor_data['0101 Group'] = deacs['0101 Group']
        funcs = cs.base_agg_funcs_0103

        sensor_data = sd.get_dummies_concat(sensor_data)
        self.data = ut.make_aggregates(
            sensor_data, cs.product_col_reg_ex, '0103 ID', funcs, drop
        )
        self.data['sensor_data'] = sensor_data
        self.data['work_table'] = work_table
