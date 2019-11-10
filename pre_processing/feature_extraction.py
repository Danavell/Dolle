import re

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
    def __init__(self):
        self.data = dict()
        self._category = None

    def feature_extraction(self, work_table, sensor_data, __, _):
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

    def feature_extraction(self, work_table, sensor_data, machine, drop_first_rows):
        columns = machine.data_generation_columns
        sensor_data = fsd.create_non_duplicates(sensor_data)
        sensor_data = fsd.calculate_pace(sensor_data, columns)
        sensor_data['0102 ID'] = make_column_arange(
            sensor_data, 'Non Duplicate 0102', fillna_groupby_col='JOBNUM'
        )
        sensor_data['0103 ID'] = make_column_arange(
            sensor_data, 'Non Duplicate 0103', fillna_groupby_col='JOBNUM'
        )
        sensor_data.loc[:, '0103 non_unique ID'] = fsd.make_ID(sensor_data, 3)
        sensor_data = sd.get_dummies_concat(sensor_data)

        reg_ex = r'^[A-Z]{2}[-][1-9][A-Z][-][1-9][A-Z][-][1-9][A-Z][-][1-9]{2}[A-Z]$'
        drop_first_rows = a_0102.drop_first_rows if drop_first_rows else None
        aggs = ut.make_aggregates(
            sensor_data, reg_ex, '0102 ID', cs.agg_funcs_0102, drop_first_rows,
        )
        for key in aggs.keys():
            if not re.match(r'^all (\d) products$', key):
                condition = sensor_data.loc[:, key] == 1
                data = sensor_data.loc[condition, :].copy()
            else:
                data = sensor_data.copy()

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

            agg_2 = pd.concat([
                single_deacs,
                no_deacs,
                multi_deacs.loc[multi_deacs.groupby('0102 ID').cumcount() + 1 == 1, :]
            ], axis=0, sort=False)

            agg_2 = agg_2.sort_values(['0102 ID', 'Date']).reset_index(drop=True)
            agg_2.loc[agg_2.loc[:, 'Label'] > 1, 'Label'] = 1

            frames = {
                'Non-Deactivations: 0102 Pace': aggs.loc[aggs.loc[:, 'Label'] == 0, '0102 Pace'],
                'Deactivations: time delta': aggs.loc[aggs.loc[:, 'Label'] == 1, 'Time Delta'],
                'Non-Deactivations: rows until end, 0102 pace >= 25': aggs.loc[
                    aggs.loc[:, '0102 Pace'] >= 25, 'rows until end'
                ],
                'Deactivations: rows until end': aggs.loc[aggs.loc[:, 'Label'] == 1, 'rows until end'],
            }
            percentiles = ut.calc_percentiles(frames)

            self.data['percentiles'] = percentiles
            self.data[key] = agg
            self.data[f' final {key}'] = agg_2

            print(f'{key} - {ast_0102.corr(percentiles)}')
            print('---------------')
            print(ast_0102.confusion_matrix(agg_2))
            print('---------------')
            print('\n')

        self.data['sensor_data'] = sensor_data
        self.data['work_table'] = work_table


class MLFeatureExtractor0102:
    def __init__(self):
        self.data = None

    def feature_extraction(self, work_table, sensor_data, machine, drop_first_rows=None):
        columns = machine.data_generation_columns
        sensor_data = fsd.create_non_duplicates(sensor_data)
        sensor_data = fsd.calculate_pace(sensor_data, columns)
        sensor_data['0102 ID'] = make_column_arange(
            sensor_data, 'Non Duplicate 0102', fillna_groupby_col='JOBNUM'
        )

        sensor_data.loc[:, '0103 ID'] = fsd.make_ID(sensor_data, 3)
        sensor_data = sd.get_dummies_concat(sensor_data)

        reg_ex = r'^[A-Z]{2}[/][1-9][A-Z][/][1-9][A-Z][/][1-9][A-Z][/][1-9]{2}[A-Z]$'
        drop_first_rows = a_0102.drop_first_rows if drop_first_rows else False
        self.data = ut.make_aggregates(
            sensor_data, reg_ex, '0102 ID', cs.agg_funcs_0102, drop_first_rows
        )

        self.data['sensor_data'] = sensor_data
        self.data['work_table'] = work_table


class MLFeatureExtractor0103:
    def __init__(self):
        self.data = None

    def feature_extraction(self, work_table, sensor_data, machine, drop_first_rows=None):
        columns = machine.data_generation_columns
        sensor_data = fsd.create_non_duplicates(sensor_data)
        sensor_data = fsd.sensor_groupings(sensor_data)
        sensor_data = fsd.calculate_pace(sensor_data, columns)

        sensor_data['0103 Group b-filled'] = make_column_arange(
            sensor_data, 'Non Duplicate 0103', fillna_groupby_col='JOBNUM'
        )
        sensor_data = a_0103.make_n_length_jam_durations(sensor_data)

        sensor_data = sd.get_dummies_concat(sensor_data)
        reg_ex = r'^[A-Z]{2}[/][1-9][A-Z][/][1-9][A-Z][/][1-9][A-Z][/][1-9]{2}[A-Z]$'
        drop_first_rows = a_0103.drop_first_rows if drop_first_rows else False
        self.data = ut.make_aggregates(
            sensor_data, reg_ex, '0103 ID', cs.agg_funcs_0103, drop_first_rows
        )
        self.data = a_0103.make_aggregates(sensor_data, reg_ex, drop_first_rows)
        self.data['sensor_data'] = sensor_data
        self.data['work_table'] = work_table
