import re

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

    def feature_extraction(self, work_table, sensor_data, _):
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

        sensor_data.loc[:, '0103 ID'] = fsd.make_ID(sensor_data, 3)
        sensor_data = sd.get_dummies_concat(sensor_data)

        reg_ex = r'^[A-Z]{2}[-][1-9][A-Z][-][1-9][A-Z][-][1-9][A-Z][-][1-9]{2}[A-Z]$'
        drop_first_rows = a_0102.drop_first_rows if drop_first_rows else None
        aggs = ut.make_aggregates(
            sensor_data, reg_ex, '0102 ID', cs.agg_funcs_0102, drop_first_rows
        )
        for key in aggs.keys():
            if not re.match(r'^all (\d) products$', key):
                condition = sensor_data.loc[:, key] == 1
                data = sensor_data.loc[condition, :].copy()
            else:
                data = sensor_data.copy()

            self.data[key] = aggs[key]

            time_delta_aggs = ast_0102.calc_time_since_string_in_and_deactivation(
                data, aggs[key]
            )
            all_deacs, aggs_singles, aggs_multis = ast_0102.add_unique_deactivations_to_0102_IDs(
                aggs[key], time_delta_aggs
            )
            self.data[f'{key} - all deacs'] = all_deacs
            self.data[f'{key} - single rows'] = aggs_singles
            self.data[f'{key} - multi rows'] = aggs_multis

            percentiles = ast_0102.calc_0103_Pace_and_string_deac_t_delta_percentiles(
                aggs[key], time_delta_aggs
            )
            self.data[f'{key} - percentiles'] = percentiles
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
