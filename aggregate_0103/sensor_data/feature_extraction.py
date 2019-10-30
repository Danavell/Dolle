import numpy as np

import pandas as pd

from aggregate_0103 import aggregates as a
from utils.utils import get_dummy_products, make_column_arange
from utils.sensor_data import feature_extraction as fsd
from utils.sensor_data import data_preparation as sd


class BaseData1405FeatureExtractor:
    def __init__(self):
        self.data = None
        self._category = None


class MLFeatureExtractor0103:
    def __init__(self):
        self.data = None
        self._sensor_data = None
        self._work_table = None

    def feature_extraction(self, sensor_data, columns):
        self._sensor_data = fsd.feature_extraction_sensor_data(sensor_data, columns)
        self._sensor_data = fsd.calculate_pace(sensor_data, columns)
        self._sensor_data = fsd.ffill_0102_per_0103(self._sensor_data)

        self._sensor_data['0103 Group b-filled'] = make_column_arange(
            self._sensor_data, 'Non Duplicate 0103', fillna_groupby_col='JOBNUM'
        )
        self._sensor_data = a.make_n_length_jam_durations(self._sensor_data)

        self._sensor_data = sd.get_dummies_concat(self._sensor_data)
        reg_ex = r'^[A-Z]{2}[/][1-9][A-Z][/][1-9][A-Z][/][1-9][A-Z][/][1-9]{2}[A-Z]$'
        self.data = a.make_aggregates(self._sensor_data, reg_ex)


class StatsFeatureExtractor:
    def __init__(self, category):
        self.data = None

        self._category = category
        self._sensor_data = None
        self._work_table = None
        self._stats = None
        self._cols = None
        self._aggregate_stats = None

    def feature_extraction(self, work_table, sensor_data, columns):
        self._sensor_data = fsd.create_non_duplicates(sensor_data, columns, phantoms=False)
        self._sensor_data = fsd.calculate_pace(self._sensor_data, columns)
        self._stats = st.generate_statistics(
            self._sensor_data.copy(), work_table, columns
        )
        self._cols, self._stats = get_dummy_products(self._stats)
        strings_per_ladders = self.average_stat('Strings per Ladder')
        count_vs_expected = self.average_stat('0103 Count Vs Expected', drop=True)
        pace_in = self.average_stat(
            '0102 Pace median(s)', drop=True, output_col='0102 Pace(s)'
        )
        pace_out = self.average_stat(
            '0103 Pace median(s)', drop=True, output_col='0103 Pace(s)'
        )
        self._aggregate_stats = pd.concat([
                strings_per_ladders,
                count_vs_expected,
                pace_in,
                pace_out
            ], axis=1
        )
        self.data['category'] = self._category
        self.data['sensor_data'] = self._sensor_data
        self.data['work_table'] = self._work_table
        self.data['stats'] = self._stats,
        self.data['agg_stats'] = self._aggregate_stats

    def average_stat(self, target_col, drop=False, output_col=None):
        temp = []
        for column in self._cols:
            condition = self._stats[column] != 0
            filtered = self._stats.loc[condition, :].copy()
            temp.append([
                column, np.sum(filtered[target_col]) / len(filtered.index), np.median(filtered[target_col])
            ])
        if isinstance(output_col, str):
            target_col = output_col
        data = pd.DataFrame(temp, columns=['Product', f'Average {target_col}', f'Median {target_col}'])
        if drop:
            data.drop('Product', axis=1, inplace=True)
        return data


