import numpy as np

import pandas as pd

from aggregate_0103 import aggregates as a
from utils.utils import get_dummy_products, make_column_arange
from utils.sensor_data import feature_extraction as fsd
from utils.sensor_data import data_preparation as sd
from utils.STATS import STATS as st


class BaseData1405FeatureExtractor:
    def __init__(self):
        self.data = None
        self._category = None


class MLFeatureExtractor0103:
    def __init__(self):
        self.data = None
        self._sensor_data = None
        self._work_table = None

    def feature_extraction(self, _, sensor_data, machine):
        columns = machine.data_generation_columns
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
    def __init__(self):
        self.data = None
        self._sensor_data = None
        self._work_table = None

    def feature_extraction(self, work_table, sensor_data, machine):
        columns = machine.data_generation_columns
        self._sensor_data = fsd.create_non_duplicates(sensor_data, columns, phantoms=False)
        self._sensor_data = fsd.calculate_pace(self._sensor_data, columns)

        columns = machine.generate_statistics
        stats = st.generate_statistics(self._sensor_data.copy(), work_table, columns)
        agg_dict = {
            'Job Length(s)': 'sum',
            'Strings per Ladder': ['median', 'mean'],
            '0103 Count Vs Expected': ['median', 'mean'],
            '0102 Pace median(s)': 'median',
            '0102 Pace avg(s)': 'mean',
            '0103 Pace median(s)': 'median',
            '0103 Pace avg(s)': 'mean'
        }
        self.data = dict()
        self.data['stats'] = stats
        self.data['agg_stats'] = stats.groupby('Product').agg(agg_dict)

