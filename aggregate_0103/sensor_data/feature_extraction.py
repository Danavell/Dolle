from aggregate_0103 import aggregates as a
from utils.utils import make_column_arange
from utils.sensor_data import feature_extraction as fsd
from utils.sensor_data import data_preparation as sd


class BaseData1405FeatureExtractor:
    def __init__(self):
        self.data = dict()
        self._category = None

    def feature_extraction(self, work_table, sensor_data):
        self.data['sensor_data'] = sensor_data
        self.data['work_table'] = work_table


class MLFeatureExtractor0103:
    def __init__(self):
        self.data = None

    def feature_extraction(self, work_table, sensor_data, machine):
        columns = machine.data_generation_columns
        sensor_data = fsd.feature_extraction_sensor_data(sensor_data, columns)
        sensor_data = fsd.calculate_pace(sensor_data, columns)
        sensor_data = fsd.ffill_0102_per_0103(sensor_data)

        sensor_data['0103 Group b-filled'] = make_column_arange(
            sensor_data, 'Non Duplicate 0103', fillna_groupby_col='JOBNUM'
        )
        sensor_data = a.make_n_length_jam_durations(sensor_data)

        sensor_data = sd.get_dummies_concat(sensor_data)
        reg_ex = r'^[A-Z]{2}[/][1-9][A-Z][/][1-9][A-Z][/][1-9][A-Z][/][1-9]{2}[A-Z]$'
        self.data = a.make_aggregates(sensor_data, reg_ex)
        self.data['sensor_data'] = sensor_data
        self.data['work_table'] = work_table
