import os

from models import machines
import utils.work_table.data_preparation as wt
import utils.sensor_data.data_preparation as sd
from utils.utils import get

class PreProcess:
    def __init__(self, base_data, feature_extractor, persister):
        self._base_data = base_data
        self._feature_extractor = feature_extractor
        self._persister = persister

    def get_base_data(self, stats=False):
        self._base_data.get_base_data(stats=stats)

    def feature_extraction(self):
        self._feature_extractor.extract_features(
            self._base_data.sensor_data
        )

    def save(self):
        self._persister.save(**self._feature_extractor.data)


class SettingsThreeMainLadders1405:
    def __init__(self, wt_csv, sd_csv):
        machine = machines.Machine1405()
        columns = machine.data_generation_columns
        path =
path = r'/home/james/Documents/Development/Dolle/csvs/'
wt_path = os.path.join(path, 'WORK_TABLE.csv')
sd_path = os.path.join(path, '01-01-18 to 01-01-19/datacollection.csv')

wt_cleaner = wt.CleanWorkTable(
    wt_path, columns, False, wt.CleanerThreeMainLadders()
)
sd_cleaner = sd.SensorDataCleaner1405()
work_table, sensor_data = sd.prepare_base_data(wt_cleaner, sd_cleaner, wt_path, sd_path)
