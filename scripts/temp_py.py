import os

from models import machines
import utils.work_table.data_preparation as wt
import utils.sensor_data.data_preparation as sd
from utils.utils import get_csv_directory


class SettingsThreeMainLadders1405:
    def __init__(self, wt_csv, sd_csv, path='/home/james/Documents/Development/dolle_csvs'):
        self.machine = machines.Machine1405()
        self.work_table = None
        self.sensor_data = None

        self._columns = self.machine.data_generation_columns
        self._wt_path = os.path.join(path, wt_csv)
        self._sd_path = os.path.join(path, sd_csv)

    def get_base_data(self, stats=False):
        wt_cleaner = wt.CleanWorkTable(
            self._wt_path, self._columns, stats, wt.CleanerThreeMainLadders()
        )
        sd_cleaner = sd.SensorDataCleaner1405()
        self.work_table, self.sensor_data = sd.prepare_base_data(
            wt_cleaner, sd_cleaner, self._wt_path, self._sd_path
        )

base_data = SettingsThreeMainLadders1405('WORK_TABLE.csv', r'01-01-18 to 01-01-19/datacollection.csv')
base_data.get_base_data()