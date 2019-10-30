import os

import pandas as pd

from aggregate_0103.work_table import data_preparation as awt
from utils.sensor_data import data_preparation as sd
from utils.utils import get_csv_directory
from utils.work_table import data_preparation as wt


class BaseDataAdapter:
    def __init__(self, work_table, sensor_data):
        self.work_table = work_table
        self.sensor_data = sensor_data


class BaseData:
    def __init__(self, machine, sd_cleaner, remove_overlaps, ladder_filter):
        self.machine = machine
        self.columns = self.machine.data_generation_columns

        self._folder = None
        self._sd_cleaner = sd_cleaner
        self._remove_overlaps = remove_overlaps
        self._ladder_filter = ladder_filter

    def get_base_data(self, work_table, sensor_data, stats=False):
        wt_cleaner = awt.WorkTableCleaner(
            work_table, stats=stats, remove_overlaps=self._remove_overlaps, ladder_filter=self._ladder_filter
        )
        self._sd_cleaner.sensor_data = sensor_data
        wt_prep = wt.PrepareWorkTable(self.columns, stats, wt_cleaner)
        return sd.prepare_base_data(wt_prep, self._sd_cleaner)
