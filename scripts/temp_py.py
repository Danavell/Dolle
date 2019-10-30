import os

from models import machines
import utils.work_table.data_preparation as wt
import utils.sensor_data.data_preparation as sd
from utils.utils import get_csv_directory


base_data = BaseDataThreeMainLadders1405(r'01-01-18 to 01-01-19')
base_data.get_base_data()