import os

from models import machines
import utils.work_table.data_preparation as wt
import utils.sensor_data.data_preparation as sd


machine = machines.Machine1405()
columns = machine.data_generation_columns
path = r'/home/james/Documents/Development/Dolle/csvs/'
wt_path = os.path.join(path, 'WORK_TABLE.csv')
sd_path = os.path.join(path, '01-01-18 to 01-01-19/datacollection.csv')

wt_cleaner = wt.CleanWorkTable(
    wt_path, columns, False, wt.CleanerThreeMainLadders()
)
sd_cleaner = sd.SensorDataCleaner1405()
work_table, sensor_data = sd.prepare_base_data(wt_cleaner, sd_cleaner, wt_path, sd_path)
