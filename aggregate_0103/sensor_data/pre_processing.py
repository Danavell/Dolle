from aggregate_0103.sensor_data.base_data import BaseData
from aggregate_0103.sensor_data import feature_extraction as fe
from utils.utils import CSVReadWriter, Machine1405, PreProcess
from utils.sensor_data import data_preparation as sd
from utils.work_table import data_preparation as wt


pre_process = PreProcess(
    folder='01-01-18 to 01-01-19',
    category='MLAgg0103_1405: 3 Main Ladders',
    base_data=BaseData(
        machine=Machine1405(),
        sd_cleaner=sd.SensorDataCleaner1405(fix_duplicates=True),
        remove_overlaps=wt.remove_all_overlaps,
        ladder_filter=sd.filter_three_main_ladders_1405,
    ),
    feature_extractor=fe.MLFeatureExtractor0103(),
    read_writer=CSVReadWriter
)
pre_process.get_base_data()
pre_process.feature_extraction()
pre_process.save()

# from utils.sensor_data import feature_extraction as fsd
# from utils.utils import Machine1405, make_column_arange
# import aggregate_0103.aggregates as a
#
# machine = Machine1405()
# columns = machine.data_generation_columns
# data = dict()
# sensor_data = fsd.feature_extraction_sensor_data(sensor_data, columns)
# sensor_data = fsd.calculate_pace(sensor_data, columns)
# sensor_data = fsd.ffill_0102_per_0103(sensor_data)
#
# sensor_data['0103 Group b-filled'] = make_column_arange(
#     sensor_data, 'Non Duplicate 0103', fillna_groupby_col='JOBNUM'
# )
# sensor_data = a.make_n_length_jam_durations(sensor_data)
# sensor_data = sd.get_dummies_concat(sensor_data)
# reg_ex = r'^[A-Z]{2}[/][1-9][A-Z][/][1-9][A-Z][/][1-9][A-Z][/][1-9]{2}[A-Z]$'
# data['additional_files'] = a.make_aggregates(sensor_data, reg_ex)

# 'Base Data 1405: 3 main ladders',
# fe.MLFeatureExtractor0103(category='MLAgg0103_1405')

# def pre_process_agg_data_factory(data, base_dir):
#     """
#     1. MLAgg0103 3 Main Ladders,
#     2. Stats: all ladders, no overlaps
#     3. Base Data 1405: all ladders
#     4. Base Data 1405: 3 main ladders
#     """
#     if data == 1:
#         pre_process = PreProcess(
#             BaseData(
#                 dir=base_dir,
#                 machine=Machine1405(),
#                 sd_cleaner=sd.SensorDataCleaner1405(),
#                 remove_overlaps=wt.remove_all_overlaps,
#                 ladder_filter=filter_three_main_ladders_1405
#             ),
#             fe.MLFeatureExtractor0103(category=data),
#         )
#     elif data == 2:
#         pre_process = PreProcess(
#             BaseData(
#                 dir=base_dir,
#                 machine=Machine1405(),
#                 sd_cleaner=sd.SensorDataCleaner1405(),
#                 remove_overlaps=wt.remove_all_overlaps,
#                 ladder_filter=None
#             ),
#             fe.StatsFeatureExtractor(category=data)
#         )
#     elif data == 3:
#         pre_process = PreProcess(
#             BaseData(
#                 dir='01-01-18 to 01-01-19',
#                 machine=Machine1405(),
#                 sd_cleaner=sd.SensorDataCleaner1405(),
#                 remove_overlaps=wt.remove_all_overlaps,
#                 ladder_filter=None
#             ),
#             fe.BaseData1405FeatureExtractor(category=data),
#         )
#     elif data == 4:
#         pre_process = PreProcess(
#             BaseData(
#                 dir='01-01-18 to 01-01-19',
#                 machine=Machine1405(),
#                 sd_cleaner=sd.SensorDataCleaner1405(),
#                 remove_overlaps=wt.remove_all_overlaps,
#                 ladder_filter=filter_three_main_ladders_1405
#             ),
#             fe.BaseData1405FeatureExtractor(category=data),
#         )
#     else:
#         raise ValueError('data param invalid')
#
#     pre_process.get_base_data()
#     pre_process.feature_extraction()
#     pre_process.save()
#
#
# pre_process_agg_data_factory('MLAgg0103 3 Main Ladders', '01-01-18 to 01-01-19')