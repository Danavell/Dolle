from aggregate_0103.sensor_data.base_data import BaseData
from aggregate_0103.sensor_data import feature_extraction as fe
from utils.utils import CSVReadWriter, Machine1405, PreProcess
from utils.sensor_data import data_preparation as sd
from utils.work_table import data_preparation as wt


# pre_process = PreProcess(
#     folder='01-01-18 to 01-01-19',
#     category='MLAgg0103_1405: 3 Main Ladders',
#     base_data=BaseData(
#         machine=Machine1405(),
#         sd_cleaner=sd.SensorDataCleaner1405(fix_duplicates=True),
#         remove_overlaps=wt.remove_all_overlaps,
#         ladder_filter=sd.filter_three_main_ladders_1405_auto_input,
#     ),
#     feature_extractor=fe.MLFeatureExtractor0103(),
#     read_writer=CSVReadWriter
# )
pre_process = PreProcess(
    folder='01-01-18 to 01-01-19',
    category='Stats All Ladders',
    machine=Machine1405(),
    base_data=BaseData(
        sd_cleaner=sd.SensorDataCleaner1405(fix_duplicates=True),
        remove_overlaps=wt.remove_all_overlaps,
        ladder_filter=sd.filter_SW_or_CF_1405,
    ),
    feature_extractor=fe.StatsFeatureExtractor(),
    read_writer=CSVReadWriter
)

pre_process.get_base_data()
pre_process.feature_extraction()
pre_process.save()


# 'Base Data 1405: 3 main ladders',
# fe.MLFeatureExtractor0103(category='MLAgg0103_1405')

# def pre_process_agg_data_factory(data, base_dir):
#     """
#     1. MLAgg0103 3 Main Ladders,
#     2. Stats: all ladders, no overlaps
#     3. Base Data 1405: all ladders, no overlaps
#     4. Base Data 1405: 3 main ladders, no overlaps
#     """
#     if data == 1:
#         remove_overlaps = wt.remove_all_overlaps
#         ladder_filter = sd.filter_three_main_ladders_1405
#         feature_extractor = fe.MLFeatureExtractor0103()
#
#     elif data == 2:
#         remove_overlaps = wt.remove_all_overlaps,
#         ladder_filter = sd.filter_SW_or_CF_1405
#         feature_extractor = fe.StatsFeatureExtractor()
#
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