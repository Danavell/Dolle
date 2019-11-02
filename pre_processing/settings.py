from pre_processing import feature_extraction as fe
from utils.sensor_data import data_preparation as sd
from utils.STATS import STATS as st
from utils.work_table import data_preparation as wt
from utils import utils as ut

settings = {
    'Stats: all ladders, no overlaps': {
        'category': 'Stats 1405 All ladders',
        'machine': ut.Machine1405(),
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_SW_or_CF_1405,
        ),
        'feature_extractor': st.StatsFeatureExtractor(),
        'sd_cleaner': sd.SensorDataCleaner1405,
        'stats': True
    },
    'MLAgg0103 1405: 1 SW, no overlaps': {
        'category': 'MLAgg0103 1405: 1 SW, no overlaps',
        'machine': ut.Machine1405(),
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_1405_auto_input,
        ),
        'feature_extractor': fe.MLFeatureExtractor0103(),
        'sd_cleaner': sd.SensorDataCleaner1405,
    },
    'MLAgg0103 1405: 1 SW, no overlaps, drop first rows': {
        'category': 'MLAgg0103 1405: 1 SW, no overlaps',
        'machine': ut.Machine1405(),
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_1405_auto_input,
        ),
        'feature_extractor': fe.MLFeatureExtractor0103(),
        'sd_cleaner': sd.SensorDataCleaner1405,
        'drop_first_rows': True
    },
    'MLAgg0103 1405: 1 SW, 2 CF, no overlaps': {
        'category': 'MLAgg0103 1405: 1 SW, 2 CF, no overlaps',
        'machine': ut.Machine1405(),
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_and_2_CF_1405_auto_input,
        ),
        'feature_extractor': fe.MLFeatureExtractor0103(),
        'sd_cleaner': sd.SensorDataCleaner1405,
    },
    'MLAgg0103 1405: 1 SW, 2 CF, no overlaps, drop first rows': {
        'category': 'MLAgg0103 1405: 1 SW, 2 CF, no overlaps',
        'machine': ut.Machine1405(),
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_and_2_CF_1405_auto_input,
        ),
        'feature_extractor': fe.MLFeatureExtractor0103(),
        'sd_cleaner': sd.SensorDataCleaner1405,
        'drop_first_rows': True
    },
    'MLAgg0103 1405: 1 SW, 3 CF, no overlaps': {
        'category': 'MLAgg0103 1405: 1 SW, 3 CF, no overlaps',
        'machine': ut.Machine1405(),
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_and_3_CF_1405_auto_input,
        ),
        'feature_extractor': fe.MLFeatureExtractor0103(),
        'sd_cleaner': sd.SensorDataCleaner1405,
    },
    'MLAgg0103 1405: 1 SW, 3 CF, no overlaps, drop first rows': {
        'category': 'MLAgg0103 1405: 1 SW, 3 CF, no overlaps',
        'machine': ut.Machine1405(),
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_and_3_CF_1405_auto_input,
        ),
        'feature_extractor': fe.MLFeatureExtractor0103(),
        'sd_cleaner': sd.SensorDataCleaner1405,
        'drop_first_rows': True
    },
    'MLAgg0103 1405: 3 CF, no overlaps': {
        'category': 'MLAgg0103 1405: 3 CF, no overlaps',
        'machine': ut.Machine1405(),
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_3_main_CF_ladders_1405_auto_input,
        ),
        'feature_extractor': fe.MLFeatureExtractor0103(),
        'sd_cleaner': sd.SensorDataCleaner1405,
    },
    'MLAgg0103 1405: 3 CF, no overlaps, drop first rows': {
        'category': 'MLAgg0103 1405: 3 CF, no overlaps',
        'machine': ut.Machine1405(),
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_3_main_CF_ladders_1405_auto_input,
        ),
        'feature_extractor': fe.MLFeatureExtractor0103(),
        'sd_cleaner': sd.SensorDataCleaner1405,
        'drop_first_rows': True
    },
    'BaseData 1405: 1 SW, no overlaps': {
        'category': 'BaseData 1405: 1 SW, no overlaps',
        'machine': ut.Machine1405(),
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_ladder_1405_auto_input,
        ),
        'feature_extractor': fe.BaseData1405FeatureExtractor(),
        'sd_cleaner': sd.SensorDataCleaner1405,
        'base': True
    },
    'BaseData 1405: 1 SW, 2 CF, no overlaps': {
        'category': 'BaseData 1405: 1 SW, 2 CF, no overlaps',
        'machine': ut.Machine1405(),
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_and_2_CF_1405_auto_input,
        ),
        'feature_extractor': fe.BaseData1405FeatureExtractor(),
        'sd_cleaner': sd.SensorDataCleaner1405,
        'base': True
    },
    'BaseData 1405: 1 SW, 3 CF, no overlaps': {
        'category': 'BaseData 1405: 1 SW, 3 CF, no overlaps',
        'machine': ut.Machine1405(),
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_and_3_CF_1405_auto_input,
        ),
        'feature_extractor': fe.BaseData1405FeatureExtractor(),
        'sd_cleaner': sd.SensorDataCleaner1405,
        'base': True
    },
    'BaseData 1405: 1 CF, no overlaps': {
        'category': 'BaseData 1405: 1 CF, no overlaps',
        'machine': ut.Machine1405(),
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_1405_auto_input,
        ),
        'feature_extractor': fe.BaseData1405FeatureExtractor(),
        'sd_cleaner': sd.SensorDataCleaner1405,
        'base': True
    },
    'BaseData 1405: all ladders, no overlaps': {
        'category': 'BaseData 1405: 1 CF, no overlaps',
        'machine': ut.Machine1405(),
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_SW_or_CF_1405,
        ),
        'feature_extractor': fe.BaseData1405FeatureExtractor(),
        'sd_cleaner': sd.SensorDataCleaner1405,
        'base': True
    },
}

