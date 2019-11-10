from pre_processing import feature_extraction as fe
from utils import utils as ut
from utils.sensor_data import data_preparation as sd
from utils.work_table import data_preparation as wt

settings = {
    'Stats 1405: all ladders, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_SW_or_CF_1405,
        ),
        'feature_extractor': fe.StatsFeatureExtractor0102Agg(),
        'sd_cleaner': sd.SensorDataCleaner1405,
    },
    'Stats 1405: 0102, 1 SW, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_1405_auto_input,
        ),
        'feature_extractor': fe.StatsFeatureExtractor0102Agg,
        'sd_cleaner': sd.SensorDataCleaner1405,
    },
    'Stats 1405: 0102, 1 SW, no overlaps, drop first rows': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_1405_auto_input,
        ),
        'feature_extractor': fe.StatsFeatureExtractor0102Agg,
        'sd_cleaner': sd.SensorDataCleaner1405,
    },
    'Stats 1405: 0102, 1 SW, 2 CF, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_and_2_CF_1405_auto_input,
        ),
        'feature_extractor': fe.StatsFeatureExtractor0102Agg,
        'sd_cleaner': sd.SensorDataCleaner1405,
    },
    'Stats 1405: 0102, 1 SW, 2 CF, no overlaps, drop first rows': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_and_2_CF_1405_auto_input,
        ),
        'feature_extractor': fe.StatsFeatureExtractor0102Agg,
        'sd_cleaner': sd.SensorDataCleaner1405,
    },
    'Stats 1405: 0102, 1 SW, 3 CF, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_and_3_CF_1405_auto_input,
        ),
        'feature_extractor': fe.StatsFeatureExtractor0102Agg,
        'sd_cleaner': sd.SensorDataCleaner1405,
    },
    'Stats 1405: 0102, 1 SW, 3 CF, no overlaps, drop first rows': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_and_3_CF_1405_auto_input,
        ),
        'feature_extractor': fe.StatsFeatureExtractor0102Agg,
        'sd_cleaner': sd.SensorDataCleaner1405,
    },
    'Stats 1405: 0102, 3 CF, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_3_main_CF_ladders_1405_auto_input,
        ),
        'feature_extractor': fe.StatsFeatureExtractor0102Agg,
        'sd_cleaner': sd.SensorDataCleaner1405,
    },
    'Stats 1405: 0102, 3 CF, no overlaps, drop first rows': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_3_main_CF_ladders_1405_auto_input,
        ),
        'feature_extractor': fe.StatsFeatureExtractor0102Agg,
        'sd_cleaner': sd.SensorDataCleaner1405,
    },
    'MLAgg0103 1405: 1 SW, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_1405_auto_input,
        ),
        'feature_extractor': fe.MLFeatureExtractor0103,
        'sd_cleaner': sd.SensorDataCleaner1405,
    },
    'MLAgg0103 1405: 1 SW, no overlaps, drop first rows': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_1405_auto_input,
        ),
        'feature_extractor': fe.MLFeatureExtractor0103,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'drop_first_rows': True
    },
    'MLAgg0103 1405: 1 SW, 2 CF, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_and_2_CF_1405_auto_input,
        ),
        'feature_extractor': fe.MLFeatureExtractor0103,
        'sd_cleaner': sd.SensorDataCleaner1405,
    },
    'MLAgg0103 1405: 1 SW, 2 CF, no overlaps, drop first rows': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_and_2_CF_1405_auto_input,
        ),
        'feature_extractor': fe.MLFeatureExtractor0103,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'drop_first_rows': True
    },
    'MLAgg0103 1405: 1 SW, 3 CF, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_and_3_CF_1405_auto_input,
        ),
        'feature_extractor': fe.MLFeatureExtractor0103,
        'sd_cleaner': sd.SensorDataCleaner1405,
    },
    'MLAgg0103 1405: 1 SW, 3 CF, no overlaps, drop first rows': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_and_3_CF_1405_auto_input,
        ),
        'feature_extractor': fe.MLFeatureExtractor0103,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'drop_first_rows': True
    },
    'MLAgg0103 1405: 3 CF, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_3_main_CF_ladders_1405_auto_input,
        ),
        'feature_extractor': fe.MLFeatureExtractor0103,
        'sd_cleaner': sd.SensorDataCleaner1405,
    },
    'MLAgg0103 1405: 3 CF, no overlaps, drop first rows': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_3_main_CF_ladders_1405_auto_input,
        ),
        'feature_extractor': fe.MLFeatureExtractor0103,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'drop_first_rows': True
    },
    'BaseData 1405: 1 CF, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_1405_auto_input,
        ),
        'feature_extractor': fe.BaseData1405FeatureExtractor,
        'sd_cleaner': sd.SensorDataCleaner1405,
    },
    'BaseData 1405: 1 SW, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_ladder_1405_auto_input,
        ),
        'feature_extractor': fe.BaseData1405FeatureExtractor,
        'sd_cleaner': sd.SensorDataCleaner1405,
    },
    'BaseData 1405: 1 SW, 2 CF, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_and_2_CF_1405_auto_input,
        ),
        'feature_extractor': fe.BaseData1405FeatureExtractor,
        'sd_cleaner': sd.SensorDataCleaner1405,
    },
    'BaseData 1405: 1 SW, 3 CF, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_and_3_CF_1405_auto_input,
        ),
        'feature_extractor': fe.BaseData1405FeatureExtractor,
        'sd_cleaner': sd.SensorDataCleaner1405,
    },
    'BaseData 1405: all ladders, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_SW_or_CF_1405,
        ),
        'feature_extractor': fe.BaseData1405FeatureExtractor,
        'sd_cleaner': sd.SensorDataCleaner1405,
    },
    'BaseData 1405: 3 CF, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_3_main_CF_ladders_1405_auto_input,
        ),
        'feature_extractor': fe.BaseData1405FeatureExtractor,
        'sd_cleaner': sd.SensorDataCleaner1405,
    },
    'OriginalData Cleaned 1405: 1 CF, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_1405_auto_input,
        ),
        'feature_extractor': fe.BaseData1405FeatureExtractor,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'base': True
    },
    'OriginalData Cleaned: 1 SW, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_ladder_1405_auto_input,
        ),
        'feature_extractor': fe.BaseData1405FeatureExtractor,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'base': True
    },
    'OriginalData Cleaned: 1 SW, 2 CF, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_and_2_CF_1405_auto_input,
        ),
        'feature_extractor': fe.BaseData1405FeatureExtractor,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'base': True
    },
    'OriginalData Cleaned: 1 SW, 3 CF, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_main_SW_and_3_CF_1405_auto_input,
        ),
        'feature_extractor': fe.BaseData1405FeatureExtractor,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'base': True
    },
    'OriginalData Cleaned: all ladders, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_SW_or_CF_1405,
        ),
        'feature_extractor': fe.BaseData1405FeatureExtractor,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'base': True
    },
    'OriginalData Cleaned: 3 CF, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData(
            remove_overlaps=wt.remove_all_overlaps,
            ladder_filter=wt.filter_3_main_CF_ladders_1405_auto_input,
        ),
        'feature_extractor': fe.BaseData1405FeatureExtractor,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'base': True
    },
}

agg_funcs_0102 = {
    'JOBNUM': 'first',
    'Date': 'first',
    '0103 ID': 'first',
    '0103 non_unique ID': 'first',
    '0102 ID': 'first',
    '0101 Duration': 'sum',
    'Non Duplicate 0101': 'sum',
    'Non Duplicate 0103': 'sum',
    'Non Duplicate 0104': 'sum',
    'Non Duplicate 0105': 'sum',
    'Non Duplicate 0106': 'sum',
    '0102 Pace': 'first',
    '0104 Alarm Time': 'sum',
    '0105 Alarm Time': 'sum',
    '0106 Alarm Time': 'sum',
}

agg_funcs_0103 = {
    'JOBNUM': 'first',
    '0103 ID': 'first',
    'Non Duplicate 0101': 'sum',
    'Non Duplicate 0102': 'sum',
    'Non Duplicate 0104': 'sum',
    'Non Duplicate 0105': 'sum',
    'Non Duplicate 0106': 'sum',
    '0103 Pace': 'first',
    '0104 Alarm Time': 'sum',
    '0105 Alarm Time': 'sum',
    '0106 Alarm Time': 'sum',
    'Sum 0102 Jam >= 20': 'sum',
    'Sum 0102 Jam >= 19': 'sum',
    'Sum 0102 Jam >= 18': 'sum',
    'Sum 0102 Jam >= 17': 'sum',
    'Sum 0102 Jam >= 16': 'sum',
    'Sum 0102 Jam >= 15': 'sum',
    'Sum 0102 Jam >= 14': 'sum',
    'Sum 0102 Jam >= 13': 'sum',
    'Sum 0102 Jam >= 12': 'sum',
    'Sum 0102 Jam >= 11': 'sum',
    'Sum 0102 Jam >= 10': 'sum',
    'Sum 0102 Jam >= 9': 'sum',
    'Sum 0102 Jam >= 8': 'sum',
    'Sum 0102 Jam >= 7': 'sum',
    'Sum 0102 Jam >= 6': 'sum',
    'Sum 0102 Jam >= 5': 'sum',
    'Sum 0102 Jam >= 4': 'sum',
    'Sum 0102 Jam >= 3': 'sum',
    'Sum 0102 Jam >= 2': 'sum',
    'Sum 0103 Jam >= 20': 'sum',
    'Sum 0103 Jam >= 19': 'sum',
    'Sum 0103 Jam >= 18': 'sum',
    'Sum 0103 Jam >= 17': 'sum',
    'Sum 0103 Jam >= 16': 'sum',
    'Sum 0103 Jam >= 15': 'sum',
    'Sum 0103 Jam >= 14': 'sum',
    'Sum 0103 Jam >= 13': 'sum',
    'Sum 0103 Jam >= 12': 'sum',
    'Sum 0103 Jam >= 11': 'sum',
    'Sum 0103 Jam >= 10': 'sum',
    'Sum 0103 Jam >= 9': 'sum',
    'Sum 0103 Jam >= 8': 'sum',
    'Sum 0103 Jam >= 7': 'sum',
    'Sum 0103 Jam >= 6': 'sum',
    'Sum 0103 Jam >= 5': 'sum',
    'Sum 0103 Jam >= 4': 'sum',
    'Sum 0103 Jam >= 3': 'sum',
    'Sum 0103 Jam >= 2': 'sum',
}

