from pre_processing import feature_extraction as fe
from utils import utils as ut
from utils.sensor_data import data_preparation as sd
from utils.work_table import data_preparation as wt

settings = {
    'Stats 1405: all ladders, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData,
        'remove_overlaps': wt.remove_all_overlaps,
        'ladder_filter': wt.filter_SW_or_CF_1405,
        'feature_extractor': fe.StatsFeatureExtractor,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'meta': {
            'stats': True
        }
    },
    'Stats 1405: 0102, 1 SW, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData,
        'remove_overlaps': wt.remove_all_overlaps,
        'ladder_filter': wt.filter_main_SW_1405_auto_input,
        'feature_extractor': fe.StatsFeatureExtractor0102Agg,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'meta': {}
    },
    'Stats 1405: 0102, 1 SW, no overlaps, drop first rows': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData,
        'remove_overlaps': wt.remove_all_overlaps,
        'ladder_filter': wt.filter_main_SW_1405_auto_input,
        'feature_extractor': fe.StatsFeatureExtractor0102Agg,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'meta': {
            'drop_first': True
        }
    },
    'Stats 1405: 0102, 1 SW, 2 CF, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData,
        'remove_overlaps': wt.remove_all_overlaps,
        'ladder_filter': wt.filter_main_SW_and_2_CF_1405_auto_input,
        'feature_extractor': fe.StatsFeatureExtractor0102Agg,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'meta': {}
    },
    'Stats 1405: 0102, 1 SW, 2 CF, no overlaps, drop first rows': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData,
        'remove_overlaps': wt.remove_all_overlaps,
        'ladder_filter': wt.filter_main_SW_and_2_CF_1405_auto_input,
        'feature_extractor': fe.StatsFeatureExtractor0102Agg,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'meta': {
            'drop_first': True
        }
    },
    'Stats 1405: 0102, 1 SW, 3 CF, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData,
        'remove_overlaps': wt.remove_all_overlaps,
        'ladder_filter': wt.filter_main_SW_and_3_CF_1405_auto_input,
        'feature_extractor': fe.StatsFeatureExtractor0102Agg,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'meta': {}
    },
    'Stats 1405: 0102, 1 SW, 3 CF, no overlaps, drop first rows': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData,
        'remove_overlaps': wt.remove_all_overlaps,
        'ladder_filter': wt.filter_main_SW_and_3_CF_1405_auto_input,
        'feature_extractor': fe.StatsFeatureExtractor0102Agg,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'meta': {
            'drop_first': True
        }
    },
    'Stats 1405: 0102, 3 CF, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData,
        'remove_overlaps': wt.remove_all_overlaps,
        'ladder_filter': wt.filter_3_main_CF_ladders_1405_auto_input,
        'feature_extractor': fe.StatsFeatureExtractor0102Agg,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'meta': {}
    },
    'Stats 1405: 0102, 3 CF, no overlaps, drop first rows': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData,
        'remove_overlaps': wt.remove_all_overlaps,
        'ladder_filter': wt.filter_3_main_CF_ladders_1405_auto_input,
        'feature_extractor': fe.StatsFeatureExtractor0102Agg,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'meta': {
            'drop_first': True
        }
    },
    'MLAgg0103 1405: 1 SW, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData,
        'remove_overlaps': wt.remove_all_overlaps,
        'ladder_filter': wt.filter_main_SW_1405_auto_input,
        'feature_extractor': fe.MLFeatureExtractor0103,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'meta': {
            'jams': 20,
        }
    },
    'MLAgg0103 1405: 1 SW, no overlaps, drop first rows': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData,
        'remove_overlaps': wt.remove_all_overlaps,
        'ladder_filter': wt.filter_main_SW_1405_auto_input,
        'feature_extractor': fe.MLFeatureExtractor0103,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'meta': {
            'jams': 20,
            'drop_first': True
        }
    },
    'MLAgg0103 1405: 1 SW, 2 CF, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData,
        'remove_overlaps': wt.remove_all_overlaps,
        'ladder_filter': wt.filter_main_SW_and_2_CF_1405_auto_input,
        'feature_extractor': fe.MLFeatureExtractor0103,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'meta': {
            'jams': 20,
        }
    },
    'MLAgg0103 1405: 1 SW, 2 CF, no overlaps, drop first rows': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData,
        'remove_overlaps': wt.remove_all_overlaps,
        'ladder_filter': wt.filter_main_SW_and_2_CF_1405_auto_input,
        'feature_extractor': fe.MLFeatureExtractor0103,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'meta': {
            'jams': 20,
            'drop_first': True
        }
    },
    'MLAgg0103 1405: 1 SW, 3 CF, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData,
        'remove_overlaps': wt.remove_all_overlaps,
        'ladder_filter': wt.filter_main_SW_and_3_CF_1405_auto_input,
        'feature_extractor': fe.MLFeatureExtractor0103,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'meta': {
            'jams': 20,
        }
    },
    'MLAgg0103 1405: 1 SW, 3 CF, no overlaps, drop first rows': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData,
        'remove_overlaps': wt.remove_all_overlaps,
        'ladder_filter': wt.filter_main_SW_and_3_CF_1405_auto_input,
        'feature_extractor': fe.MLFeatureExtractor0103,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'meta': {
            'jams': 20,
            'drop_first': True
        }
    },
    'MLAgg0103 1405: 3 CF, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData,
        'remove_overlaps': wt.remove_all_overlaps,
        'ladder_filter': wt.filter_3_main_CF_ladders_1405_auto_input,
        'feature_extractor': fe.MLFeatureExtractor0103,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'meta': {
            'jams': 20
        }
    },
    'MLAgg0103 1405: 3 CF, no overlaps, drop first rows': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData,
        'remove_overlaps': wt.remove_all_overlaps,
        'ladder_filter': wt.filter_3_main_CF_ladders_1405_auto_input,
        'feature_extractor': fe.MLFeatureExtractor0103,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'meta': {
            'jams': 20,
            'drop_first': True
        }
    },
    'BaseData 1405: 1 CF, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData,
        'remove_overlaps': wt.remove_all_overlaps,
        'ladder_filter': wt.filter_main_SW_1405_auto_input,
        'feature_extractor': fe.BaseData1405FeatureExtractor,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'meta': {}
    },
    'BaseData 1405: 1 SW, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData,
        'remove_overlaps': wt.remove_all_overlaps,
        'ladder_filter': wt.filter_main_SW_ladder_1405_auto_input,
        'feature_extractor': fe.BaseData1405FeatureExtractor,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'meta': {}
    },
    'BaseData 1405: 1 SW, 2 CF, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData,
        'remove_overlaps': wt.remove_all_overlaps,
        'ladder_filter': wt.filter_main_SW_and_2_CF_1405_auto_input,
        'feature_extractor': fe.BaseData1405FeatureExtractor,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'meta': {}
    },
    'BaseData 1405: 1 SW, 3 CF, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData,
        'remove_overlaps': wt.remove_all_overlaps,
        'ladder_filter': wt.filter_main_SW_and_3_CF_1405_auto_input,
        'feature_extractor': fe.BaseData1405FeatureExtractor,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'meta': {}
    },
    'BaseData 1405: all ladders, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData,
        'remove_overlaps': wt.remove_all_overlaps,
        'ladder_filter': wt.filter_SW_or_CF_1405,
        'feature_extractor': fe.BaseData1405FeatureExtractor,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'meta': {}
    },
    'BaseData 1405: 3 CF, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData,
        'remove_overlaps': wt.remove_all_overlaps,
        'ladder_filter': wt.filter_3_main_CF_ladders_1405_auto_input,
        'feature_extractor': fe.BaseData1405FeatureExtractor,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'meta': {}
    },
    'OriginalData Cleaned 1405: 1 CF, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData,
        'remove_overlaps': wt.remove_all_overlaps,
        'ladder_filter': wt.filter_main_SW_1405_auto_input,
        'feature_extractor': fe.BaseData1405FeatureExtractor,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'meta': {
            'base': True
        }
    },
    'OriginalData Cleaned: 1 SW, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData,
        'remove_overlaps': wt.remove_all_overlaps,
        'ladder_filter': wt.filter_main_SW_ladder_1405_auto_input,
        'feature_extractor': fe.BaseData1405FeatureExtractor,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'meta': {
            'base': True
        }
    },
    'OriginalData Cleaned: 1 SW, 2 CF, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData,
        'remove_overlaps': wt.remove_all_overlaps,
        'ladder_filter': wt.filter_main_SW_and_2_CF_1405_auto_input,
        'feature_extractor': fe.BaseData1405FeatureExtractor,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'meta': {
            'base': True
        }
    },
    'OriginalData Cleaned: 1 SW, 3 CF, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData,
        'remove_overlaps': wt.remove_all_overlaps,
        'ladder_filter': wt.filter_main_SW_and_3_CF_1405_auto_input,
        'feature_extractor': fe.BaseData1405FeatureExtractor,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'meta': {
            'base': True
        }
    },
    'OriginalData Cleaned: all ladders, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData,
        'remove_overlaps': wt.remove_all_overlaps,
        'ladder_filter': wt.filter_SW_or_CF_1405,
        'feature_extractor': fe.BaseData1405FeatureExtractor,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'meta': {
            'base': True
        }
    },
    'OriginalData Cleaned: 3 CF, no overlaps': {
        'machine': ut.Machine1405,
        'base_data': ut.BaseData,
        'remove_overlaps': wt.remove_all_overlaps,
        'ladder_filter': wt.filter_3_main_CF_ladders_1405_auto_input,
        'feature_extractor': fe.BaseData1405FeatureExtractor,
        'sd_cleaner': sd.SensorDataCleaner1405,
        'meta': {
            'base': True
        }
    },
}

product_col_reg_ex = r'^[A-Z]{2}[-][1-9][A-Z][-][1-9][A-Z][-][1-9][A-Z][-][1-9]{2}[A-Z]$'

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

base_agg_funcs_0103 = {
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
}

