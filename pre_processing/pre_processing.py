from aggregations.aggregate_0103.sensor_data import feature_extraction as fe
from utils.sensor_data import data_preparation as sd
from utils.STATS import STATS as st
from utils.utils import BaseData, CSVReadWriter, Machine1405, PreProcess
from utils.work_table import data_preparation as wt


class BaseDataFactory:
    settings = {
        'Stats: all ladders, no overlaps': {
            'category': 'Stats 1405 All ladders',
            'machine': Machine1405(),
            'base_data': BaseData(
                remove_overlaps=wt.remove_all_overlaps,
                ladder_filter=sd.filter_SW_or_CF_1405,
            ),
            'feature_extractor': st.StatsFeatureExtractor(),
            'sd_cleaner': sd.SensorDataCleaner1405,
            'stats': True
        },
        'MLAgg0103_1405: main auto SW, no overlaps': {
            'category': 'MLAgg0103_1405: main auto SW, no overlaps',
            'machine': Machine1405(),
            'base_data': BaseData(
                remove_overlaps=wt.remove_all_overlaps,
                ladder_filter=sd.filter_main_SW_1405_auto_input,
            ),
            'feature_extractor': fe.MLFeatureExtractor0103(),
            'sd_cleaner': sd.SensorDataCleaner1405,
        },
        'MLAgg0103_1405: main auto SW and 2 main auto CF, no overlaps': {
            'category': 'MLAgg0103_1405: main auto SW and 2 main auto CF, no overlaps',
            'machine': Machine1405(),
            'base_data': BaseData(
                remove_overlaps=wt.remove_all_overlaps,
                ladder_filter=sd.filter_main_SW_and_2_CF_1405_auto_input,
            ),
            'feature_extractor': fe.MLFeatureExtractor0103(),
            'sd_cleaner': sd.SensorDataCleaner1405,
        },
        'MLAgg0103_1405: main auto SW and 3 main auto CF, no overlaps': {
            'category': 'MLAgg0103_1405: main auto SW and 3 main auto CF, no overlaps',
            'machine': Machine1405(),
            'base_data': BaseData(
                remove_overlaps=wt.remove_all_overlaps,
                ladder_filter=sd.filter_main_SW_and_3_CF_1405_auto_input,
            ),
            'feature_extractor': fe.MLFeatureExtractor0103(),
            'sd_cleaner': sd.SensorDataCleaner1405,
        },
        'MLAgg0103_1405: 3 main auto CF, no overlaps': {
            'category': 'MLAgg0103_1405: 3 main auto CF, no overlaps',
            'machine': Machine1405(),
            'base_data': BaseData(
                remove_overlaps=wt.remove_all_overlaps,
                ladder_filter=sd.filter_3_main_CF_ladders_1405_auto_input,
            ),
            'feature_extractor': fe.MLFeatureExtractor0103(),
            'sd_cleaner': sd.SensorDataCleaner1405,
        },
        'MLAgg0103_1405: all ladders, no overlaps': {
            'category': 'MLAgg0103_1405: all ladders, no overlaps',
            'machine': Machine1405(),
            'base_data': BaseData(
                remove_overlaps=wt.remove_all_overlaps,
                ladder_filter=sd.filter_SW_or_CF_1405,
            ),
            'feature_extractor': fe.MLFeatureExtractor0103(),
            'sd_cleaner': sd.SensorDataCleaner1405,
        },
        'BaseData 1405: main auto SW, no overlaps': {},
        'BaseData 1405: main auto SW and 2 main auto CF, no overlaps': {},
        'BaseData 1405: main auto SW and 3 main auto CF, no overlaps': {},
        'BaseData 1405: 3 main auto CF, no overlaps': {},
        'BaseData 1405: all ladders, no overlaps': {},
    }

    ladder_codes = dict(zip((i + 1 for i in range(len(settings.keys()))), settings.keys()))

    @classmethod
    def get_ladder_codes(cls):
        for i, code in enumerate(cls.settings.keys()):
            print(f'{i + 1}. {code}')

    @classmethod
    def factory(cls, code, folder, read_writer=CSVReadWriter, fix_duplicates=False):
        key = cls.ladder_codes[code]
        config = cls.settings[key]
        stats = config.pop('stats') if config.get('stats') else False
        sd_cleaner = config.pop('sd_cleaner')(fix_duplicates=fix_duplicates)
        pre_process = PreProcess(folder=folder, read_writer=read_writer, **config)
        pre_process.base_data.sd_cleaner = sd_cleaner
        pre_process.get_base_data(stats=stats)
        pre_process.feature_extraction()
        pre_process.save()
        return pre_process


