import numpy as np
import pandas as pd

from utils import utils as ut
import pre_processing.config_settings as cs


class BaseDataFactory:
    ladder_codes = {i + 1: code for i, code in enumerate(cs.settings.keys())}

    @classmethod
    def get_ladder_codes(cls):
        for i, code in enumerate(cs.settings.keys()):
            print(f'{i + 1}. {code}')

    @classmethod
    def factory(cls, code, folder, read_writer=ut.CSVReadWriter, fix_duplicates=False, save=True):
        key = cls.ladder_codes[code]
        config = cs.settings[key]
        config['category'] = key
        config['base_data'] = config.get('base_data')(
            config.pop('remove_overlaps'), config.pop('ladder_filter')
        )
        meta = config.pop('meta')
        config['stats_folder'] = meta.pop('stats_folder') if meta.get('stats_folder') else False
        stats = meta.pop('stats') if meta.get('stats') else False
        base = meta.pop('base') if meta.get('base') else False
        sd_cleaner = config.pop('sd_cleaner')(fix_duplicates=fix_duplicates)
        pre_process = PreProcess(folder=folder, read_writer=read_writer, **config)
        pre_process.base_data.sd_cleaner = sd_cleaner
        pre_process.get_base_data(stats=stats, base=base)
        pre_process.feature_extraction(meta=meta)
        if save:
            pre_process.save()
        return pre_process.get_data()


class PreProcess:
    """
    Generic class for pre-processing data into final format
    """
    def __init__(self, folder, category, machine, base_data, feature_extractor, read_writer, stats_folder):
        self._machine = machine()
        columns = self._machine.data_generation_columns
        self.base_data = base_data
        self.base_data.columns = columns

        self._feature_extractor = feature_extractor()
        self._read_writer = read_writer(folder=folder, columns=columns, category=category)
        self._stats_folder = stats_folder
        self._work_table = None
        self._sensor_data = None

    def get_base_data(self, stats=False, base=False):
        work_table = self._read_writer.read_raw_work_table()
        sensor_data = self._read_writer.read_raw_sensor_data()
        self._work_table, self._sensor_data = self.base_data.get_base_data(
            work_table, sensor_data, stats=stats, base=base
        )

    def feature_extraction(self, meta):
        if hasattr(self._feature_extractor, 'feature_extraction'):
            if self._stats_folder:
                self._read_writer.stats_folder = self._stats_folder
                if not self._read_writer.check_stats_exist():
                    print('CREATING STATS FOLDER')
                    BaseDataFactory.factory(
                        code=list(cs.settings.keys()).index(self._stats_folder) + 1,
                        folder=self._read_writer.folder,
                        read_writer=type(self._read_writer),
                        fix_duplicates=self.base_data.sd_cleaner.fix_duplicates
                    )
                    print('STATS FOLDER CREATED')
                else:
                    print('STATS FOLDER ALREADY EXISTS')
                self._feature_extractor.stats = self._read_writer.read_agg_stats()

            self._feature_extractor.feature_extraction(
                self._work_table, self._sensor_data, self._machine, meta
            )

    def save(self):
        if hasattr(self._feature_extractor, 'data'):
            self._read_writer.save(self._feature_extractor.data)

    def get_data(self):
        if hasattr(self._feature_extractor, 'data'):
            return self._feature_extractor.data


def make_aggregates(sensor_data, reg_ex, agg_column, funcs, drop_first_row_func=None):
    """
    Aggregates raw sensor data into final form
    """
    data = dict()
    products = sensor_data.columns[sensor_data.columns.str.match(reg_ex)]

    for product in products:
        product_data = sensor_data.loc[sensor_data.loc[:, product] == 1].copy()
        product_data = _make_aggregate(product_data, agg_column, funcs)
        data[product] = drop_first_row_func(product_data) if drop_first_row_func else product_data

    if products.size > 1:
        all_products = _make_aggregate(sensor_data, agg_column, funcs)
        column = f'all {products.size} products'
        data[column] = drop_first_row_func(all_products) if drop_first_row_func else all_products
    return data


def _make_aggregate(sensor_data, agg_column, funcs, set_to_zero=True):
    agg_data = sensor_data.groupby(agg_column).agg(funcs)
    agg_data = agg_data.reset_index(drop=True)
    return _set_agg_deacs_to_one(agg_data) if set_to_zero else _make_labels(agg_data)


def _set_agg_deacs_to_one(agg_data):
    agg_data.loc[agg_data.loc[:, 'Non Duplicate 0101'] >= 1, 'Label'] = 1
    agg_data.loc[agg_data.loc[:, 'Non Duplicate 0101'] == 0, 'Label'] = 0
    return agg_data


def _make_labels(agg_data):
    agg_data.loc[:, 'Label'] = agg_data.loc[:, 'Non Duplicate 0101']
    return agg_data


def _calc_percentile(data):
    return pd.DataFrame(
        np.percentile(data, [i for i in range(1, 100)]),
        [f'{i}%' for i in range(1, 100)]
    )


def calc_percentiles(frames):
    """
    Calculates the percentiles from 0 to 99 for dataframe passed in the frames dict
    """
    output = pd.concat([_calc_percentile(frames[key]) for key in frames.keys()], axis=1)
    output.columns = [key for key in frames.keys()]
    return output
