from utils import utils as ut
import pre_processing.config_settings as cs


class BaseDataFactory:
    ladder_codes = {i + 1: code for i, code in enumerate(cs.settings.keys())}

    @classmethod
    def get_ladder_codes(cls):
        for i, code in enumerate(cs.settings.keys()):
            print(f'{i + 1}. {code}')

    @classmethod
    def factory(cls, code, folder, read_writer=ut.CSVReadWriter, fix_duplicates=False):
        key = cls.ladder_codes[code]
        config = cs.settings[key]
        stats = config.pop('stats') if config.get('stats') else False
        base = config.pop('base') if config.get('base') else False
        drop_first_rows = config.pop('drop_first_rows') if config.get('drop_first_rows') else False
        sd_cleaner = config.pop('sd_cleaner')(fix_duplicates=fix_duplicates)
        pre_process = PreProcess(folder=folder, read_writer=read_writer, **config)
        pre_process.base_data.sd_cleaner = sd_cleaner
        pre_process.get_base_data(stats=stats, base=base)
        pre_process.feature_extraction(drop_first_rows=drop_first_rows)
        pre_process.save()
        return pre_process


class PreProcess:
    """
    Generic class for pre-processing data into final format
    """
    def __init__(self, folder, category, machine, base_data, feature_extractor, read_writer):
        self._machine = machine
        columns = self._machine.data_generation_columns
        self.base_data = base_data
        self.base_data.columns = columns

        self._feature_extractor = feature_extractor
        self._read_writer = read_writer(folder=folder, columns=columns, category=category)
        self._work_table = None
        self._sensor_data = None

    def get_base_data(self, stats=False, base=False):
        work_table = self._read_writer.read_raw_work_table()
        sensor_data = self._read_writer.read_raw_sensor_data()
        self._work_table, self._sensor_data = self.base_data.get_base_data(
            work_table, sensor_data, stats=stats, base=base
        )

    def feature_extraction(self, drop_first_rows):
        if hasattr(self._feature_extractor, 'feature_extraction'):
            self._feature_extractor.feature_extraction(
                self._work_table, self._sensor_data, self._machine, drop_first_rows
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


def _make_aggregate(sensor_data, agg_column, funcs, set_to_zero=False):
    agg_data = sensor_data.groupby(agg_column).agg(funcs)
    agg_data = agg_data.reset_index(drop=True)
    return _set_agg_deacs_to_one(agg_data) if set_to_zero else _make_labels(agg_data)


def _set_agg_deacs_to_one(agg_data):
    condition = agg_data.loc[:, 'Non Duplicate 0101'] > 1
    indices = agg_data[condition].index
    agg_data.loc[:, 'Label'] = 0
    agg_data.loc[indices, 'Label'] = 1
    return agg_data


def _make_labels(agg_data):
    agg_data.loc[:, 'Label'] = agg_data.loc[:, 'Non Duplicate 0101']
    return agg_data
