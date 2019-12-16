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


def calc_time_delta_between_deac_and_010n(data, agg, x):
    no_deacs = agg.loc[agg['Non Duplicate 0101'] == 0, :]
    deacs_sd = data.loc[
        (data['Non Duplicate 0101'] == 1) & (data[f'010{x} ID'].isin(agg[f'010{x} ID'])),
        ['Date', f'010{x} ID']
    ]
    """
    Separate 010X IDs with 1 and more than 1 deactivation. Slightly different functions
    will have to be applied to both. Therefore, it makes sense to separate them as 
    early as possible
    """
    single_deacs = _calc_t_delta_and_merge(deacs_sd, agg, agg['Non Duplicate 0101'] == 1, x=x)
    multi_deacs = _calc_t_delta_and_merge(
        deacs_sd, agg, agg['Non Duplicate 0101'] > 1, multi=True, x=x
    )

    agg = pd.concat([
        single_deacs,
        no_deacs,
        multi_deacs.loc[multi_deacs.groupby(f'010{x} ID').cumcount() + 1 == 1, :]
    ], axis=0, sort=False)
    return agg.sort_values([f'010{x} ID', 'Date']).reset_index(drop=True)


def _calc_t_delta_and_merge(deacs_sd, agg, condition, multi=False, x=2):
    """
    Calculates the time between when a string enters a machine then concats
    the time vector for all deacs with the agg data
    """
    data = agg.loc[condition, :]
    time_delta_aggs = _calc_time_since_string_in_and_deactivation(
        deacs_sd, data, x
    )
    """
    If the agg data contains rows with only 1 deac per 010n ID then it doesn't
    matter which dataframe is on the 'left'. This is not true when the 010X ID contains
    multiple deactivations. In that case the time deltas, which contain all the 
    deactivations in an 010X ID, must be on the left
    """
    left = time_delta_aggs if multi else data
    right = data if multi else time_delta_aggs
    return pd.merge(left=left, right=right, left_on=f'010{x} ID', right_on=f'010{x} ID')


def _calc_time_since_string_in_and_deactivation(sd_deacs, agg_deacs, x):
    """
    Returns a dataframe containing 0102 ID, the time of each string in and deactivation
    as well as the time delta between them
    """
    agg_deacs = agg_deacs.loc[:, ['Date', f'010{x} ID', 'Non Duplicate 0101']].copy()
    merged = pd.merge(
        left=agg_deacs, right=sd_deacs, how='left', left_on=f'010{x} ID', right_on=f'010{x} ID'
    )
    merged.loc[:, 'Time Delta'] = (merged.loc[:, 'Date_y'] - merged.loc[:, 'Date_x']) \
        .dt.total_seconds() \
        .astype(int)
    return merged.loc[:, [f'010{x} ID', 'Date_y', 'Time Delta']]


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


def give_unique_010n_ids_end_jobnum(sensor_data, n=3):
    """
    At the end of jobnums, no ladders are produced and 0103 IDs are nan.
    Conversely, at the end of jobnums, no strings enter the machine and
    0102 IDs are also nan.

    It is important that they have unique JOBNUMS since aggregate stats
    treat all of the nan 0103 IDs in all of the JOBNUMs as one group.
    Though this function gives each block of nans per JOBNUM a unique ID,
    they are distinguished from real 0103 IDs by the fact that they go from
    -1, -2 ...
    """
    condition = (sensor_data[f'010{n} ID'] == -1) & (sensor_data[f'prev_010{n} ID'] >= 1)
    sensor_data[f'010{n} ID'] = sensor_data[f'010{n} ID'].replace(-1, np.nan)
    end_of_jobnum = sensor_data.loc[condition].copy()
    end_of_jobnum.loc[:, 'temp'] = np.arange(1, len(end_of_jobnum.index) + 1) * -1
    sensor_data.loc[end_of_jobnum.index, f'010{n} ID'] = end_of_jobnum['temp']
    sensor_data[f'010{n} ID'] = sensor_data.groupby('JOBNUM')[f'010{n} ID']\
        .fillna(method='ffill')
    return sensor_data


def pie_chart(data, labels, title):
    import matplotlib.pyplot as plt

    plt.pie(data, startangle=90)
    plt.title(title)
    plt.axis('equal')
    plt.legend(
        loc='right',
        labels=[
            '%s, %1.1f %%' %
            (int(l), d * 100) for l, d in zip(labels, data)
        ]
    )
    plt.show()
