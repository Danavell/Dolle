import os

import numpy as np
import pandas as pd

from utils.load_data import load_csv
from utils.sensor_data import data_preparation as sd
from utils.work_table import data_preparation as wt


class BaseData:
    def __init__(self, remove_overlaps, ladder_filter):
        self.machine = None
        self.columns = None
        self.sd_cleaner = None

        self._folder = None
        self._remove_overlaps = remove_overlaps
        self._ladder_filter = ladder_filter

    def get_base_data(self, work_table, sensor_data, stats=False, base=False):
        wt_cleaner = wt.WorkTableCleaner(
            work_table, stats=stats, remove_overlaps=self._remove_overlaps, ladder_filter=self._ladder_filter
        )
        self.sd_cleaner.sensor_data = sensor_data
        wt_prep = wt.PrepareWorkTable(self.columns, stats, wt_cleaner)
        return sd.prepare_base_data(wt_prep, self.sd_cleaner, base_data=base)


class CSVReadWriter:
    """
    Handles all csv read write operations
    """
    def __init__(self, folder, columns, category):
        self.folder = folder
        self.stats_folder = None

        self._dir = os.path.join(get_csv_directory(), self.folder)
        self._columns = columns
        self._category = category
        self._cat_directory = os.path.join(self._dir, self._category)
        self._sensor_path = os.path.join(self._cat_directory, 'sensor_data.csv')
        self._work_table_pat = os.path.join(self._cat_directory, 'work_table.csv')

    def read_raw_sensor_data(self):
        return pd.read_csv(
            os.path.join(self._dir, 'sensor_data.csv'), sep=',', parse_dates=['Date'], infer_datetime_format=True
        )

    def read_sensor_data(self):
        return pd.read_csv(self._sensor_path, sep=',')

    def read_raw_work_table(self):
        return load_csv.read_work_table(os.path.join(self._dir, 'work_table.csv'), columns=self._columns)

    def read_work_table(self):
        return pd.read_csv(self._work_table_pat, sep=',')

    def check_stats_exist(self):
        stats_path = os.path.join(self._dir, f'{self.stats_folder}/agg_stats.csv')
        return os.path.exists(stats_path)

    def read_agg_stats(self):
        stats_path = os.path.join(self._dir, f'{self.stats_folder}/agg_stats.csv')
        return pd.read_csv(stats_path, sep=',')

    def save(self, data):
        if not os.path.exists(self._cat_directory):
            os.mkdir(self._cat_directory)
            for key in data.keys():
                if isinstance(data[key], pd.DataFrame):
                    path = os.path.join(self._cat_directory, f"{key.replace('/', '-')}.csv")
                    data[key].to_csv(path, sep=',', index=False)
                if isinstance(data[key], dict):
                    path = os.path.join(self._cat_directory, key)
                    if not os.path.exists(path):
                        os.mkdir(path)
                        for new_key in data[key].keys():
                            new_path = os.path.join(path, f"{new_key.replace('/', '-')}.csv")
                            data[key][new_key].to_csv(new_path, sep=',', index=False)
        else:
            raise Exception('DIRECTORY ALREADY EXISTS')


def make_column_arange_gte(first_slice, target_column, fillna_groupby_col=None, fill='bfill'):
    """
    Makes a new column containing group ids used by subsequent pandas groupbys
    """
    condition = first_slice.loc[:, target_column] != 0
    second_slice = first_slice.loc[condition]
    second_slice.loc[:, 'temp'] = np.arange(1, len(second_slice.index) + 1)
    first_slice.loc[:, 'temp'] = second_slice.loc[:, 'temp'].copy()
    if isinstance(fillna_groupby_col, str) or isinstance(fillna_groupby_col, list):
        groupby = first_slice.groupby(fillna_groupby_col)
        first_slice = groupby.fillna(method=fill)
    elif fill:
        first_slice.loc[:, 'temp'] = first_slice['temp'].fillna(method=fill).copy()
    return first_slice.loc[:, 'temp']


def get_csv_directory():
    path = os.getcwd()
    while True:
        path = os.path.dirname(path)
        last = path.split('/')[-1].lower()
        if last == 'dolle':
            path = os.path.dirname(path)
            path = os.path.join(path, 'dolle_csvs')
            if not os.path.exists(path):
                os.mkdir(path)
            break
    return path


def get_dummy_products(data):
    cols = data.loc[:, 'NAME'].str.split(':', expand=True).dropna()
    data.loc[:, 'NAME'] = cols[0].apply(
        lambda x: x[:2].lstrip().rstrip()) + '/' + cols[5].apply(
        lambda x: x.lstrip().rstrip()
    )
    return cols, pd.concat([data, pd.get_dummies(data.loc[:, 'NAME'])], axis=1)


def get_base_dolle_directory():
    path = os.getcwd()
    while path.split('/')[-1].lower() != 'dolle':
        path = os.path.dirname(path)
        last = path.split('/')[-1].lower()
        if last == 'dolle':
            break
        if last == '/':
            raise Exception('No Dolle Directory on the path')
    return path


class Machine1405:
    def __init__(self):
        self.machine_id = 1405
        self.product_reg_ex = '^CF|^SW'
        self.previous_1_to_6 = [f'previous_010{i}' for i in range(1, 7)]
        self.next_4_to_6 = [f'next_010{i}' for i in range(4, 7)]
        self.current_1_to_6 = [f'Indgang 010{X}' for X in range(1, 7)]
        self.non_duplicates_1_to_6 = [f'Non Duplicate 010{X}' for X in range(1, 7)]
        self.data_generation_columns = {
            'init_raw_sensor_columns': merge(['Date', 'Time'], ['Indgang 010{}'.format(i) for i in range(1, 7)]),
            'init_sample_work_table': [
                'JOBREF', 'StartDate', 'StartTime', 'Seconds', 'StopDate', 'StopTime', 'SysQtyGood', 'WrkCtrId'
            ],
            'init_work_prepared': ['NAME', 'WRKCTRID', 'JOBREF', 'QTYGOOD', 'StartDateTime', 'StopDateTime', 'Seconds'],
            'init_product_table': ['Name', 'ProdId'],
            'previous_shifted': merge(['previous_Date'], self.previous_1_to_6),
            'previous_1_to_6': self.previous_1_to_6,
            'gen_shifted_columns_1': merge(['Date'], self.current_1_to_6),
            'next_shifted': merge(['next_Date', 'next_0101'], self.next_4_to_6),
            'remove_nans_and_floats': merge(merge(self.previous_1_to_6, merge(['next_0101'], self.next_4_to_6)),
                                            self.non_duplicates_1_to_6),
            'init_columns': merge(['Non Duplicate 0103'], merge(self.previous_1_to_6,
                            merge(['f_0101'], self.next_4_to_6))),
            '_gen_shifted_columns_-1': merge(['Date', 'Indgang 0101'], [f'Indgang 010{i}' for i in range(4, 7)]),
            'convert_to_seconds': merge(['0101 Down Time'], [f'010{i} Alarm Time' for i in range(4, 7)]),
            'columns_to_keep': [
                        'Date', 'JOBREF', 'JOBNUM',
                        'Indgang 0101', 'Non Duplicate 0101', '0101 Down Time',
                        'Indgang 0102', 'Non Duplicate 0102', '0102 Pace',
                        'Indgang 0103', 'Non Duplicate 0103', '0103 Pace',
                        'Indgang 0104', 'Non Duplicate 0104', '0104 Alarm Time',
                        'Indgang 0105', '0105 Alarm Time',
                        'Indgang 0106', '0106 Alarm Time',
                    ]
        }

        self.generate_statistics = {
            'data_agg_dict': {
                'Date': ['first', 'last'],
                'JOBREF': 'first',
                'Non Duplicate 0101': 'sum',
                '0101 Duration': 'sum',
                'Non Duplicate 0102': 'sum',
                '0102 Pace': ['mean', 'median', 'std'],
                'Non Duplicate 0103': 'sum',
                'Non Duplicate 0104': 'sum',
                'Non Duplicate 0105': 'sum',
                'Non Duplicate 0106': 'sum',
                '0103 Pace': ['mean', 'median', 'std'],
                '0104 Alarm Time': 'sum',
                '0105 Alarm Time': 'sum',
                '0106 Alarm Time': 'sum',
            },

            'work_table_agg_dict': {
                'QTYGOOD': 'sum',
                'Seconds': 'sum',
                'NAME': 'first',
            },
            'ordered_stats': [
                ('JOBREF', 'first'),
                ('Date', 'first'),
                ('Date', 'last'),
                'Seconds',
                'No. Deactivations/hour',
                ('0101 Duration', 'sum'),
                '% Down Time',
                'No. 0104/hour',
                '% 0104',
                ('0104 Alarm Time', 'sum'),
                'No. 0105/hour',
                '% 0105',
                'No. 0106/hour',
                '% 0106',
                ('Non Duplicate 0102', 'sum'),
                ('Non Duplicate 0103', 'sum'),
                'QTYGOOD',
                '0103 Count Vs Expected',
                'Strings per Ladder',
                ('0105 Alarm Time', 'sum'),
                ('0106 Alarm Time', 'sum'),
                ('0102 Pace', 'mean'),
                ('0102 Pace', 'median'),
                ('0102 Pace', 'std'),
                ('0103 Pace', 'mean'),
                ('0103 Pace', 'median'),
                ('0103 Pace', 'std'),
                'NAME',
            ],
            'stats_final columns': [
                'JOBREF',
                'Start Time',
                'Stop Time',
                'Job Length(s)',
                'No. Deactivations/hour',
                'Down Time(s)',
                '% Down Time',
                'No. 0104/hour',
                '% 0104',
                '0104 Alarm Sum(s)',
                'No. 0105/hour',
                '% 0105',
                'No. 0106/hour',
                '% 0106',
                '0102 Sum',
                '0103 Sum',
                'Expected 0103',
                '0103 Count Vs Expected',
                'Strings per Ladder',
                '0105 Alarm Sum(s)',
                '0106 Alarm Sum(s)',
                '0102 Pace avg(s)',
                '0102 Pace median(s)',
                '0102 std',
                '0103 Pace avg(s)',
                '0103 Pace median(s)',
                '0103 std',
                'Product',
            ],
        }


def merge(first, second):
    return [item for sublist in [first, second] for item in sublist]


