import os

import pandas as pd


class PreProcess:
    """
    Generic class for pre-processing data into final format
    """
    def __init__(self, base_data, feature_extractor, persister):
        self._base_data = base_data
        self._machine = self._base_data.machine
        self._feature_extractor = feature_extractor
        self._persister = persister

    def get_base_data(self, stats=False):
        self._base_data.get_base_data(stats=stats)

    def feature_extraction(self):
        self._feature_extractor(
            self._base_data.work_table,
            self._base_data.sensor_data,
            self._machine
        )

    def save(self):
        self._persister(**self._feature_extractor.data)


def CSVSaver(category, sensor_data, work_table, additional_files):
    """
    If one doesn't already exists, creates directory whose name is start to end
    date of sensor data. Inside, creates another directory, which is the data category
    where csv files are saved
    """
    first_index = sensor_data.index[0]
    first_date = str(sensor_data.loc[first_index, 'Date'])
    last_index = len(sensor_data.index) - 1
    last_date = str(sensor_data.loc[last_index, 'Date'])

    path = get_csv_directory()
    folder = f'{first_date.split()[0]}  ->  {last_date.split()[0]}'
    directory = os.path.join(path, folder)
    if not os.path.exists(directory):
        os.mkdir(directory)

    cat_directory = os.path.join(directory, category)
    if not os.path.exists(cat_directory):
        os.mkdir(cat_directory)
        sensor_path = os.path.join(cat_directory, 'sensor_data.csv')
        sensor_data.to_csv(sensor_path, sep=';')
        work_path = os.path.join(cat_directory, 'work_table.csv')
        work_table.to_csv(work_path, sep=';')
        if additional_files:
            for key in additional_files.keys():
                if isinstance(additional_files[key], pd.DataFrame):
                    arg_path = os.path.join(cat_directory, f'{key}.csv')
                    additional_files[key].to_csv(arg_path, sep=';', index=False)
        else:
            pass
    else:
        print('DIRECTORY ALREADY EXISTS')


def get_csv_directory():
    path = os.getcwd()
    if path.split('/')[-1].lower() != 'dolle':
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
    return pd.concat([data, pd.get_dummies(data.loc[:, 'NAME'])], axis=1)


def get_base_dolle_directory():
    path = os.getcwd()
    if path.split('/')[-1].lower() != 'dolle':
        while True:
            path = os.path.dirname(path)
            last = path.split('/')[-1].lower()
            if last == 'dolle':
                break
            if last == '/':
                raise Exception('No Dolle Directory on the path')
    return path


