import os

import pandas as pd


def get_dummy_products(data):
    cols = data.loc[:, 'NAME'].str.split(':', expand=True).dropna()
    data.loc[:, 'NAME'] = cols[0].apply(lambda x: x[:2].lstrip().rstrip()) + '/' + cols[5].apply(lambda x: x.lstrip().rstrip())

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
