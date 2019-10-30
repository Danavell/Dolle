import os

import numpy as np
import pandas as pd

from utils.load_data import utilities as ut


def get_data(file_path, target):
    current_directory = os.getcwd()
    data = pd.read_csv(os.path.join(current_directory, file_path), sep=';')
    X = data.drop(target, axis=1)
    y = data.loc[:, target]
    return X, y


def remake_date_times(sensor_data):
    from pandas.api.types import is_string_dtype
    groupby = sensor_data.groupby('JOBNUM')
    dates = {'Date', 'previous_Date', 'next_Date'}
    for date in dates:
        step = 0
        if date == 'previous_Date':
            if date in sensor_data.columns:
                if is_string_dtype(sensor_data[date]):
                    sensor_data.drop(date, axis=1, inplace=True)
                    step = 1
        elif date == 'next_Date':
            if date in sensor_data.columns:
                if is_string_dtype(sensor_data[date]):
                    sensor_data.drop(date, axis=1, inplace=True)
                    step = -1
        else:
            if date not in sensor_data.columns:
                raise Exception('No Date column in the sensor data')
        sensor_data[date] = groupby['Date'].shift(step)
    return sensor_data


def read_sensor_data(path_in, columns=None, dates=None, remake_dates=False, sep=';'):
    if not dates:
        dates = ['Date']
    if columns:
        data = pd.read_csv(path_in, sep=sep,
                           usecols=columns['init_columns'],
                           na_values=0,
                           parse_dates=[dates],
                           infer_datetime_format=True
        )
    else:
        data = pd.read_csv(path_in, sep=sep,
                           na_values=0,
                           parse_dates=dates,
                           infer_datetime_format=True
        )
    if remake_dates:
        data = remake_date_times(data)
    return data


def read_work_table(path_in, columns, sep=';'):
    return pd.read_csv(path_in, sep=sep,
                       usecols=columns['init_work_prepared'],
                       encoding="ISO-8859-1",
                       parse_dates=['StartDateTime', 'StopDateTime'],
                       infer_datetime_format=True
                       )


def read_sample_work_table(work_path, prod_path, machine_id, reg_ex, columns):
    prod_table = pd.read_csv(prod_path, sep=';', encoding="ISO-8859-1", usecols=columns['init_product_table'])
    work_table = pd.read_csv(work_path,
                             sep=';',
                             usecols=columns['init_sample_work_table'],
                             parse_dates={'StartDateTime': ['StartDate', 'StartTime'],
                                          'StopDateTime': ['StopDate', 'StopTime']},
                             infer_datetime_format=True,)
    work_table['SysQtyGood'] = ut.convert_to_float(work_table['SysQtyGood'])
    work_table = work_table.merge(prod_table, left_on='JOBREF', right_on='ProdId')
    work_table['WRKCTRID'] = work_table['WrkCtrId']
    work_table['NAME'] = work_table['Name']
    work_table['QTYGOOD'] = work_table['SysQtyGood']
    work_table.drop('WrkCtrId', axis=1, inplace=True)
    work_table.drop('Name', axis=1, inplace=True)
    return filter_work_table(work_table, machine_id, reg_ex)


def read_sample_data(path_in, columns):
    data = pd.read_csv(path_in,
                       sep=';',
                       usecols=columns['init_raw_sensor_columns'],
                       dtype=columns['raw_sensor_dtypes'])
    condition = data['Time'].str.contains(r'\d{2}:\d{2}:\d{2}', regex=True)
    times = data[condition]
    if times.size > 0:
        times.loc[:, 'Date'] = times['Date'].astype(str) + ' ' + times['Time'].astype(str)
        data.drop(data.loc[condition].index, inplace=True)
        data = pd.concat([data, times], axis=0)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date').reset_index(drop=True).drop('Time', axis=1).fillna(0)
    data.iloc[:, 1:] = data.iloc[:, 1:].astype(np.int8)
    return data


