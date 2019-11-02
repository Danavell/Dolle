import numpy as np
import pandas as pd
from utils.sensor_data import feature_extraction as fsd
from utils.utils import make_column_arange


def make_aggregates(sensor_data, reg_ex, drop_first_rows=True):
    """
    Aggregates raw sensor data into final form
    """
    data = dict()
    sensor_data['0103 ID'] = sensor_data.loc[:, '0103 Group b-filled']
    products = sensor_data.columns[sensor_data.columns.str.match(reg_ex)]

    for product in products:
        product_data = sensor_data.loc[sensor_data.loc[:, product] == 1].copy()
        product_data = _make_aggregate(product_data)
        data[product] = _drop_first_rows(product_data) if drop_first_rows else product_data

    if products.size > 1:
        all_products = _make_aggregate(sensor_data)
        column = f'all {products.size} products'
        data[column] = _drop_first_rows(all_products) if drop_first_rows else all_products
    return data


def _drop_first_rows(data):
    """
    The first row of many JOBNUMs contain strange readings that are unrepresentative of
    the data as a whole, implying that they should be dropped
    """
    data['previous_JOBNUM'] = data.loc[:, 'JOBNUM'].shift(1)
    condition = data.loc[:, 'JOBNUM'] != data.loc[:, 'previous_JOBNUM']
    indices = data[condition].index
    data.drop(indices, axis=0, inplace=True)
    return data


def _make_aggregate(sensor_data, set_to_zero=False):
    agg_dict = {
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
    grouped = sensor_data.groupby(['JOBNUM', '0103 Group b-filled'])
    agg_data = grouped.agg(agg_dict)
    agg_data.reset_index(drop=True, inplace=True)
    if set_to_zero:
        agg_data = _set_agg_deacs_to_one(agg_data)
    else:
        agg_data = _make_labels(agg_data)
    return agg_data


def _set_agg_deacs_to_one(agg_data):
    condition = agg_data.loc[:, 'Non Duplicate 0101'] > 1
    indices = agg_data[condition].index
    agg_data.loc[indices, 'Non Duplicate 0101'] = 1
    return agg_data


def _make_labels(agg_data):
    condition = agg_data.loc[:, 'Non Duplicate 0101'] >= 1
    indices = agg_data[condition].index
    agg_data.loc[:, 'Label'] = 0
    agg_data.loc[indices, 'Label'] = 1
    return agg_data


def _make_010n_jam_groups(sensor_data, n=2):
    condition = (sensor_data[f'Indgang 010{n}'] == 1) \
                & (sensor_data[f'previous_010{n}'] == 1) \
                | (sensor_data[f'Indgang 010{n}'] == 1) \
                & (sensor_data[f'next_010{n}'] == 1)
    multi_time_0102 = sensor_data.loc[condition].copy()
    condition_2 = multi_time_0102[f'Non Duplicate 010{n}'] == 1
    non_dup_ml_0102 = multi_time_0102.loc[condition_2].copy()
    non_dup_ml_0102['Multi Line Group'] = np.arange(1, len(non_dup_ml_0102) + 1)
    multi_time_0102[f'010{n} Jam'] = non_dup_ml_0102['Multi Line Group']
    multi_time_0102[f'010{n} Jam'] = multi_time_0102.groupby('JOBNUM')[f'010{n} Jam'].fillna(method='ffill')
    multi_time_0102['Duplicate Index'] = multi_time_0102.index
    return multi_time_0102[f'010{n} Jam']


def _make_non_duplicate_jams(sensor_data, column, col=2):
    condition = (sensor_data[f'Non Duplicate 010{col}'] == 1) & (sensor_data[column] >= 0)
    jams = sensor_data.loc[condition]
    sensor_data.loc[jams.index, f'Sum {column}'] = 1
    return sensor_data


def _duration_010n_jam(sensor_data, n):
    agg_dict = {
        f'010{n} Jam': 'first',
        'Date': 'first',
        'next_Date': 'last'
    }
    aggregate = sensor_data.groupby(f'010{n} Jam').agg(agg_dict)
    aggregate['Duration'] = aggregate['next_Date'] - aggregate['Date']
    aggregate['Duration'] = aggregate['Duration'].dt.total_seconds()
    return aggregate


def _duration_010n_jam_greater_than_n(n, sensor_data, col):
    jams = _duration_010n_jam(sensor_data, col)
    condition = jams['Duration'] >= n
    greater_than_n = jams.loc[condition]
    return set(greater_than_n[f'010{col} Jam'])


def make_n_length_jam_durations(sensor_data):
    groupby = sensor_data.groupby('JOBNUM')

    if '0102 Group' not in sensor_data.columns and '0103 Group' not in sensor_data.columns:
        sensor_data = fsd.sensor_groupings(sensor_data)

    for i in range(2, 4):
        if f'previous_010{i}' not in sensor_data.columns:
            sensor_data[f'previous_010{i}'] = groupby[f'Indgang 010{i}'].shift(1)

    for i in range(2, 4):
        if f'next_010{i}' not in sensor_data.columns:
            sensor_data[f'next_010{i}'] = groupby[f'Indgang 010{i}'].shift(-1)

    for i in range(2, 4):
        columns = [
            'JOBNUM', f'Indgang 010{i}', f'previous_010{i}', f'next_010{i}',
            f'Non Duplicate 010{i}'
        ]
        sliced = sensor_data.loc[:, columns].copy()
        sensor_data[f'010{i} Jam'] = _make_010n_jam_groups(sliced, n=i)

    sensor_data = _make_new_jam_durations_loop(sensor_data, '0102 Jam', 2)
    sensor_data = _make_new_jam_durations_loop(sensor_data, '0103 Jam', 3)
    return sensor_data


def _make_new_jam_durations_loop(sensor_data, label, col):
    sensor_data = _make_non_duplicate_jams(sensor_data, label, col)
    for i in range(2, 21):
        sensor_data = _make_new_jam_durations(i, sensor_data, col)
    return sensor_data


def _make_new_jam_durations(n, sensor_data, col=2):
    jams = _duration_010n_jam_greater_than_n(n, sensor_data, col)
    condition = sensor_data[f'010{col} Jam'].isin(jams)
    groups = sensor_data.loc[condition]
    column = f'010{col} Jam >= {n}'
    sensor_data[column] = make_column_arange(groups, f'Non Duplicate 010{col}')
    sensor_data = _make_non_duplicate_jams(sensor_data, column)
    return sensor_data
