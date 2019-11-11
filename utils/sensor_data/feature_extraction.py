import pandas as pd
import numpy as np

from utils.sensor_data import data_preparation as sd


def pace_diff_column_sqrt_cube(row, column, agg_stats, agg_sensor, new_column, old_column, sqrt=True):
    temp = agg_stats.set_index('Product')
    avg_pace = temp.loc[row, column].copy()
    condition = agg_sensor.loc[:, old_column] < avg_pace
    less_than_avg_pace = agg_sensor.loc[condition, :]
    new_col = less_than_avg_pace.loc[:, old_column] - avg_pace
    if sqrt:
        less_than_avg_pace.loc[:, new_column] = np.sqrt(abs(new_col ** 2))
    else:
        less_than_avg_pace.loc[:, new_column] = abs(new_col)
    condition = agg_sensor.index.isin(less_than_avg_pace.index)
    agg_sensor = agg_sensor.loc[~condition, :]
    agg_sensor.loc[:, new_column] = agg_sensor.loc[:, old_column] - avg_pace
    agg_sensor = pd.concat([agg_sensor, less_than_avg_pace], axis=0)
    agg_sensor.sort_index(inplace=True)
    return agg_sensor


def calculate_pace(sensor_data, columns):
    sensor_data[columns['next_shifted']] = sensor_data.groupby('JOBNUM')[columns['_gen_shifted_columns_-1']].shift(-1)

    """
    Remove nans and convert floats to integers
    """
    sensor_data[columns['remove_nans_and_floats']] = sensor_data[columns['remove_nans_and_floats']]\
        .fillna(0).astype(np.int8)

    """
    Adds deactivation times to sensor data
    """
    sensor_data = _deactivations(sensor_data)

    """
    generates the time difference between subsequent 0102 and 0103 sensor
    readings. This is the pace in and pace out.
    """
    for i in range(2, 4):
        sensor_data[f'010{i} Pace'] = _gen_pace_data(sensor_data, f'Non Duplicate 010{i}')

    """
    calculates the duration of each row of alarm data
    """
    for i in range(4, 7):
        sensor_data[f'010{i} Alarm Time'] = calc_error_time(sensor_data, f'Indgang 010{i}')
    return sensor_data


def _gen_pace_data(sensor_data, column):
    filtered = sensor_data[sensor_data[column] != 0].copy()
    date_shifted = filtered.groupby(['JOBNUM'])['Date'].shift(1)
    pace = filtered['Date'] - date_shifted
    return _to_seconds(pace)


def calc_error_time(sensor_data, column, groupby_cols='JOBNUM'):
    sensor_data['previous_Date'] = sensor_data.groupby(groupby_cols)['Date'].shift(1)
    sensor_data = sensor_data.loc[sensor_data[column] != 0, ['previous_Date', 'Date']].copy()
    return _to_seconds(sensor_data['Date'] - sensor_data['previous_Date'])


def _to_seconds(times):
    return times.dt.total_seconds().fillna(0).astype(int)


def _deactivations(data):
    """
    calculates the duration of single and multi row deactivations
    """
    data = sd.calc_single_row_0101_duration(data)
    data = sd.calc_multi_row_0101_duration(data)
    return data


def get_dummy_products(data):
    agg = data.copy(deep=True)
    if 'PRODUCT' in agg.columns:
        product = 'PRODUCT'
    elif 'Product' in agg.columns:
        product = 'PRODUCT'
    elif 'NAME' in agg.columns:
        product = 'NAME'
    elif 'Name' in agg.columns:
        product = 'Name'
    else:
        for column in data.columns:
            print(column)
        raise ValueError('invalid product name')

    cols = agg[product].str.split(':', expand=True)
    agg.drop(product, axis=1, inplace=True)
    agg[product] = cols[0].apply(lambda x: x[:2].lstrip().rstrip()) + '/' + cols[5].apply(lambda x: x.lstrip().rstrip())
    return pd.get_dummies(agg[product]), pd.concat([agg, pd.get_dummies(agg[product])], axis=1)


def sensor_groupings(data):
    for i in range(2, 4):
        final = f'010{i} bfill'
        ndp = f'Non Duplicate 010{i}'
        data.loc[:, final] = _sensor_groupings(data[[ndp, 'JOBNUM']].loc[data[ndp] == 1, :])
        data.loc[:, final] = data.groupby('JOBNUM')[final].fillna(method='bfill')
    return data


def make_ID(sensor_data, n=3):
    ndp = f'Non Duplicate 010{n}'
    filtered = sensor_data[[ndp, 'JOBNUM']].loc[sensor_data.loc[:, ndp] == 1, :].copy()
    filtered.loc[:, 'temp'] = filtered.groupby('JOBNUM')[ndp].cumcount() + 1
    sensor_data.loc[:, 'temp'] = filtered.loc[:, 'temp']
    return sensor_data.groupby('JOBNUM')['temp'].fillna(method='bfill')


def _sensor_groupings(data):
    return data.groupby('JOBNUM').cumcount() + 1


def cumcount_per_010n(original, outer=3):
    try:
        if outer == 2:
            inner = 3
        elif outer == 3:
            inner = 2
        else:
            raise ValueError('Incorrect Column Number')

        data = original.copy(deep=True)
        column = f'010{inner} per 010{outer}'
        bfill = f'010{outer} bfill'
        groupby = data.groupby(['JOBNUM', bfill])['Non Duplicate 0102', 'Non Duplicate 0103']
        data = groupby.apply(do_count, f'Non Duplicate 010{inner}', column)
        original.loc[:, column] = data.loc[:, column].copy()
        return original

    except ValueError:
        raise Exception('Incorrect Column Number')


def create_non_duplicate(sensor_data, groupby, first, column_num):
    indices = groupby.apply(_non_duplicates, f'Indgang 010{column_num}')
    column = f'Non Duplicate 010{column_num}'
    sensor_data.loc[indices, column] = 1

    if column_num is not 1:
        sensor_data.loc[first, column] = sensor_data.loc[first, f'Indgang 010{column_num}']

    sensor_data.loc[:, column] = sensor_data.loc[:, column]\
        .fillna(0)\
        .astype(int)
    return sensor_data


def create_non_duplicates(sensor_data, phantoms=False):
    groupby = sensor_data.groupby('JOBNUM')
    first, _ = _gen_first_and_last(sensor_data)

    for i in range(1, 7):
        if f'Non Duplicate 010{i}' not in sensor_data.columns:

            if f'previous_010{i}' not in sensor_data.columns:
                sensor_data[f'previous_010{i}'] = groupby[[f'Indgang 010{i}']].shift(1)
                sensor_data[f'previous_010{i}'] = sensor_data[f'previous_010{i}'].fillna(0)\
                                                                                 .astype(int)

            create_non_duplicate(sensor_data, groupby, first, i)

    if phantoms:
        sensor_data = _remove_phantom_0103(sensor_data)
    return sensor_data


def single_non_duplicate(column_num, sensor_data):
    if f'previous_010{column_num}' not in sensor_data.columns:
        groupby = sensor_data.groupby('JOBNUM')
        sensor_data[f'previous_010{column_num}'] = groupby[[f'Indgang 010{column_num}']].shift(1)
        sensor_data[f'previous_010{column_num}'] = sensor_data[f'previous_010{column_num}'].fillna(0) \
                                                                                           .astype(int)

    groupby = sensor_data.groupby('JOBNUM')
    first, _ = _gen_first_and_last(sensor_data)
    return create_non_duplicate(sensor_data, groupby, first, column_num)


def _non_duplicates(sensor_data, column):
    condition = (sensor_data[column] == 1) \
                & (sensor_data[f'previous_{column[-4:]}'] == 0)

    indices = sensor_data.loc[condition]
    return pd.Series(indices.index)


def do_count(data_slice, inner, column, fill=None):
    condition = data_slice[inner] != 0
    second_slice = data_slice.loc[condition, :]
    second_slice.loc[:, column] = np.arange(1, len(second_slice.index) + 1)
    data_slice.loc[:, column] = second_slice.loc[:, column]
    return data_slice


def _gen_first_and_last(data):
    data['Duplicated Index'] = data.index
    first = data.groupby('JOBNUM').first().set_index('Duplicated Index', drop=True)
    last = data.groupby('JOBNUM').last().set_index('Duplicated Index', drop=True)
    first_and_last = pd.concat([
        pd.Series(first.index),
        pd.Series(last.index)
    ]).sort_values().reset_index(drop=True)
    data.drop('Duplicated Index', axis=1, inplace=True)
    return first.index, first_and_last


def _remove_phantom_0103(sensor_data):
    """
    machine 0405 often makes the error of recording duplicated sensor data.
    This function produces new columns for all sensor data, taking only the
    first of any duplicated readings
    """
    non_duplicates = sensor_data.loc[sensor_data['Non Duplicate 0103'] == 1, :]
    sensor_data['0103 Cum Count'] = non_duplicates.groupby('JOBNUM').cumcount() + 1
    sensor_data['Non Duplicate 0103'] = sensor_data['Non Duplicate 0103'].loc[sensor_data['0103 Cum Count'] % 2 == 0]
    sensor_data['Non Duplicate 0103'] = sensor_data['Non Duplicate 0103'].fillna(0).astype(np.int8)
    sensor_data.drop('0103 Cum Count', axis=1, inplace=True)
    return sensor_data


def ffill_0102_per_0103(data):
    groupby_2 = data.groupby('JOBNUM')['0102 per 0103']
    data.loc[:, '0102 per 0103 ffill'] = groupby_2.fillna(method='ffill')
    return data
