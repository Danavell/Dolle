import os
import datetime as dt

import numpy as np
import pandas as pd


class SensorDataCleaner1405:
    """
    Contains machine 1405 sensor data attributes and concrete implementation of
    prep_sensor_data
    """
    def __init__(self, fix_duplicates):
        self.deactivation_sensor_id = '0101'
        self.sensor_breaks_columns = [
            'Date', 'Non Duplicate 0101', '0101 Duration'
        ]
        self.original_columns = [
            'Date', 'Indgang 0101', 'Indgang 0102', 'Indgang 0103',
            'Indgang 0104', 'Indgang 0105', 'Indgang 0106'
        ]
        self.sensor_data = None
        self._fix_duplicates = fix_duplicates

    def prep_sensor_data(self, work_table):
        self.sensor_data = filter_sensor_data(work_table, self.sensor_data)
        self.sensor_data['Indgang 0101'] = abs(self.sensor_data['Indgang 0101'] - 1)
        self.sensor_data = single_non_duplicate(1, self.sensor_data)
        self.sensor_data = calc_on_off_data(self.sensor_data)
        if self._fix_duplicates:
            self.sensor_data = fix_0103(self.sensor_data)
        return self.sensor_data


def filter_three_main_ladders_1405(work_table):
    """
    filters work_table to contain only the three most popular ladders produced
    by machine 1405
    """
    condition = (work_table.loc[:, 'CF/3D/3F/2B/12T'] == 1) | \
                (work_table.loc[:, 'CF/3D/4F/4B/12T'] == 1) | \
                (work_table.loc[:, 'SW/3D/3F/3B/12T'] == 1)
    return work_table.loc[condition, :]


def filter_SW_or_CF_1405(work_table):
    """
    filters work_table to contain only ladders whose name starts with SW or CF
    for machine 1405
    """
    condition = (work_table['NAME'].str.contains('^SW|CF', regex=True)) & \
                (work_table['WRKCTRID'] == 1405)
    return work_table.loc[condition, :]


def get_dummies_concat(data):
    products = pd.get_dummies(data['NAME'])
    return pd.concat([data, products], axis=1)


def prepare_base_data(wt_cleaner, sd_cleaner, base_data=False):

    """
    1. REMOVE ALL ERRORS CONTAINED IN THE WORK TABLE
    """
    work_table = wt_cleaner.prep_work_table()

    """
    2. GROUP AND FILTER SENSOR_DATA ACCORDING TO THE JOBNUMS IN THE WORKTABLE
    """
    sensor_data = sd_cleaner.prep_sensor_data(work_table)
    sensor_id = sd_cleaner.deactivation_sensor_id

    """
    3. RETURN ALL THE ROWS IN THE SENSOR DATA THAT OCCUR DURING BREAK TIMES
    """
    breaks = get_sensor_data_breaks(sensor_data, sensor_id)

    """
    4. SPLIT ANY ROWS IN THE WORK TABLE THAT HAVE BREAKS. THE FIRST NEW ROW ENDS AT THE BEGINNING 
    OF THE BREAK AND THE NEW ROW AFTER BEGINS AT THE END OF THE BREAK
    """
    work_table_new = wt_cleaner.remove_breaks(sensor_data, breaks)

    """
    5. LIKE IN STEP 1, GROUP AND FILTER SENSOR DATA BUT USING THE NEW WORKTABLE MADE IN STEP 3
    """
    sensor_data.drop('JOBNUM', axis=1, inplace=True)
    sensor_data_new = filter_sensor_data(work_table_new, sensor_data)

    """
    6. SOME JOBS END WITH A DEACTIVATION. THESE NEED TO BE SET TO 'ON'.
    SOME JOBS HAD MORE THAN ONE DEACTIVATED ROW AT THE END OF THE JOB. THESE DEACTIVATIONS MUST 
    BE SET TO 'ON' 
    """
    sensor_data_new = set_first_and_last_row_to_zero(sensor_data_new, sensor_id)

    columns = sd_cleaner.sensor_breaks_columns
    indices = get_sensor_data_breaks(sensor_data_new[columns], sensor_id).index
    sensor_data_new.loc[indices, f'Indgang {sensor_id}'] = 0
    sensor_data_new.loc[indices, f'Non Duplicate {sensor_id}'] = 0
    if base_data:
        sensor_data_new = sensor_data_new.loc[:, sd_cleaner.original_columns]
    return work_table_new, sensor_data_new


def filter_sensor_data(work_table, sensor_data):
    filtered_data = pd.DataFrame()
    for _, row in work_table.iterrows():
        start = row['StartDateTime']
        stop = row['StopDateTime']
        condition = sensor_data['Date'].between(start, stop, inclusive=True)
        new_data = sensor_data.loc[condition, :].copy()
        if len(new_data.index) > 0:
            new_data.loc[:, 'JOBREF'] = row['JOBREF']
            new_data.loc[:, 'JOBNUM'] = row['JOBNUM']
            new_data.loc[:, 'NAME'] = row['NAME']
            filtered_data = pd.concat([filtered_data, new_data], sort=True)
    filtered_data = filtered_data.sort_values('Date').reset_index(drop=True)
    return filtered_data


def get_sensor_data_breaks(sensor_data, sensor_id):
    sensor_data.loc[:, 'weekday'] = sensor_data.loc[:, 'Date'].apply(lambda x: dt.datetime.strftime(x, '%A')).copy()
    condition = \
        (sensor_data['weekday'] != 'Friday') \
        & (sensor_data['Date'].dt.strftime('%H:%M:%S').between('08:55:00', '09:20:00')) \
        | (sensor_data['weekday'] != 'Friday') \
        & (sensor_data['Date'].dt.strftime('%H:%M:%S').between('11:55:00', '12:25:00')) \
        | (sensor_data['weekday'] != 'Friday') \
        & (sensor_data['Date'].dt.strftime('%H:%M:%S').between('18:25:00', '18:55:00')) \
        | (sensor_data['weekday'] != 'Friday') \
        & (sensor_data['Date'].dt.strftime('%H:%M:%S').between('21:25:00', '21:50:00')) \
        | (sensor_data['weekday'] == 'Friday') \
        & (sensor_data['Date'].dt.strftime('%H:%M:%S').between('11:25:00', '11:55:00'))
    breaks = sensor_data.loc[condition]
    sensor_data_breaks = breaks.loc[breaks[f'Non Duplicate {sensor_id}'] == 1]
    sensor_data_breaks = sensor_data_breaks.loc[sensor_data_breaks[f'{sensor_id} Duration'] > 1200].copy()
    return sensor_data_breaks


def initial_fix(work_table, sensor_data, fix_duplicates=False):
    sensor_data = filter_sensor_data(work_table, sensor_data)
    sensor_data['Indgang 0101'] = abs(sensor_data['Indgang 0101'] - 1)
    sensor_data = single_non_duplicate(1, sensor_data)
    sensor_data = calc_on_off_data(sensor_data)
    if fix_duplicates:
        sensor_data = fix_0103(sensor_data)
    return sensor_data


def set_first_and_last_row_to_zero(sensor_data, sensor_id):
    """
    First and last indices of Indgang 0101 are set to 0
    """
    first, first_and_last = _gen_first_and_last(sensor_data)
    sensor_data.loc[first_and_last, f'Indgang {sensor_id}'] = 0
    return sensor_data


def single_non_duplicate(column_num, sensor_data):
    if f'previous_010{column_num}' not in sensor_data.columns:
        groupby = sensor_data.groupby('JOBNUM')
        sensor_data[f'previous_010{column_num}'] = groupby[[f'Indgang 010{column_num}']].shift(1)
        sensor_data[f'previous_010{column_num}'] = sensor_data[f'previous_010{column_num}']\
            .fillna(0) \
            .astype(int)

    groupby = sensor_data.groupby('JOBNUM')
    first, _ = _gen_first_and_last(sensor_data)
    return create_non_duplicate(sensor_data, groupby, first, column_num)


def calc_on_off_data(sensor_data):
    sensor_data[['next_0101', 'next_Date']] = sensor_data[['Indgang 0101', 'Date']].shift(-1)
    sensor_data = calc_single_row_0101_duration(sensor_data)
    sensor_data = calc_multi_row_0101_duration(sensor_data)
    return sensor_data


def fix_0103(sensor_data, keep=True):
    """
    Remove duplicate readings in the Indang 0103 column caused by a sensor
    accidently being activated twice
    """
    sensor_data = single_non_duplicate(3, sensor_data)
    sensor_data = make_column_2_levels(sensor_data, 'Indgang 0103', 'Non Duplicate 0103', '0103 CC Group')

    sensor_data = create_unique_id(sensor_data, '0103 Even CC Group')

    if keep:
        sensor_data = create_unique_id(sensor_data, '0103 Odd CC Group')
        sensor_data.loc[:, 'Original 0103'] = sensor_data.loc[:, 'Indgang 0103']

    condition = sensor_data.loc[:, '0103 Group'] != 0
    indices = sensor_data.loc[condition].index

    condition = sensor_data.index.isin(indices)
    indices = sensor_data.loc[~condition].index

    sensor_data.loc[indices, 'Indgang 0103'] = 0
    sensor_data.drop('Non Duplicate 0103', axis=1, inplace=True)
    return sensor_data


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


def calc_single_row_0101_duration(sensor_data):
    condition = (sensor_data['next_0101'] == 0) \
                & (sensor_data['previous_0101'] == 0) \
                & (sensor_data['Indgang 0101'] == 1)

    time_diff_0101 = sensor_data.loc[condition].copy()
    time_diff_0101['Duration'] = time_diff_0101['next_Date'] - time_diff_0101['Date']
    time_diff_0101['Duration'] = time_diff_0101['Duration'].dt.total_seconds()
    sensor_data['0101 Duration'] = time_diff_0101['Duration']
    sensor_data['0101 Duration'] = sensor_data['0101 Duration'].fillna(0).astype(int)
    return sensor_data


def calc_multi_row_0101_duration(sensor_data):
    condition = (sensor_data['Indgang 0101'] == 1) \
                & (sensor_data['previous_0101'] == 1) \
                | (sensor_data['Indgang 0101'] == 1) \
                & (sensor_data['next_0101'] == 1)

    multi_time_0101 = make_column_2_levels(sensor_data,
                                           None,
                                           'Non Duplicate 0101',
                                           '0101 Group',
                                           condition_1=condition,
                                           cumcount=False,
                                           fillna=False)

    multi_time_0101['Duplicate Index'] = multi_time_0101.index

    agg_dict = {
        'Date': 'first',
        'next_Date': 'last',
        'Duplicate Index': 'first'
    }

    mt_0101_groupby = multi_time_0101.groupby('0101 Group').agg(agg_dict)
    mt_0101_groupby.set_index('Duplicate Index', inplace=True)

    mt_0101_groupby['delta_time'] = calc_group_pace(mt_0101_groupby, future=True)
    sensor_data = add_column(sensor_data, multi_time_0101, '0101 Group')
    sensor_data = add_column(sensor_data, mt_0101_groupby, '0101 Duration',
                             child_col='delta_time', indices=mt_0101_groupby.index)
    return sensor_data


def make_column_2_levels(sensor_data,
                         group_column,
                         non_duplicate_column,
                         output_column,
                         condition_1=None,
                         condition_2=None,
                         cumcount=True,
                         fillna=True):
    """
    make a new column that requires two successive sub-slices. If the user sets
    cumcount == False, np.arange will be used instead
    """
    data = sensor_data.copy(deep=True)
    second_slice = slice_data(data, group_column, mask=condition_1)
    third_slice = slice_data(second_slice, non_duplicate_column, mask=condition_2)

    if cumcount:
        third_slice.loc[:, 'temp'] = third_slice.groupby('JOBNUM').cumcount() + 1
    else:
        third_slice.loc[:, 'temp'] = np.arange(1, len(third_slice.index) + 1)

    second_slice.loc[:, 'temp'] = third_slice.loc[:, 'temp'].copy()
    second_slice.loc[:, 'temp'] = second_slice.groupby('JOBNUM')['temp'].fillna(method='ffill')

    data.loc[:, output_column] = second_slice.loc[:, 'temp']

    if fillna:
        data.loc[:, output_column] = data.loc[:, output_column]\
            .fillna(0)\
            .astype(int)
    return data


def create_unique_id(data, column_name):
    """
    adds a column to the sensor data containing unique identifiers for either
    the odd or even column
    """
    return convert_non_unique_identifier_into_unique(
        data, column_name, '0103 CC Group',
        'Non Duplicate 0103', '0103 Group'
    )


def convert_non_unique_identifier_into_unique(sensor_data,
                                              odd_or_even_column,
                                              odd_and_even_column,
                                              non_duplicate_column,
                                              output_column,
                                              even=True):
    """
    Takes a column with non-unique cum-counted group names per JOBNUM and returns unique
    group names valid across the entire dataframe
    """
    sensor_data = return_even_or_odd_groups(
        odd_and_even_column, odd_or_even_column, sensor_data, even
    )
    return make_column_2_levels(sensor_data,
                                odd_or_even_column,
                                non_duplicate_column,
                                output_column,
                                cumcount=False
                                )


def calc_group_pace(dates, future=False):
    if future:
        times = dates['next_Date'] - dates['Date']
    else:
        times = dates['Date'] - dates['previous_Date']
    return times.dt.total_seconds().fillna(0).astype(int)


def add_column(parent, child, col, child_col=None, fillna=True, indices=None):
    parent_col = col
    if not isinstance(child_col, str):
        child_col = col

    if isinstance(indices, pd.Int64Index):
        parent.loc[indices, parent_col] = child.loc[indices, child_col]
    else:
        parent.loc[:, parent_col] = child.loc[:, parent_col]

    if fillna:
        parent.loc[:, parent_col] = parent.loc[:, parent_col]\
                                          .fillna(0)\
                                          .astype(int)
    return parent


def return_even_or_odd_groups(column, output_column, sensor_data, even=True):
    """
    returns either all the odd or even numbered cum-counted groups in a JOBNUM
    """
    if even:
        con = (sensor_data.loc[:, column] % 2 == 0)
    else:
        con = (sensor_data.loc[:, column] % 2 != 0)
    condition = (sensor_data.loc[:, column] > 0) & con
    second_slice = sensor_data.loc[condition, :].copy()
    sensor_data.loc[:, output_column] = second_slice.loc[:, column]
    sensor_data.loc[:, output_column] = sensor_data.loc[:, output_column]\
                                                   .fillna(0)\
                                                   .astype(int)
    return sensor_data


def _non_duplicates(sensor_data, column):
    condition = (sensor_data[column] == 1) \
                & (sensor_data[f'previous_{column[-4:]}'] == 0)

    indices = sensor_data.loc[condition]
    return pd.Series(indices.index)


def slice_data(sensor_data, column, mask=None):
    """
    returns a slice of sensor data based on a condition
    """
    if isinstance(mask, pd.Series):
        condition = mask
    else:
        condition = sensor_data.loc[:, column] != 0
    sensor_data = sensor_data.loc[condition, :].copy()
    return sensor_data


