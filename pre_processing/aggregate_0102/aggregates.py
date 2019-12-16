import numpy as np
import pandas as pd


def drop_first_rows(data):
    """
    The first rows of many JOBNUMs, where many strings enter the machine and no
    ladders leave  contain strange readings that are unrepresentative of the data
    as a whole. If called, this function will drop them.
    """
    indices = data.loc[data.loc[:, '0103 ID'] == 1].index
    return data.drop(indices, axis=0)


def calc_time_delta_last_ladder_out(sensor_data):
    """
    For each row of the sensor data, the time difference is calculated between
    that row and when the last ladder left the machine
    """
    condition = sensor_data['0103 ID'] != sensor_data['prev_0103 ID']
    sensor_data['0103 ID Start'] = sensor_data.loc[condition, 'Date']
    groupby = sensor_data.groupby('JOBNUM')
    sensor_data['0103 ID Start'] = groupby['0103 ID Start'].fillna(method='ffill')
    sensor_data['Time Since Last 0103'] = (
            sensor_data['Date'] - sensor_data['0103 ID Start']
    ).dt.total_seconds().astype(int)
    return sensor_data


def sum_num_pace_ins_larger_than_n(data, n, n_rows_back=30):
    """
    Groups on JOBNUM, looks back a maximum of n_rows_back and sums the number
    of pace-ins longer than n
    """
    groupby = data.groupby('JOBNUM')
    return groupby.apply(_sum_num_pace_ins_larger_than_n, n, n_rows_back)\
                  .reset_index(drop=True)


def _sum_num_pace_ins_larger_than_n(data, n, n_rows_back):
    """
    Iterates through each pace >= n ID in each JOBNUM and calculates how many
    pace >= n occured n_rows_back
    """
    ids = data.loc[data[f'0102 Pace >= {n} ID'] >= 1, :]
    for index, row in ids.iterrows():
        """
        check whether there are less than n_rows_back before the 
        0102 pace >= n ID         
        """
        if index - n_rows_back >= data.index[0]:
            sliced = data.loc[index - n_rows_back:index + 1, :]
        else:
            sliced = data.loc[data.index[0]:index+1, :]

        data.loc[index, f'0102 Sum Pace >= {n}'] = sliced\
            .aggregate({f'0102 Pace >= {n} Count': 'sum'})\
            .squeeze() - 1
    return data
