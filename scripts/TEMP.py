from load_data import load_csv
from data_preparation import sensor_data as sd
from data_preparation import work_table as wt
from models import machines
import os
import pandas as pd
import numpy as np
from statisticss import sensor_data_stats as sds


def first_last_deactivations(sensor_data):
    sensor_data = sensor_data.copy()
    sensor_data.loc[:, 'duplicate_index'] = sensor_data.index
    groupby = sensor_data.groupby('JOBNUM')
    aggregation = {'Indgang 0101': ['first', 'last'], 'duplicate_index': ['first', 'last']}
    result = groupby.agg(aggregation)
    first = get_slice(result, 'first')
    last = get_slice(result, 'last')
    affected_indices = pd.concat([first, last])
    aggregation = {'Indgang 0101': 'sum'}
    grouped_deactivations = groupby.agg(aggregation)
    return affected_indices, len(affected_indices), grouped_deactivations


def remove_index_and_f_index(worktable, data_slice, condition):
    indices_to_delete = np.where(
        condition,
        data_slice.index,
        data_slice.index + 1
    )
    worktable = worktable.loc[~worktable.index.isin(indices_to_delete)].copy(deep=True)
    return worktable


def remove_jobrefs(worktable, data_slice):
    job_refs = data_slice['JOBREF']
    f_job_refs = data_slice['f_JOBREF']
    erroneous_jobs = pd.concat([job_refs, f_job_refs])
    erroneous_jobs = set(erroneous_jobs)
    worktable = worktable.loc[~worktable['JOBREF'].isin(erroneous_jobs), :].reset_index(drop=True)
    return worktable


def unique_error_count(data_slice, worktable):
    index_current = pd.Series(data_slice.index)
    index_next = pd.Series(data_slice.index + 1)
    indices = pd.concat([index_current, index_next])

    count_error = len(set(indices))
    count_unique_jobrefs = len(set(worktable['JOBREF']))
    return count_error, count_unique_jobrefs


def get_slice(data, column):
    condition = data[('Indgang 0101', column)] == 1
    data_slice = data.loc[condition, ('duplicate_index', column)]
    return data_slice


machine = machines.Machine1405()
columns = machine.data_generation_columns
current = os.getcwd()
path_in = os.path.join(current, r'csvs')
csv = os.path.join(path_in, r'WORK_TABLE.csv')
work_table = load_csv.work_table(csv, columns)
work_table = wt.prepare_work_table(work_table)
work_table = wt.clean_work_table(work_table)
work_table = wt.filter_work_table(work_table, reg_ex=r'^SW|^CF')

data_path_in = os.path.join(current, r'csvs/01-01-18 to 01-01-19/datacollection.csv')
original_data = pd.read_csv(data_path_in, sep=';', parse_dates=['Date'], infer_datetime_format=True)
original_data.drop('Unnamed: 0', axis=1, inplace=True)
sensor_data = original_data.copy(deep=True)

sensor_data = sd.filter_sensor_data(work_table, original_data)
sensor_data['Indgang 0101'] = abs(sensor_data['Indgang 0101'] - 1)
sensor_data[['next_0101', 'next_Date']] = sensor_data[['Indgang 0101', 'Date']].shift(-1)
sensor_data = sd.create_non_duplicates(sensor_data, columns, phantoms=False)
sensor_data = sd.calc_single_row_0101_duration(sensor_data)
sensor_data = sd.calc_multi_row_0101_duration(sensor_data)
breaks = sd.get_sensor_data_breaks(sensor_data)

work_table_new, multis, singles = wt.fix_work_table(breaks, work_table, sensor_data)
work_table_new = work_table_new.sort_values('StartDateTime').reset_index(drop=True)
sensor_data_new = sd.filter_sensor_data(work_table_new, original_data)
sensor_data_new['Indgang 0101'] = abs(sensor_data_new['Indgang 0101'] - 1)
sensor_data_new = sd.set_first_and_last_row_to_zero(sensor_data)

sensor_data_new = sd.create_non_duplicates(sensor_data_new, columns, phantoms=True)
sensor_data_new = sd.calculate_pace(sensor_data_new, columns)
stats = sds.generate_statistics(sensor_data_new, work_table_new, machine.generate_statistics)
cols, stats = sd.feature_extraction(stats)
# sds.products_dummies(stats)
corr = stats.corr()

# sensor_data = sd.filter_sensor_data(work_table, sensor_data)
# sensor_data['Indgang 0101'] = abs(sensor_data['Indgang 0101'] - 1)
#
# affected_indices, affected, grouped_deactivations = first_last_deactivations(sensor_data)
#
# sensor_data = sensor_data.copy(deep=True)
# previous_columns = ["prev_Indgang 0101", "prev_Indgang 0105", "prev_JOBNUM"]
# next_columns = ["next_Indgang 0101", "next_Indgang 0105", "next_JOBNUM"]
# current_columns = ["Indgang 0101", "Indgang 0105", "JOBNUM"]
#
# copy[previous_columns] = copy[current_columns].shift(1)
# copy[next_columns] = copy[current_columns].shift(-1)
#
# condition = (copy["prev_Indgang 0101"] == 1) \
#             & (copy["Indgang 0101"] == 1) \
#             & (copy["Indgang 0105"] != copy["prev_Indgang 0105"]) \
#             & (copy["prev_JOBNUM"] == copy["JOBNUM"]) \
#             | (copy["next_Indgang 0101"] == 1) \
#             & (copy["Indgang 0101"] == 1) \
#             & (copy["Indgang 0105"] != copy["next_Indgang 0105"]) \
#             & (copy["next_JOBNUM"] == copy["JOBNUM"])
#
# data_slice = copy.loc[condition, :]
# groupby_alternating = data_slice.groupby('JOBNUM')
#
# output = groupby_alternating.agg({'Indgang 0101': 'sum'})

# condition_duplicate_0101 = (copy['Indgang 0101'] == 1) \
#                            & (copy['prev_Indgang 0101'] == 1) \
#                            & (~copy.isin([873, 1202, 1120, 698, 1141, 1032, 1211, 995])) \
#                            | (copy['Indgang 0101'] == 1) \
#                            & (copy['next_Indgang 0101'] == 1) \
#                            & (~copy.isin([873, 1202, 1120, 698, 1141, 1032, 1211, 995]))
#
# duplicates_0101s = copy.loc[condition_duplicate_0101, :]


# copy = sensor_data.copy(deep=True)
#
# current_values = ['Indgang 0102', 'Indgang 0103']
# previous_values = ['previous_0102', 'previous_0103']
# next_values = ['next_0102', 'next_0103']
#
# copy_groupby = copy.groupby('JOBNUM')
#
# copy[previous_values] = copy_groupby[current_values].shift(1)
# next_values = ['next_0102', 'next_0103']
# copy[next_values] = copy_groupby[current_values].shift(-1)
#

# mode = mode(aggregations_0102[('Duration', '')])
#
# sensor_data.drop('Time', axis=1, inplace=True)

