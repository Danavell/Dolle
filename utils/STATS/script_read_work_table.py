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
    worktable = worktable.loc[~worktable.index.isin(indices_to_delete)]
    return worktable


def remove_jobrefs(worktable, data_slice):
    job_refs = data_slice['JOBREF']
    f_job_refs = data_slice['f_JOBREF']
    erroneous_jobs = pd.concat([job_refs, f_job_refs])
    erroneous_jobs = set(erroneous_jobs)
    worktable = worktable.loc[~worktable['JOBREF'].isin(erroneous_jobs), :]
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
path_in = os.path.join(current, r'csvs\01-01-18 to 01-01-19\WORK_TABLE.csv')
work_table = pd.read_csv(path_in, sep=',',
                   usecols=columns['init_work_prepared'],
                   encoding="ISO-8859-1",
                   parse_dates=['StartDateTime', 'StopDateTime'],
                   infer_datetime_format=True)

work_table.sort_values('StartDateTime', inplace=True)

work_table = work_table.loc[~work_table[['JOBREF',
                                         'QTYGOOD',
                                         'StartDateTime',
                                         'StopDateTime',
                                         'Seconds',
                                         'NAME']].duplicated(keep='first')]

work_table = work_table.loc[work_table['WRKCTRID'] == 1405, :]
work_table = work_table.reset_index(drop=True)

next_columns = ['f_StartDateTime', 'f_StopDateTime', 'f_QTYGOOD', 'f_JOBREF', 'f_NAME']
original_columns = ['StartDateTime', 'StopDateTime', 'QTYGOOD', 'JOBREF', 'NAME']

work_table[next_columns] = work_table[original_columns].shift(-1)

diff_jobrefs = work_table[
    # partial overlap
    (work_table['f_StopDateTime'] > work_table['StopDateTime'])
    & (work_table['StopDateTime'] > work_table['f_StartDateTime'])
    & (work_table['f_StartDateTime'] > work_table['StartDateTime'])
    & (work_table['JOBREF'] != work_table['f_JOBREF'])

    # complete overlap
    | (work_table['StartDateTime'] == work_table['f_StartDateTime'])
    & (work_table['StopDateTime'] == work_table['f_StopDateTime'])
    & (work_table['JOBREF'] != work_table['f_JOBREF'])

    # n + 1 overlap
    | (work_table['StartDateTime'] == work_table['f_StartDateTime'])
    & (work_table['StopDateTime'] < work_table['f_StopDateTime'])
    & (work_table['JOBREF'] != work_table['f_JOBREF'])

    # n overlap
    | (work_table['StartDateTime'] < work_table['f_StartDateTime'])
    & (work_table['StopDateTime'] > work_table['f_StopDateTime'])
    & (work_table['JOBREF'] != work_table['f_JOBREF'])
]

work_table = remove_jobrefs(work_table, diff_jobrefs)

complete_overlaps = work_table[
    (work_table['StartDateTime'] == work_table['f_StartDateTime'])
    & (work_table['StopDateTime'] == work_table['f_StopDateTime'])
]

work_table = remove_index_and_f_index(
    worktable=work_table,
    data_slice=complete_overlaps,
    condition=complete_overlaps['QTYGOOD'] == 0
)

n_1_overlaps = work_table[
   (work_table['StartDateTime'] == work_table['f_StartDateTime'])
   & (work_table['StopDateTime'] < work_table['f_StopDateTime'])
]

work_table = remove_index_and_f_index(
    worktable=work_table,
    data_slice=n_1_overlaps,
    condition=n_1_overlaps['QTYGOOD'] == 0
)

n_1_overlaps_zero = work_table[
   (work_table['StartDateTime'] == work_table['f_StartDateTime'])
   & (work_table['StopDateTime'] < work_table['f_StopDateTime'])
   & (work_table['QTYGOOD'] == 0)
   & (work_table['f_QTYGOOD'] == 0)
]

time_diff = n_1_overlaps_zero['f_StopDateTime'] - n_1_overlaps_zero['StopDateTime']
n_1_overlaps_zero['time diff'] = time_diff.dt.total_seconds().fillna(0).astype(int)

work_table = remove_index_and_f_index(
    worktable=work_table,
    data_slice=n_1_overlaps_zero,
    condition=n_1_overlaps_zero['time diff'] > 0
)

n_1_overlaps_non_zero = work_table[
   (work_table['StartDateTime'] == work_table['f_StartDateTime'])
   & (work_table['StopDateTime'] < work_table['f_StopDateTime'])
   & (work_table['QTYGOOD'] >= 0)
   & (work_table['f_QTYGOOD'] >= 0)
]

if len(n_1_overlaps_non_zero.index) != 0:
    work_table = remove_jobrefs(work_table, n_1_overlaps_non_zero)
    work_table.sort_values('StartDateTime', inplace=True)

n_overlaps_zero_n_plus_1 = work_table[
   (work_table['StartDateTime'] < work_table['f_StartDateTime'])
   & (work_table['StopDateTime'] > work_table['f_StopDateTime'])
   & (work_table['f_QTYGOOD'] == 0)
]

indices_to_delete = n_overlaps_zero_n_plus_1.index + 1
work_table = work_table[~work_table.index.isin(indices_to_delete)]

n_overlaps_non_zero_n_plus_1 = work_table[
   (work_table['StartDateTime'] < work_table['f_StartDateTime'])
   & (work_table['StopDateTime'] > work_table['f_StopDateTime'])
   & (work_table['QTYGOOD'] == 0)
   & (work_table['f_QTYGOOD'] != 0)
]

n_overlaps_non_zero_n_plus_1['QTYGOOD'] = n_overlaps_non_zero_n_plus_1['f_QTYGOOD']

indices_to_delete = n_overlaps_non_zero_n_plus_1.index + 1
work_table = work_table[~work_table.index.isin(indices_to_delete)]

condition = (work_table['f_StopDateTime'] > work_table['StopDateTime']) \
            & (work_table['StopDateTime'] > work_table['f_StartDateTime']) \
            & (work_table['f_StartDateTime'] > work_table['StartDateTime']) \
            & (work_table['JOBREF'] != work_table['f_JOBREF'])

partial_overlaps = work_table.loc[condition]

work_table = load_csv.filter_work_table(work_table, machine_id=1405, reg_ex=r'^SW|^CF')

data_path_in = os.path.join(current, r'csvs\01-01-18 to 01-01-19\datacollection.csv')
original_data = pd.read_csv(data_path_in, sep=';', parse_dates=['Date'], infer_datetime_format=True)
original_data.drop('Unnamed: 0', axis=1, inplace=True)
sensor_data = original_data.copy(deep=True)

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
sensor_data_new = sd.create_non_duplicates(sensor_data_new, columns, phantoms=True)
sensor_data_new = sd.calculate_pace(sensor_data_new, columns)
stats = sds.generate_statistics(sensor_data_new, work_table_new, machine.generate_statistics)
cols, stats = sd.feature_extraction(stats)
# sds.products_dummies(stats)
corr = stats.corr()


def create_non_duplicate_0102_frame(sensor_data_new):
    sensor_data_new_groupby = sensor_data_new.groupby('JOBNUM')
    next_values = ['next_0102', 'next_0103']
    current_values = ['Indgang 0102', 'Indgang 0103']
    sensor_data_new[next_values] = sensor_data_new_groupby[current_values].shift(-1)
    condition = (sensor_data_new['Indgang 0102'] == 1) \
                & (sensor_data_new['previous_0102'] == 1) \
                | (sensor_data_new['Indgang 0102'] == 1) \
                & (sensor_data_new['next_0102'] == 1)
    only_sequential_duplicates = sensor_data_new.loc[condition].copy()
    condition = only_sequential_duplicates['Non Duplicate 0102'] == 1
    non_duplicate_0102_frame = only_sequential_duplicates.loc[condition].copy()
    return non_duplicate_0102_frame, only_sequential_duplicates


def create_only_sequential_duplicates(sensor_data_new):
    non_duplicate_0102_frame, only_sequential_duplicates = create_non_duplicate_0102_frame(sensor_data_new)
    stop = len(non_duplicate_0102_frame.index) + 1
    non_duplicate_0102_frame['count'] = np.arange(1, stop)
    only_sequential_duplicates.loc[:, 'count'] = non_duplicate_0102_frame['count']
    osd_groupby = only_sequential_duplicates.groupby('JOBNUM')
    only_sequential_duplicates['0102 Duplicate Group'] = osd_groupby['count'].fillna(method='ffill')
    only_sequential_duplicates['0102 Duplicate Group'].fillna(0, inplace=True)
    return only_sequential_duplicates


def first_attempt(sensor_data_new):
    only_sequential_duplicates = create_only_sequential_duplicates(sensor_data_new)
    osd_groupby_2 = only_sequential_duplicates.groupby('0102 Duplicate Group')
    aggregation_dict = {'JOBNUM': 'first',
                        'Non Duplicate 0101': 'sum',
                        '0102 Duplicate Group': 'first',
                        'Indgang 0102': 'sum',
                        'Date': ['first', 'last']}
    aggregations_0102 = osd_groupby_2.agg(aggregation_dict)
    aggregations_0102['Duration'] = aggregations_0102[('Date', 'last')] - aggregations_0102[('Date', 'first')]
    aggregations_0102['Duration'] = aggregations_0102['Duration'].dt.total_seconds()
    return aggregations_0102


def second_attempt(sensor_data_new):
    only_sequential_duplicates = create_only_sequential_duplicates(sensor_data_new)
    temp = only_sequential_duplicates.loc[only_sequential_duplicates['Non Duplicate 0101'] > 0].copy()
    groups = set(temp['0102 Duplicate Group'])
    grouped_0102_deacs = only_sequential_duplicates.loc[only_sequential_duplicates['0102 Duplicate Group'].isin(groups)]
    grouped_0102_deacs['Non Duplicate 0101'].replace(0, np.nan, inplace=True)
    gd = grouped_0102_deacs.groupby('JOBNUM')
    grouped_0102_deacs['Non Duplicate 0101'] = gd['Non Duplicate 0101'].fillna(method='ffill')
    grouped_0102_pre_deac = grouped_0102_deacs.loc[grouped_0102_deacs['Non Duplicate 0101'] != 1]
    gd_2 = grouped_0102_pre_deac.groupby('JOBNUM')
    aggregation_dict = {'JOBNUM': 'first',
                        '0102 Duplicate Group': 'first',
                        'Indgang 0102': 'sum',
                        'Date': 'first',
                        'next_Date': 'last'}

    aggregations_0102 = gd_2.agg(aggregation_dict)
    aggregations_0102['Duration'] = aggregations_0102['next_Date'] - aggregations_0102['Date']
    aggregations_0102['Duration'] = aggregations_0102['Duration'].dt.total_seconds()
    sensor_data_new['Non Duplicate 0102 Group'] = only_sequential_duplicates['0102 Duplicate Group']
    return aggregations_0102


first_aggregations = first_attempt(sensor_data_new)
products, work_table_newest = wt.get_product_dummies(work_table_new)

columns = \
    ['JOBNUM', 'CF/3D/3F/2B/12T', 'CF/3D/4F/3B/12T', 'CF/3D/4F/4B/12T',
    'CF/4D/3F/2B/12T', 'CF/4D/3F/3B/11T', 'CF/4D/3F/3B/12T',
    'CF/4D/4F/3B/12T', 'CF/4D/4F/4B/12T', 'SW/2D/4F/4B/3T',
    'SW/2D/4F/4B/4T', 'SW/3D/3F/3B/11T', 'SW/3D/3F/3B/12T',
    'SW/3D/3G/3B/12T']

work_table_jobnum_products = work_table_newest[columns].copy()

work_table_jobnum_products.set_index('JOBNUM', drop=True, inplace=True)
work_table_jobnum_products.sort_index(inplace=True)

sensor_data_newest = pd.merge(left=sensor_data_new, right=work_table_jobnum_products, left_on='JOBNUM', right_on='JOBNUM')


results = pd.DataFrame()
product_range = products.columns
for i in range(len(product_range)):
    condition = (sensor_data_newest[product_range[i]] == 1)
    data = sensor_data_newest.loc[condition].copy()
    resultant = pd.DataFrame()
    for j in range(10, 110, 10):
        aggregation = second_attempt(data)
        
# statisticss' code below -----------------------------------------------------------------------------------------------

def error_grouping(df):
    copy = df.loc[df['Non Duplicate 0101'] == 1].copy()
    copy['shifted_date'] = copy['Date']
    groupby = copy.groupby([copy['JOBNUM']])[['shifted_date']].shift(1)
    groupby['Date'] = copy['Date'].copy()
    groupby['JOBNUM'] = copy['JOBNUM'].copy()
    groupby['delta_time'] = (groupby['Date'] - groupby['shifted_date']).dt.total_seconds()
    return groupby


_groupby = error_grouping(sensor_data_new)


def error_merging(df,grouped_df):
    copy = df.copy()
    groupby = copy.groupby('JOBNUM')
    aggregation = {'Date': ['first', 'last']}
    result = groupby.agg(aggregation)
    result['JOBNUM'] = result.index
    merged = grouped_df.merge(right=result, left_on='JOBNUM', right_on='JOBNUM').copy()
    merged['seconds_since_start'] = (merged['Date'] - merged[('Date', 'first')]).dt.total_seconds()
    merged['seconds_until_end'] = (merged[('Date', 'last')] - merged['Date']).dt.total_seconds()
    return merged


grouped_errors = error_merging(sensor_data_new, _groupby)

# errors which are ending the job
# _ = grouped_errors['seconds_until_end'] == 0
# _ = grouped_errors.loc[grouped_errors['seconds_until_end'] == 0].copy()
# ending_error_jobnums = _['JOBNUM']
# ending_error_jobnums = ending_error_jobnums.sort_values()
#
# condition = (sensor_data_new['JOBNUM'].isin(ending_error_jobnums))
# jobs_ending_with_error = sensor_data_new.loc[condition]

# dummies preparation
_, d = wt.get_product_dummies(work_table_new)
dummies = d[columns].copy()
merged_with_dummies = grouped_errors.merge(dummies, left_on='JOBNUM', right_on='JOBNUM').copy()

# to remove the first error of the grouped errors
merged_with_dummies_without_nan = merged_with_dummies.dropna().copy()
columns = \
    ['CF/3D/3F/2B/12T', 'CF/3D/4F/3B/12T', 'CF/3D/4F/4B/12T',
    'CF/4D/3F/2B/12T', 'CF/4D/3F/3B/11T', 'CF/4D/3F/3B/12T',
    'CF/4D/4F/3B/12T', 'CF/4D/4F/4B/12T', 'SW/2D/4F/4B/3T',
    'SW/2D/4F/4B/4T', 'SW/3D/3F/3B/11T', 'SW/3D/3F/3B/12T',
    'SW/3D/3G/3B/12T']


def deact_handling(df, columns, length):
    deact = df[columns].sum() / length * 100
    return deact


error_gonna_happen_again = deact_handling(merged_with_dummies, columns, len(merged_with_dummies_without_nan.index))
condition = (merged_with_dummies['delta_time'] < 600)
under600 = merged_with_dummies.loc[condition]
error_gonna_happen_again_in_10 = deact_handling(under600, columns, len(merged_with_dummies_without_nan.index))


def mean_delta_time(df):
    groupby = df.groupby('JOBNUM')
    aggregation = {'JOBNUM': 'first', 'delta_time': 'mean'}
    delta_time = groupby.agg(aggregation)
    return delta_time


# mean delta time / JOBNUM
m_delta_time = mean_delta_time(merged_with_dummies)
m_delta_time.dropna(inplace=True)
m_delta_time_under_10 = mean_delta_time(under600)


def merge_x_with_product_names(wt, df, on_what):
    copy = wt[['JOBNUM', 'Product']].copy()
    df = df.merge(copy, left_on='JOBNUM', right_on='JOBNUM').copy()
    groupby = df.groupby('Product')
    aggregation = {'Product': 'first', 'delta_time': on_what}
    df = groupby.agg(aggregation)
    return df


mean_delta_time = merge_x_with_product_names(work_table_newest, m_delta_time, 'mean')
mean_delta_time_under_10 = merge_x_with_product_names(work_table_newest, m_delta_time_under_10, 'mean')
median_delta_time = merge_x_with_product_names(work_table_newest, m_delta_time, 'median')
median_delta_time_under_10 = merge_x_with_product_names(work_table_newest, m_delta_time_under_10, 'median')
std_delta_time = merge_x_with_product_names(work_table_newest, m_delta_time, 'std')
std_delta_time_under_10 = merge_x_with_product_names(work_table_newest, m_delta_time_under_10, 'std')


def total_production_time(wt):
    copy = wt.copy()
    copy['time'] = (copy['StopDateTime'] - copy['StartDateTime']).dt.total_seconds()
    groupby = copy.groupby('Product')
    aggregation = {'Product': 'first', 'time': 'sum'}
    df = groupby.agg(aggregation)
    return df


# time length
total_time = total_production_time(work_table_newest)

percentage_of_errors = pd.DataFrame({'Product': error_gonna_happen_again.index,
                  '% of more than one error in a job': error_gonna_happen_again.values,
                  '% of more than one error in a job within 10 min': error_gonna_happen_again_in_10.values,
                  'number of errors': merged_with_dummies[columns].sum(),
                  'number of errors in 10 minutes': under600[columns].sum(),
                  'percentage of errors under 10 minutes': under600[columns].sum() / merged_with_dummies_without_nan[columns].sum() * 100,
                  'job length': total_time['time'].values,
                  'sec/No': total_time['time'].values / merged_with_dummies[columns].sum()
                  })


def merge_mean_data_with_df(df, df_to_merge, column_name):
    df = df.merge(df_to_merge, left_on='Product', right_on='Product').copy()
    df.rename(columns={'delta_time': column_name}, inplace=True)
    return df


percentage_of_errors = merge_mean_data_with_df(percentage_of_errors, mean_delta_time, 'mean_delta_time')
percentage_of_errors = merge_mean_data_with_df(percentage_of_errors, median_delta_time, 'median_delta_time')
percentage_of_errors = merge_mean_data_with_df(percentage_of_errors, std_delta_time, 'std_delta_time')
percentage_of_errors = merge_mean_data_with_df(percentage_of_errors, mean_delta_time_under_10, 'mean_delta_time_under_10')
percentage_of_errors = merge_mean_data_with_df(percentage_of_errors, median_delta_time_under_10, 'median_delta_time_under_10')
percentage_of_errors = merge_mean_data_with_df(percentage_of_errors, std_delta_time_under_10, 'std_delta_time_under_10')


def x_0104_alarm_time(df, on_what):
    copy = df.copy()
    groupby = copy.groupby('Product')
    aggregation = {'Product': 'first', '% 0104': on_what}
    results = groupby.agg(aggregation)
    return results


# 0104 alarm time
grouped_0104_mean = x_0104_alarm_time(stats, 'mean')
grouped_0104_sum = x_0104_alarm_time(stats, 'sum')
percentage_of_errors['% of 0104 Alarm'] = grouped_0104_mean['% 0104'].values.copy()
percentage_of_errors['total 0104 Alarm duration'] = grouped_0104_sum['% 0104'].values.copy()
