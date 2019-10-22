import pandas as pd
import numpy as np
from load_data import load_csv


def clean_work_table(work_table, stats):
    work_table = _remove_complete_duplicates(work_table)
    work_table = _remove_diff_jobref_overlaps(work_table)
    work_table = _remove_complete_overlaps(work_table)
    work_table = _remove_top_heavy_overlaps(work_table, stats)
    work_table = _remove_bottom_heavy_overlaps(work_table, stats)
    work_table = _remove_partial_overlaps(work_table, stats)
    return work_table


def _remove_jobrefs(worktable, data_slice):
    job_refs = data_slice['JOBREF']
    f_job_refs = data_slice['f_JOBREF']
    erroneous_jobs = pd.concat([job_refs, f_job_refs])
    erroneous_jobs = set(erroneous_jobs)
    worktable = worktable.loc[~worktable['JOBREF'].isin(erroneous_jobs), :].reset_index(drop=True)
    return worktable


def _remove_index_or_f_index(worktable, data_slice, condition):
    indices_to_delete = np.where(
        condition,
        data_slice.index,
        data_slice.index + 1
    )
    worktable = worktable.loc[~worktable.index.isin(indices_to_delete)].copy(deep=True)
    return worktable


def _remove_complete_duplicates(work_table):
    work_table = work_table.loc[~work_table[['JOBREF',
                                             'QTYGOOD',
                                             'StartDateTime',
                                             'StopDateTime',
                                             'Seconds',
                                             'NAME']].duplicated(keep='first')]

    work_table = work_table.loc[work_table['WRKCTRID'] == 1405, :]
    return work_table


def _remove_diff_jobref_overlaps(work_table):
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
    return _remove_jobrefs(work_table, diff_jobrefs)


def _remove_complete_overlaps(work_table):
    complete_overlaps = work_table[
        (work_table['StartDateTime'] == work_table['f_StartDateTime'])
        & (work_table['StopDateTime'] == work_table['f_StopDateTime'])
    ]
    condition = work_table.index.isin(complete_overlaps.index)
    work_table = work_table.loc[~condition, :]
    return work_table


def _remove_top_heavy_overlaps(work_table, stats=True):
    # REMOVE TOP HEAVY OVERLAPS WHERE QTYGOOD == 0 AND f_QTYGOOD >= 0
    top_heavy_overlaps = work_table[
        (work_table['StartDateTime'] == work_table['f_StartDateTime'])
        & (work_table['StopDateTime'] < work_table['f_StopDateTime'])
        & (work_table['QTYGOOD'] == 0)
        & (work_table['f_QTYGOOD'] >= 0)
    ]

    condition = work_table.index.isin(top_heavy_overlaps.index)
    work_table = work_table.loc[~condition, :]

    # CHECKS FOR TOP HEAVY OVERLAPS WHERE QTYGOOD >= 0 AND f_QTYGOOD >= 0.
    # IF STATS=TRUE and ANY ARE FOUND, THE JOBREFS CONTAINING THEM ARE DELETED
    top_heavy_overlaps_non_zero = work_table[
        (work_table['StartDateTime'] == work_table['f_StartDateTime'])
        & (work_table['StopDateTime'] <= work_table['f_StopDateTime'])
        & (work_table['QTYGOOD'] != 0)
        & (work_table['f_QTYGOOD'] != 0)
    ]

    if stats:
        if len(top_heavy_overlaps_non_zero.index) != 0:
            work_table = _remove_jobrefs(work_table, top_heavy_overlaps_non_zero)
    else:
        indices = top_heavy_overlaps_non_zero.index
        work_table = work_table[~work_table.index.isin(indices)]
    return work_table


def _remove_n_plus_1_index_and_set_column(work_table, data_slice, columns):
    indices = data_slice.index
    f_indices = indices + 1
    for key in columns.keys():
        column = columns[key]['column']
        f_column = columns[key]['f_column']
        work_table.loc[indices, column] = work_table.loc[indices, f_column]
    work_table = work_table[~work_table.index.isin(f_indices)]
    return work_table


def bottom_heavy_overlaps_check(work_table):
    positive = True
    conditionals = {
        (
            (work_table['QTYGOOD'] != 0), (work_table['f_QTYGOOD'] == 0)
        ),
        (
            (work_table['QTYGOOD'] == 0), (work_table['f_QTYGOOD'] >= 0)
        ),
        (
            (work_table['QTYGOOD'] != 0), (work_table['f_QTYGOOD'] != 0)
        ),
    }
    for i in range(1, 4):
        bottom_heavy_overlaps = work_table[
            (work_table['StartDateTime'] <= work_table['f_StartDateTime'])
            & (work_table['StopDateTime'] >= work_table['f_StopDateTime'])
            & conditionals[i][0]
            & conditionals[i][1]
        ]
        if len(bottom_heavy_overlaps.index) != 0:
            positive = False
    return positive


def _remove_bottom_heavy_overlaps(work_table, stats=True):
    bottom_heavy_overlaps = work_table[
        (work_table['StartDateTime'] <= work_table['f_StartDateTime'])
        & (work_table['StopDateTime'] >= work_table['f_StopDateTime'])
        & (work_table['QTYGOOD'] != 0)
        & (work_table['f_QTYGOOD'] == 0)
    ]
    indices_to_delete = bottom_heavy_overlaps.index + 1
    work_table = work_table[~work_table.index.isin(indices_to_delete)]

    bottom_heavy_overlaps = work_table[
        (work_table['StartDateTime'] <= work_table['f_StartDateTime'])
        & (work_table['StopDateTime'] >= work_table['f_StopDateTime'])
        & (work_table['QTYGOOD'] == 0)
        & (work_table['f_QTYGOOD'] >= 0)
    ]
    work_table = _remove_n_plus_1_index_and_set_column(
        work_table,
        bottom_heavy_overlaps,
        {'payload': {'column': 'QTYGOOD', 'f_column': 'f_QTYGOOD'}, }
    )

    # CHECKS FOR BOTTOM HEAVY OVERLAPS WHERE QTYGOOD != 0 AND f_QTYGOOD != 0.
    # IF ANY ARE FOUND, THE JOBREFS CONTAINING THEM ARE DELETED
    bottom_heavy_overlaps_both_non_zero = work_table[
        (work_table['StartDateTime'] <= work_table['f_StartDateTime'])
        & (work_table['StopDateTime'] >= work_table['f_StopDateTime'])
        & (work_table['QTYGOOD'] != 0)
        & (work_table['f_QTYGOOD'] != 0)
    ]

    if stats:
        if len(bottom_heavy_overlaps_both_non_zero.index) != 0:
            work_table = _remove_jobrefs(work_table, bottom_heavy_overlaps_both_non_zero)
    else:
        indices_to_delete = bottom_heavy_overlaps_both_non_zero.index + 1
        work_table = work_table[~work_table.index.isin(indices_to_delete)]

    return work_table


def _apply_condition_partial(work_table, partial_overlaps, payload_dict):
    work_table = _remove_n_plus_1_index_and_set_column(
        work_table,
        partial_overlaps,
        payload_dict
    )
    return work_table


def _remove_partial_overlaps(work_table, stats=True):
    """STILL NEED TO DECIDE WHAT TO DO"""
    payload_1 = {'column': 'QTYGOOD', 'f_column': 'f_QTYGOOD'}
    payload_2 = {'column': 'StopDateTime', 'f_column': 'f_StopDateTime'}

    both = {'payload_1': payload_1, 'payload_2': payload_2,}
    only_2 = {'payload_2': payload_2, }

    condition = (work_table['f_StopDateTime'] > work_table['StopDateTime']) \
                & (work_table['StopDateTime'] > work_table['f_StartDateTime']) \
                & (work_table['f_StartDateTime'] > work_table['StartDateTime']) \
                & (work_table['QTYGOOD'] == 0) \
                & (work_table['f_QTYGOOD'] != 0)
    partial_overlaps = work_table.loc[condition]
    work_table = _apply_condition_partial(work_table, partial_overlaps, both)

    condition = (work_table['f_StopDateTime'] > work_table['StopDateTime']) \
                & (work_table['StopDateTime'] > work_table['f_StartDateTime']) \
                & (work_table['f_StartDateTime'] > work_table['StartDateTime']) \
                & (work_table['QTYGOOD'] != 0) \
                & (work_table['f_QTYGOOD'] == 0)
    partial_overlaps = work_table.loc[condition]
    work_table = _apply_condition_partial(work_table, partial_overlaps, both)

    condition = (work_table['f_StopDateTime'] > work_table['StopDateTime']) \
                & (work_table['StopDateTime'] > work_table['f_StartDateTime']) \
                & (work_table['f_StartDateTime'] > work_table['StartDateTime']) \
                & (work_table['QTYGOOD'] != 0) \
                & (work_table['f_QTYGOOD'] != 0)
    partial_overlaps = work_table.loc[condition]
    if stats:
        work_table = _remove_jobrefs(work_table, partial_overlaps)
    else:
        work_table = _apply_condition_partial(work_table, partial_overlaps, only_2)
    return work_table


def fix_work_table(breaks, original_work_table, sensor_data, stats):
    work_table = original_work_table.copy(deep=True)
    condition = work_table['JOBNUM'].isin(breaks['JOBNUM'])
    work_table_breaks = work_table.loc[condition].copy(deep=True)
    work_table_breaks = work_table_breaks.sort_values('JOBNUM').reset_index(drop=True)
    breaks = breaks.sort_values('JOBNUM').reset_index(drop=True)

    duplicates = breaks['JOBNUM'].duplicated(keep='first').copy(deep=True)
    duplicates = breaks.loc[duplicates]

    work_table_breaks = work_table_breaks[~work_table_breaks['JOBNUM'].isin(duplicates['JOBNUM'])]
    breaks = breaks[~breaks['JOBNUM'].isin(duplicates['JOBNUM'])]

    condition = \
        (breaks['Indgang 0101'] == 1) \
        & (breaks['next_0101'] == 1) \

    multi_row_breaks = breaks.loc[condition].copy(deep=True)
    single_row_breaks = breaks.loc[~breaks['JOBNUM'].isin(multi_row_breaks['JOBNUM'])].copy(deep=True)
    work_table_singles = work_table_breaks.loc[~work_table_breaks['JOBNUM'].isin(multi_row_breaks['JOBNUM'])].copy(deep=True)
    work_table = work_table.loc[~work_table['JOBNUM'].isin(work_table_singles['JOBNUM'])].copy(deep=True)
    single_row_breaks_fixed = _fix_breaks(work_table_singles.copy(), single_row_breaks['Date'], single_row_breaks['next_Date'])

    single_row_breaks_fixed = single_row_breaks_fixed.loc[single_row_breaks_fixed['Seconds'] > 0]
    work_table = pd.concat([work_table, single_row_breaks_fixed], axis=0)

    condition = sensor_data['0101 Group'].isin(multi_row_breaks['0101 Group'])
    multi_row_breaks = sensor_data[condition].copy()
    before_break_finish = get_date(multi_row_breaks.copy(), 'Date', 'last')
    after_break_start = get_date(multi_row_breaks.copy(), 'next_Date', 'last')

    work_table_multis = work_table_breaks.loc[work_table_breaks['JOBNUM'].isin(multi_row_breaks['JOBNUM'])]
    work_table = work_table.loc[~work_table['JOBNUM'].isin(work_table_multis['JOBNUM'])]
    multi_row_breaks_fixed = _fix_breaks(work_table_multis, before_break_finish, after_break_start)

    multi_row_breaks_fixed = multi_row_breaks_fixed.loc[multi_row_breaks_fixed['Seconds'] > 0]
    work_table = pd.concat([work_table, multi_row_breaks_fixed])
    work_table = work_table.sort_values('StartDateTime').reset_index(drop=True)

    work_table = clean_work_table(work_table, stats=stats)
    work_table = create_jobnums(work_table)
    return work_table, multi_row_breaks_fixed, single_row_breaks_fixed


def prep_work_table(csv, columns, stats):
    work_table_original = load_csv.work_table(csv, columns)
    work_table = work_table_original.copy(deep=True)
    count = 0
    while True:
        if count < 10:
            work_table = prepare_work_table(work_table)
            work_table = clean_work_table(work_table, stats=stats)
            work_table = filter_work_table(work_table, reg_ex=r'^SW|^CF')

            work_table.reset_index(drop=True, inplace=True)
            work_table.drop(['f_StopDateTime', 'f_StartDateTime'], axis=1, inplace=True)
            work_table[['f_StopDateTime', 'f_StartDateTime']] = work_table[['StopDateTime', 'StartDateTime']].shift(-1)

            condition = (work_table['StopDateTime'] == work_table['f_StopDateTime'])
            wrong = work_table.loc[condition]
            count += 1
            if len(wrong) == 0:
                break
        else:
            raise Exception('Problem With work_table cleaning')
    return work_table


def get_date(multi_row_breaks, column, function):
    agg_dict = {column: function}
    dates = multi_row_breaks.groupby('0101 Group').agg(agg_dict)
    return dates


def _fix_breaks(work_table, first, second):
    before_break = first.reset_index(drop=True).copy()
    after_break = second.reset_index(drop=True).copy()
    work_table = work_table.sort_values('JOBNUM').reset_index(drop=True)
    starts = work_table.copy()
    stops = work_table.copy()

    starts['StopDateTime'] = before_break
    stops['StartDateTime'] = after_break
    stops['QTYGOOD'] = 0

    merged = pd.concat([starts, stops])
    merged.sort_values('StartDateTime', inplace=True)

    # Recalculate seconds
    merged['Seconds'] = merged['StopDateTime'] - merged['StartDateTime']
    merged['Seconds'] = merged['Seconds'].dt.total_seconds()
    return merged


def create_jobnums(work_table):
    work_table['JOBNUM'] = np.arange(1, len(work_table.index) + 1)
    return work_table


def unique_error_count(data_slice, worktable):
    index_current = pd.Series(data_slice.index)
    index_next = pd.Series(data_slice.index + 1)
    indices = pd.concat([index_current, index_next])

    count_error = len(set(indices))
    count_unique_jobrefs = len(set(worktable['JOBREF']))
    return count_error, count_unique_jobrefs


def prepare_work_table(work_table):
    work_table.sort_values('StartDateTime', inplace=True)
    work_table = work_table.loc[work_table['WRKCTRID'] == 1405, :]
    work_table = work_table.reset_index(drop=True)
    next_columns = ['f_StartDateTime', 'f_StopDateTime', 'f_QTYGOOD', 'f_JOBREF', 'f_NAME']
    original_columns = ['StartDateTime', 'StopDateTime', 'QTYGOOD', 'JOBREF', 'NAME']
    work_table[next_columns] = work_table[original_columns].shift(-1)
    return work_table


def _filter_work_table_by_work_id(work_table, reg_ex):
    return work_table.loc[work_table['NAME'].str.contains(reg_ex, regex=True), :]


def filter_work_table_by_work_id(work_table, reg_ex):
    return work_table.loc[work_table['NAME'].str.contains(reg_ex, regex=True), :]


def filter_work_table(work_table, reg_ex):
    filtered = _filter_work_table_by_work_id(work_table, reg_ex)
    filtered = filtered.loc[~filtered.index.duplicated(keep='first')]
    filtered['JOBNUM'] = np.arange(len(filtered.index))
    return filtered
