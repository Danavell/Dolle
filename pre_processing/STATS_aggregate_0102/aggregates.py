import numpy as np
import pandas as pd


def calc_t_delta_and_merge(deacs_sd, agg, condition, multi=False):
    """
    Calculates the time between when a string enters a machine then concats
    the time vector for all deacs with the agg data
    """
    data = agg.loc[condition, :]
    time_delta_aggs = calc_time_since_string_in_and_deactivation(
        deacs_sd, data
    )
    """
    If the agg data contains rows with only 1 deac per 0102 ID then it doesn't
    matter which dataframe is on the 'left'. This is not true when the 0102 ID contains
    multiple deactivations. In that case the time deltas, which contain all the 
    deactivations in an 0102 ID, must be on the left
    """
    left = time_delta_aggs if multi else data
    right = data if multi else time_delta_aggs
    return pd.merge(left=left, right=right, left_on='0102 ID', right_on='0102 ID')


def calc_time_since_string_in_and_deactivation(sd_deacs, agg_deacs):
    """
    Returns a dataframe containing 0102 ID, the time of each string in and deactivation
    as well as the time delta between them
    """
    agg_deacs = agg_deacs.loc[:, ['Date', '0102 ID', 'Non Duplicate 0101']].copy()
    merged = pd.merge(
        left=agg_deacs, right=sd_deacs, how='left', left_on='0102 ID', right_on='0102 ID'
    )
    merged.loc[:, 'Time Delta'] = (merged.loc[:, 'Date_y'] - merged.loc[:, 'Date_x']) \
        .dt.total_seconds() \
        .astype(int)
    return merged.loc[:, ['0102 ID', 'Date_y', 'Time Delta']]


def _rename_dates(data):
    return data.rename(columns={'Date_x': 'Time of Deactivation', 'Date': 'Time of String in'}) \
               .sort_values(['0102 ID', 'Time of Deactivation'])


def add_unique_deactivations_to_0102_IDs(aggs, time_delta_aggs):
    """
    Some 0102 aggregates contain more than 1 deactivation. This function does a left join of the
    unique deactivation IDs to 0102 aggregates. The result is duplicated 0102 aggregates with
    different unique deactivations and time deltas between the string in time and deactivation
    """
    aggs_singles = time_delta_aggs.loc[time_delta_aggs.loc[:, 'multi'] == 0].copy()
    aggs_singles = pd.merge(
        left=aggs, right=aggs_singles, how='left',
        left_on='0102 ID', right_on='0102 ID'
    )
    aggs_singles = _rename_dates(aggs_singles)

    aggs_multis = time_delta_aggs.loc[time_delta_aggs.loc[:, 'multi'] == 1].copy()
    condition = aggs['0102 ID'].isin(aggs_multis['0102 ID'])
    aggs_ids_like_multis = aggs.loc[condition, :]
    aggs_multis = pd.merge(
        left=aggs_multis, right=aggs_ids_like_multis, how='left',
        left_on='0102 ID', right_on='0102 ID'
    )
    aggs_multis = _rename_dates(aggs_multis)
    aggs_multis.loc[:, '0101 Duration'] = (
        aggs_multis.loc[:, 'Time of Deactivation'] - aggs_multis.loc[:, 'Time of String in']
    ).dt.total_seconds().fillna(0).astype(int)

    aggs_concat = pd.concat([aggs_multis, aggs_singles], axis=0, sort=True)
    return aggs_concat, aggs_singles, aggs_multis


def confusion_matrix(agg):
    true_pos = len(agg.loc[agg.loc[:, 'Label'] == 0].index)
    false_neg = len(agg.loc[(agg.loc[:, '0102 Pace'] >= 25) & (agg.loc[:, 'Label'] == 0)].index)
    true_neg = len(agg.loc[(agg.loc[:, 'Time Delta'] >= 25) & (agg.loc[:, 'Label'] == 1)].index)
    false_pos = len(agg.loc[(agg.loc[:, 'Time Delta'] < 25) & (agg.loc[:, 'Label'] == 1)].index)
    return np.array(
        [[true_pos, false_neg],
         [false_pos, true_neg]]
    )


def corr(percentiles):
    a = percentiles[['Non-Deactivations: 0102 Pace']]
    a['Label'] = 0
    a.columns = ['Time', 'Label']

    b = percentiles[['Deactivations: time delta']]
    b['Label'] = 1
    b.columns = ['Time', 'Label']
    return pd.concat([a, b], axis=0, sort=False).corr().iloc[1, 0]
