import numpy as np
import pandas as pd


def calc_time_since_string_in_and_deactivation(sensor_data, aggs):
    """
    Returns a dataframe containing 0102 ID, the time of each string in and deactivation
    as well as the time delta between them
    """
    condition = sensor_data.loc[:, 'Non Duplicate 0101'] == 1
    agg_condition = aggs.loc[:, 'Non Duplicate 0101'] == 1
    stats_1 = _calc_time_since_string_in_and_deactivation(
        sensor_data, aggs, condition, agg_condition
    )

    condition = sensor_data.loc[:, 'Non Duplicate 0101'] == 1
    agg_condition = aggs.loc[:, 'Non Duplicate 0101'] > 1
    stats_2 = _calc_time_since_string_in_and_deactivation(
        sensor_data, aggs, condition, agg_condition, multi=True
    )
    return pd.concat([stats_1, stats_2], axis=0, sort=False)


def _calc_time_since_string_in_and_deactivation(sensor_data, aggs, condition, agg_condition, multi=False):
    sd_deacs = sensor_data.loc[condition, ['Date', '0102 ID']].copy().dropna()
    agg_deacs = aggs.loc[
        agg_condition, ['Date', '0102 ID', 'Non Duplicate 0101']
    ]
    condition = sd_deacs['0102 ID'].isin(agg_deacs['0102 ID'])
    sd_deacs_filtered = sd_deacs.loc[condition, :]
    merged = pd.merge(
        left=sd_deacs_filtered, right=agg_deacs, how='left', left_on='0102 ID', right_on='0102 ID'
    )
    merged.loc[:, 'Time Delta'] = (merged.iloc[:, 0] - merged.iloc[:, 2])\
        .dt.total_seconds()\
        .astype(int)
    merged.loc[:, 'multi'] = 1 if multi else 0
    return merged.loc[:, ['0102 ID', 'Date_x', 'Time Delta', 'multi']]


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
    aggs_multis = time_delta_aggs.loc[time_delta_aggs.loc[:, 'multi'] == 1].copy()
    condition = aggs['0102 ID'].isin(aggs_multis['0102 ID'])
    aggs_ids_like_multis = aggs.loc[condition, :]
    aggs_multis = pd.merge(
        left=aggs_multis, right=aggs_ids_like_multis, how='left',
        left_on='0102 ID', right_on='0102 ID'
    )
    aggs_multis.loc[:, '0101 Duration'] = (
        aggs_multis.loc[:, 'Time of Deactivation'] - aggs_multis.loc[:, 'Time of String in']
    ).dt.total_seconds().fillna(0).astype(int)

    aggs_concat = pd.concat([aggs_multis, aggs_singles], axis=0, sort=True)
    aggs_concat = aggs_concat\
        .rename(columns={'Date_x': 'Time of Deactivation', 'Date': 'Time of String in'})\
        .sort_values(['0102 ID', 'Time of Deactivation'])

    return aggs_concat, aggs_singles, aggs_multis


def calc_0103_Pace_and_string_deac_t_delta_percentiles(aggs, time_delta_aggs):
    """
    Calculates the percentiles from 0 to 99 for both 0102 Pace and time delta
    between string in time and the time the deactivation occurred.
    """
    percentile_ticks = [i for i in range(1, 100)]
    percentiles = np.percentile(
        time_delta_aggs.loc[:, 'Time Delta'], percentile_ticks
    )
    indices = [f'{i}%' for i in range(1, 100)]
    percentiles = pd.DataFrame(percentiles.T, index=indices)

    percentiles_2 = np.percentile(
        aggs.loc[aggs.loc[:, 'Label'] == 0, '0102 Pace'], percentile_ticks
    )
    percentiles_2 = pd.DataFrame(percentiles_2.T, index=indices)
    columns = ['0102 Pace', 'Time between string in and alarm']
    output = pd.concat([percentiles_2, percentiles], axis=1)
    output.columns = columns
    return output


