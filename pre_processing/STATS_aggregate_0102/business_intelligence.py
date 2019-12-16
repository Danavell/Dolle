import numpy as np

from pre_processing.utils import pie_chart


def _calc_percentages(agg):
    return agg.iloc[:, 1] / np.sum(agg.iloc[:, 1])


def pie_chart_num_spikes_count(agg, roll=30):
    col = '0102 Sum Pace >= 25'
    col_count = '0102 Pace >= 25 Count'

    total_spikes = np.sum(agg[col_count])

    deacs = agg[agg['Label'] == 1]
    deacs_count = deacs.groupby(col) \
                       .count() \
                       .iloc[:, 1] \
                       .reset_index() \
                       .copy()

    non_deacs_count = agg.groupby('JOBNUM')[col_count] \
                         .rolling(roll) \
                         .sum() \
                         .fillna(0) \
                         .copy()

    non_deacs_count.index = agg.index
    agg.loc[:, 'count'] = non_deacs_count
    non_deacs_count = agg.groupby('count') \
                         .count() \
                         .iloc[:, 1] \
                         .reset_index() \
                         .copy()

    d_len = len(deacs_count)
    corrected = non_deacs_count.iloc[:d_len, 1] - deacs_count.iloc[:, 1]
    non_deacs_count.iloc[:d_len, 1] = corrected

    percs = _calc_percentages(deacs_count)
    deacs_count.loc[:, 'percentages'] = percs

    title = f'Number of spikes in previous {roll} String In paces'
    pie_chart(
        deacs_count.iloc[:, 2], deacs_count.iloc[:, 0],
        f'Deactivations: {title}'
    )

    percs = _calc_percentages(non_deacs_count)
    non_deacs_count.loc[:, 'percentages'] = percs
    pie_chart(
        non_deacs_count.iloc[:, 2], non_deacs_count.iloc[:, 0],
        f'Non Deactivations: {title}'
    )

    num_spikes_deacs = deacs_count.iloc[:, 0] * deacs_count.iloc[:, 1]
    total_num_spikes_deacs = int(np.sum(num_spikes_deacs))

    return agg, deacs_count, non_deacs_count, total_spikes, \
           num_spikes_deacs, total_num_spikes_deacs, \
           (total_num_spikes_deacs / total_spikes) * 100