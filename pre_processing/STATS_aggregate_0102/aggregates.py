import numpy as np
import pandas as pd


def confusion_matrix(agg, n=25, train=True):
    columns = ['0102 Pace', 'Label', '0102 ID']
    agg[[f'next_{column}' for column in columns]] = agg.groupby('JOBNUM')[columns].shift(-1)

    """
    Some Deactivations spill into the next 0102 ID. The following two conditions remove
    these rows to prevent them distorting the confusion matrix
    """
    over_flow_condition = (agg['Label'] == 1) & \
                          (agg['0101 Duration'] + agg['Time Delta'] > agg['0102 Pace']) & \
                          (agg['next_Label'] == 0) & \
                          (agg['next_0102 Pace'] >= n)

    false_neg_condition = (agg['0102 Pace'] >= n) & \
                          (agg['Label'] == 0) & \
                          (agg['0102 ID'].isin(agg.loc[~over_flow_condition, 'next_0102 ID']))

    true_pos = len(agg.loc[agg.loc[:, 'Label'] == 0].index)
    false_neg = len(agg[false_neg_condition].index)
    true_neg = len(agg.loc[(agg.loc[:, 'Time Delta'] >= n) & (agg.loc[:, 'Label'] == 1)].index)
    false_pos = len(agg.loc[(agg.loc[:, 'Time Delta'] < n) & (agg.loc[:, 'Label'] == 1)].index)
    if train:
        true_pos = round(true_pos / 5)
        false_neg = round(false_neg / 5)
        false_pos = round(false_pos / 5)
        true_neg = round(true_neg / 5)
    return np.array(
        [[true_pos, false_neg],
         [false_pos, true_neg]]
    )


def corr(percentiles):
    a = percentiles[['ND: 0102 Pace']]
    a['Label'] = 0
    a.columns = ['Time', 'Label']

    b = percentiles[['D: time delta']]
    b['Label'] = 1
    b.columns = ['Time', 'Label']
    return pd.concat([a, b], axis=0, sort=False).corr().iloc[1, 0]
