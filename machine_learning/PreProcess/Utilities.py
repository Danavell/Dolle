import numpy as np
import pandas as pd

import sklearn
from sklearn.preprocessing import MinMaxScaler

from machine_learning.STATS import add_label_ids


def adjust_inputs(num_rows, skip):
    skip -= 1
    first_row = num_rows + skip
    return first_row, skip


def adjust_inputs_catch(num_rows, skip, catch):
    skip -= 1
    first_row = num_rows + skip + catch - 1
    return first_row, skip


def create_lags(agg, lags, pred):
    output = pd.DataFrame()
    for lag in range(lags, pred, -1):
        for column in agg.columns:
            if column != 'JOBNUM' and column != 'Label':
                if lag == 0:
                    output.loc[:, column] = agg.loc[:, column]
                else:
                    output.loc[:, f'{column} {lag} lagged'] = agg[column].shift(lag)
    output.loc[:, 'Label'] = agg.loc[:, 'Label']
    return output


def create_lags_catch(agg, lags):
    output = pd.DataFrame()
    for lag in range(lags, -1, -1):
        for column in agg.columns:
            if column != 'JOBNUM':
                if lag == 0:
                    output.loc[:, column] = agg.loc[:, column]
                else:
                    output.loc[:, f'{column} {lag} lagged'] = agg[column].shift(lag)
    return output


def flatten(agg, first_row, pred, catch=False):
    groupby = agg.groupby('JOBNUM')
    if catch:
        agg = groupby.apply(create_lags_catch, first_row)
    else:
        agg = groupby.apply(create_lags, first_row, pred)
    agg = agg.reset_index(drop=True).dropna()
    return agg


def scale(agg, scaler=None):
    if not isinstance(scaler, MinMaxScaler):
        scaler = MinMaxScaler()
    scaled = scaler.fit_transform(agg)
    agg = pd.DataFrame(data=scaled, columns=agg.columns, index=agg.index)
    return agg, scaler


def split_deacs_and_non_deacs(agg, condition):
    deacs = agg.loc[condition, :]
    non_deacs = agg.loc[~condition, :]
    return deacs, non_deacs


def drop_original_labels(deacs, non_deacs):
    non_deacs = non_deacs.drop(['JOBNUM', 'original labels', 'label group'], axis=1)
    label_group = deacs.loc[:, 'label group']
    deacs = deacs.drop(['JOBNUM', 'original labels', 'label group'], axis=1)
    return deacs, non_deacs, label_group


def drop_original_labels_v2(data, condition):
    deacs, non_deacs = split_deacs_and_non_deacs(data, condition)
    label_group = deacs.loc[:, 'label group']
    data = data.drop(['JOBNUM', 'original labels', 'label group'], axis=1)
    return data, label_group


def separate_labels(deacs, non_deacs):
    deacs_labels = deacs.loc[:, 'Label']
    non_deacs_labels = non_deacs.loc[:, 'Label']
    return deacs_labels, non_deacs_labels


def cols_to_drop(catch, num_columns, skip, deacs, non_deacs):
    deacs, non_deacs = _drop_cols(catch, num_columns, deacs, non_deacs)

    if skip > 0:
        deacs, non_deacs = _drop_cols(skip, num_columns, deacs, non_deacs)

    cols = deacs.columns
    condition = cols.str.contains('Label')
    cols_to_drop = cols[condition]
    deacs = deacs.drop(cols_to_drop, axis=1)
    non_deacs = non_deacs.drop(cols_to_drop, axis=1)
    return deacs, non_deacs


def cols_to_drop_v2(catch, num_columns, skip, data):
    data = _drop_cols_v2(catch, num_columns, data)

    if skip > 0:
        data = _drop_cols_v2(skip, num_columns, data)

    cols = data.columns
    condition = cols.str.contains('Label')
    return data.drop(cols[condition], axis=1)


def _drop_cols_v2(multiplier, num_columns, data):
    cols_to_drop = multiplier * num_columns
    return data.iloc[:, :-cols_to_drop]


def _drop_cols(multiplier, num_columns, deacs, non_deacs):
    cols_to_drop = multiplier * num_columns
    deacs = deacs.iloc[:, :-cols_to_drop]
    non_deacs = non_deacs.iloc[:, :-cols_to_drop]
    return deacs, non_deacs


def add_labels(skip, catch, agg):
    label = skip + catch
    agg['True Label'] = 0
    for i in range(1, skip + catch + 1):
        if i == 1:
            condition = agg.loc[:, 'Label'] == 1
            agg.loc[condition, 'True Label'] = label
        else:
            label -= 1
            condition = agg.loc[:, f'Label {i-1} lagged'] == 1
            indices = agg[condition].index
            agg.loc[indices, 'True Label'] = label
    return agg


def deacs_tts(deacs, split, meta, start_id=None):
    condition = deacs.loc[:, 'label'] == 1
    sliced = deacs.loc[condition, :]
    total_num_deacs = len(sliced)
    sliced.loc[:, 'Deacs ID'] = np.arange(total_num_deacs)

    deacs.loc[:, 'Deacs ID'] = sliced.loc[:, 'Deacs ID']
    deacs.loc[:, 'Deacs ID'] = deacs.loc[:, 'Deacs ID'].fillna(method='ffill')
    num_deacs_test = round((1 - split) * total_num_deacs)

    if not isinstance(start_id, int):
        start_id = np.random.randint(0, total_num_deacs - num_deacs_test)

    end_id = start_id + num_deacs_test
    ids = np.arange(start_id, end_id)
    condition = deacs['Deacs ID'].isin(ids)
    deacs = deacs.drop('Deacs ID', axis=1)
    deacs_test = deacs.loc[condition, :]
    deacs_train = deacs.loc[~condition, :]

    meta['deac start id'] = start_id
    meta['deac end id'] = end_id
    meta['total num deacs'] = total_num_deacs
    meta['X_test num deacs'] = num_deacs_test
    return deacs_train, deacs_test, meta


def add_unique_deac_ids(agg, meta):
    jobnums = meta['JOBNUMs']
    condition = jobnums.index.isin(agg.index)
    agg['JOBNUM'] = jobnums.loc[condition, :]
    agg['prev_label'] = agg.groupby('JOBNUM')['True Label'].shift(1)
    return add_label_ids(agg, 'True Label')


def split_deacs(deacs, split, meta, true_labels=False, start_id=None):
    ids = set(deacs.loc[:, 'label group'])
    total_num_deacs = len(ids)
    num_deacs_test = round((1 - split) * total_num_deacs)

    if not isinstance(start_id, int):
        start_id = np.random.randint(0, total_num_deacs - num_deacs_test)

    end_id = start_id + num_deacs_test
    ids = np.arange(start_id, end_id)
    condition = deacs['label group'].isin(ids)
    deacs = deacs.drop('label group', axis=1)
    deacs_test = deacs.loc[condition, :]
    deacs_train = deacs.loc[~condition, :]
    if true_labels:
        deacs_train = deacs_train.drop('True Labels', axis=1)

    meta['deac start id'] = start_id
    meta['deac end id'] = end_id
    meta['total num deacs'] = total_num_deacs
    meta['X_test num deacs'] = num_deacs_test
    return deacs_train, deacs_test, meta


def return_test_indices_range(deacs_test):
    start_index = deacs_test.index[0]
    end_index = deacs_test.index[-1]
    return np.arange(start_index, end_index + 1)


def return_category(data, category):
    condition = data.loc[:, 'Label'] == category
    return data.loc[condition, :]


def resize_concat_deacs(zeros, data, category_num):
    category = return_category(data, category_num)
    diff = len(zeros.index) // len(category.index)
    final_diff = len(zeros.index) - (len(category.index) * diff)
    category_copy = category.copy()
    for _ in range(1, diff):
        category = pd.concat((category, category_copy), axis=0)
    category_copy = category_copy.reset_index(drop=True)
    indices = pd.Series(category_copy.index)
    rnd_indices = indices.sample(n=final_diff, replace=False, random_state=42)
    return pd.concat((category, category_copy.loc[rnd_indices, :]), axis=0)


# def return_catch_condition(agg, catch):
#     if catch == 1:
#         return agg.loc[:, 'Label'] == 1
#     elif catch == 2:
#         return (agg.loc[:, 'Label'] == 1) | \
#                (agg.loc[:, 'Label 1 lagged'] == 1)
#     elif catch == 3:
#         return (agg.loc[:, 'Label'] == 1) | \
#                (agg.loc[:, 'Label 1 lagged'] == 1) | \
#                (agg.loc[:, 'Label 2 lagged'] == 1)
#     elif catch == 4:
#         return (agg.loc[:, 'Label'] == 1) | \
#                (agg.loc[:, 'Label 1 lagged'] == 1) | \
#                (agg.loc[:, 'Label 2 lagged'] == 1) | \
#                (agg.loc[:, 'Label 3 lagged'] == 1)
#     elif catch == 5:
#         return (agg.loc[:, 'Label'] == 1) | \
#                (agg.loc[:, 'Label 1 lagged'] == 1) | \
#                (agg.loc[:, 'Label 2 lagged'] == 1) | \
#                (agg.loc[:, 'Label 3 lagged'] == 1) | \
#                (agg.loc[:, 'Label 4 lagged'] == 1)
#     elif catch == 6:
#         return (agg.loc[:, 'Label'] == 1) | \
#                (agg.loc[:, 'Label 1 lagged'] == 1) | \
#                (agg.loc[:, 'Label 2 lagged'] == 1) | \
#                (agg.loc[:, 'Label 3 lagged'] == 1) | \
#                (agg.loc[:, 'Label 4 lagged'] == 1) | \
#                (agg.loc[:, 'Label 5 lagged'] == 1)
#     elif catch == 7:
#         return (agg.loc[:, 'Label'] == 1) | \
#                (agg.loc[:, 'Label 1 lagged'] == 1) | \
#                (agg.loc[:, 'Label 2 lagged'] == 1) | \
#                (agg.loc[:, 'Label 3 lagged'] == 1) | \
#                (agg.loc[:, 'Label 4 lagged'] == 1) | \
#                (agg.loc[:, 'Label 5 lagged'] == 1) | \
#                (agg.loc[:, 'Label 6 lagged'] == 1)
#     elif catch == 8:
#         return (agg.loc[:, 'Label'] == 1) | \
#                (agg.loc[:, 'Label 1 lagged'] == 1) | \
#                (agg.loc[:, 'Label 2 lagged'] == 1) | \
#                (agg.loc[:, 'Label 3 lagged'] == 1) | \
#                (agg.loc[:, 'Label 4 lagged'] == 1) | \
#                (agg.loc[:, 'Label 5 lagged'] == 1) | \
#                (agg.loc[:, 'Label 6 lagged'] == 1) | \
#                (agg.loc[:, 'Label 7 lagged'] == 1)
#
#     elif catch == 9:
#         return (agg.loc[:, 'Label'] == 1) | \
#                (agg.loc[:, 'Label 1 lagged'] == 1) | \
#                (agg.loc[:, 'Label 2 lagged'] == 1) | \
#                (agg.loc[:, 'Label 3 lagged'] == 1) | \
#                (agg.loc[:, 'Label 4 lagged'] == 1) | \
#                (agg.loc[:, 'Label 5 lagged'] == 1) | \
#                (agg.loc[:, 'Label 6 lagged'] == 1) | \
#                (agg.loc[:, 'Label 7 lagged'] == 1) | \
#                (agg.loc[:, 'Label 8 lagged'] == 1)
#
#     elif catch == 10:
#         return (agg.loc[:, 'Label'] == 1) | \
#                (agg.loc[:, 'Label 1 lagged'] == 1) | \
#                (agg.loc[:, 'Label 2 lagged'] == 1) | \
#                (agg.loc[:, 'Label 3 lagged'] == 1) | \
#                (agg.loc[:, 'Label 4 lagged'] == 1) | \
#                (agg.loc[:, 'Label 5 lagged'] == 1) | \
#                (agg.loc[:, 'Label 6 lagged'] == 1) | \
#                (agg.loc[:, 'Label 7 lagged'] == 1) | \
#                (agg.loc[:, 'Label 8 lagged'] == 1) | \
#                (agg.loc[:, 'Label 9 lagged'] == 1)
#
#     elif catch == 11:
#         return (agg.loc[:, 'Label'] == 1) | \
#                (agg.loc[:, 'Label 1 lagged'] == 1) | \
#                (agg.loc[:, 'Label 2 lagged'] == 1) | \
#                (agg.loc[:, 'Label 3 lagged'] == 1) | \
#                (agg.loc[:, 'Label 4 lagged'] == 1) | \
#                (agg.loc[:, 'Label 5 lagged'] == 1) | \
#                (agg.loc[:, 'Label 6 lagged'] == 1) | \
#                (agg.loc[:, 'Label 7 lagged'] == 1) | \
#                (agg.loc[:, 'Label 8 lagged'] == 1) | \
#                (agg.loc[:, 'Label 9 lagged'] == 1) | \
#                (agg.loc[:, 'Label 10 lagged'] == 1)
#
#     elif catch == 12:
#         return (agg.loc[:, 'Label'] == 1) | \
#                (agg.loc[:, 'Label 1 lagged'] == 1) | \
#                (agg.loc[:, 'Label 2 lagged'] == 1) | \
#                (agg.loc[:, 'Label 3 lagged'] == 1) | \
#                (agg.loc[:, 'Label 4 lagged'] == 1) | \
#                (agg.loc[:, 'Label 5 lagged'] == 1) | \
#                (agg.loc[:, 'Label 6 lagged'] == 1) | \
#                (agg.loc[:, 'Label 7 lagged'] == 1) | \
#                (agg.loc[:, 'Label 8 lagged'] == 1) | \
#                (agg.loc[:, 'Label 9 lagged'] == 1) | \
#                (agg.loc[:, 'Label 10 lagged'] == 1) | \
#                (agg.loc[:, 'Label 11 lagged'] == 1)
#
#     else:
#         raise Exception("Algorithm doesn't support catch > 12")


def catch_first(agg, skip):
    if skip == 0:
        return agg.loc[:, 'Label'] == 1
    if skip > 0:
        return agg.loc[:, f'Label {skip} lagged'] == 1


def catch_n(agg, skip, n):
    return agg.loc[:, f'Label {skip + n} lagged'] == 1


def return_catch_condition(agg, skip, catch):
    if catch == 1:
        return catch_first(agg, skip)

    elif catch == 2:
        return (catch_first(agg, skip)) | \
               (catch_n(agg, skip, 1))

    elif catch == 3:
        return (catch_first(agg, skip)) | \
               (catch_n(agg, skip, 1)) | \
               (catch_n(agg, skip, 2))

    elif catch == 4:
        return (catch_first(agg, skip)) | \
               (catch_n(agg, skip, 1)) | \
               (catch_n(agg, skip, 2)) | \
               (catch_n(agg, skip, 3))

    elif catch == 5:
        return (catch_first(agg, skip)) | \
               (catch_n(agg, skip, 1)) | \
               (catch_n(agg, skip, 2)) | \
               (catch_n(agg, skip, 3)) | \
               (catch_n(agg, skip, 4))

    elif catch == 6:
        return (catch_first(agg, skip)) | \
               (catch_n(agg, skip, 1)) | \
               (catch_n(agg, skip, 2)) | \
               (catch_n(agg, skip, 3)) | \
               (catch_n(agg, skip, 4)) | \
               (catch_n(agg, skip, 5))

    elif catch == 7:
        return (catch_first(agg, skip)) | \
               (catch_n(agg, skip, 1)) | \
               (catch_n(agg, skip, 2)) | \
               (catch_n(agg, skip, 3)) | \
               (catch_n(agg, skip, 4)) | \
               (catch_n(agg, skip, 5)) | \
               (catch_n(agg, skip, 6))

    elif catch == 8:
        return (catch_first(agg, skip)) | \
               (catch_n(agg, skip, 1)) | \
               (catch_n(agg, skip, 2)) | \
               (catch_n(agg, skip, 3)) | \
               (catch_n(agg, skip, 4)) | \
               (catch_n(agg, skip, 5)) | \
               (catch_n(agg, skip, 6)) | \
               (catch_n(agg, skip, 7))

    elif catch == 9:
        return (catch_first(agg, skip)) | \
               (catch_n(agg, skip, 1)) | \
               (catch_n(agg, skip, 2)) | \
               (catch_n(agg, skip, 3)) | \
               (catch_n(agg, skip, 4)) | \
               (catch_n(agg, skip, 5)) | \
               (catch_n(agg, skip, 6)) | \
               (catch_n(agg, skip, 7)) | \
               (catch_n(agg, skip, 8))

    elif catch == 10:
        return (catch_first(agg, skip)) | \
               (catch_n(agg, skip, 1)) | \
               (catch_n(agg, skip, 2)) | \
               (catch_n(agg, skip, 3)) | \
               (catch_n(agg, skip, 4)) | \
               (catch_n(agg, skip, 5)) | \
               (catch_n(agg, skip, 6)) | \
               (catch_n(agg, skip, 7)) | \
               (catch_n(agg, skip, 8)) | \
               (catch_n(agg, skip, 9))

    elif catch == 11:
        return (catch_first(agg, skip)) | \
               (catch_n(agg, skip, 1)) | \
               (catch_n(agg, skip, 2)) | \
               (catch_n(agg, skip, 3)) | \
               (catch_n(agg, skip, 4)) | \
               (catch_n(agg, skip, 5)) | \
               (catch_n(agg, skip, 6)) | \
               (catch_n(agg, skip, 7)) | \
               (catch_n(agg, skip, 8)) | \
               (catch_n(agg, skip, 9)) | \
               (catch_n(agg, skip, 10))

    elif catch == 12:
        return (catch_first(agg, skip)) | \
               (catch_n(agg, skip, 1)) | \
               (catch_n(agg, skip, 2)) | \
               (catch_n(agg, skip, 3)) | \
               (catch_n(agg, skip, 4)) | \
               (catch_n(agg, skip, 5)) | \
               (catch_n(agg, skip, 6)) | \
               (catch_n(agg, skip, 7)) | \
               (catch_n(agg, skip, 8)) | \
               (catch_n(agg, skip, 9)) | \
               (catch_n(agg, skip, 10)) | \
               (catch_n(agg, skip, 11))

    else:
        raise Exception("Algorithm doesn't support catch > 12")


def labelize(data, label):
    data.loc[:, 'Label'] = label
    data_label = pd.DataFrame(data.loc[:, 'Label'])
    data = data.drop('Label', axis=1)
    return data, data_label
