import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix


def add_deac_non_duplicate(data, label):
    data = data.fillna(0)
    condition = (data.loc[:, label] > 0) \
                & (data.loc[:, 'prev_label'] == 0) \
                | (data.loc[:, label] > 0) \
                & (data.loc[:, label] >= data.loc[:, 'prev_label'])

    return condition.astype(int)


def _prep_data(y_test_1D, y_pred_1D, jobnums, catch):
    data = pd.concat([jobnums, y_test_1D], axis=1, sort=False)
    data['y_pred'] = y_pred_1D
    groups = y_test_1D.index + 1 - catch
    groups.index = y_test_1D.index
    data['0103 ID'] = groups
    data.columns = ['JOBNUM', 'label', 'y_pred', '0103 ID']
    condition = data.loc[:, 'label'] > 0
    target_group = data.loc[condition, '0103 ID'] + data.loc[:, 'label']
    data['target group'] = target_group.fillna(0).astype(int)
    data['prev_label'] = data.groupby('JOBNUM')['label'].shift(1)
    return data


def add_label_ids(data, label='label'):
    temp = data.groupby('JOBNUM').apply(add_deac_non_duplicate, label)
    temp.index = data.index
    data['original labels'] = temp
    data = data.drop('prev_label', axis=1)
    condition = data.loc[:, 'original labels'] == 1
    temp = data.loc[condition, :]
    temp['label group'] = np.arange(len(temp.index)) + 1
    data.loc[:, 'label group'] = temp['label group']
    condition = data.loc[:, label] > 0
    temp_2 = data.loc[condition, :]
    temp_2.loc[:, 'label group'] = temp.loc[:, 'label group']
    temp_2.loc[:, 'label group'] = temp_2.loc[:, 'label group'].fillna(method='ffill')
    data.loc[:, 'label group'] = temp_2.loc[:, 'label group']
    data.loc[:, 'label group'] = data.loc[:, 'label group'].fillna(0).astype(int)
    return data


def correctly_detected_deactivations(data, meta):
    data = add_label_ids(data)
    condition = (data.loc[:, 'label group'] > 0) & (data.loc[:, 'y_pred'] > 0)
    detected_rows = data.loc[condition, :]
    return detected_rows, data


def get_time_deltas_until_deactivations(detected_rows, sensor_data, previous_time_deltas=None):
    time_deltas = list()
    for _, row in detected_rows.iterrows():
        condition = sensor_data.loc[:, '0103 Group b-filled'] == row['0103 ID']
        times = sensor_data.loc[condition, 'Date']
        start_time = times[times.index[-2]]

        target_group = row['target group']
        condition = sensor_data.loc[:, '0103 Group b-filled'] == target_group
        non_duplicate_0101s = sensor_data.loc[condition, ['Date', 'Non Duplicate 0101']]
        first_index = non_duplicate_0101s.index[0]
        end_time = non_duplicate_0101s.loc[first_index, 'Date']
        time_delta = end_time - start_time
        time_delta = time_delta.total_seconds()
        time_deltas.append(time_delta)

    time_deltas = np.array(time_deltas)
    time_deltas = time_deltas.reshape(-1, 1)
    if previous_time_deltas:
        time_deltas = np.concatenate((previous_time_deltas, time_deltas), axis=0)
        return time_deltas
    return time_deltas


def reshape_data(y_test, y_pred, meta):
    if 'True Label' in meta.keys():
        true = meta['True Label']
        y_true_pd = pd.DataFrame(true)
        y_true_pd.index = y_test.index
        y_test_1D = y_true_pd
    else:
        y_test_pd = pd.DataFrame(y_test)
        y_test_1D = y_test_pd.idxmax(axis=1)
    y_pred_pd = pd.DataFrame(y_pred)
    y_pred_1D = y_pred_pd.idxmax(axis=1)
    y_pred_1D.index = y_test_1D.index
    return y_test_1D, y_pred_1D


def prepare_data(y_test, y_pred, meta):
    catch = meta['catch']
    jobnums = meta['JOBNUMs']
    y_test_1D, y_pred_1D = reshape_data(y_test, y_pred, meta)
    return _prep_data(y_test_1D, y_pred_1D, jobnums, catch)


def confused(y_test, y_pred, meta):
    y_test_1D, y_pred_1D = reshape_data(y_test, y_pred, meta)
    return confusion_matrix(y_test_1D, y_pred_1D)


def _keep_only_furthest_prediction(data):
    return data.loc[data.index[0], :]


def ml_stats(y_test, y_pred, sensor_data, meta):
    data = prepare_data(y_test, y_pred, meta)
    dr, data = correctly_detected_deactivations(data, meta)
    non_unique_total_seen = len(dr.loc[:, 'label group'])
    dr = dr.groupby('target group').apply(_keep_only_furthest_prediction)
    time_deltas = get_time_deltas_until_deactivations(dr, sensor_data)

    condition = data.loc[:, 'original labels'] > 0
    total_num_deactivations = len(data.loc[condition, :])
    total_seen = len(set(dr.loc[:, 'label group']))
    return total_num_deactivations, non_unique_total_seen, total_seen, time_deltas, data, dr


def confusion(y_test, y_pred):
    y_pred = pd.DataFrame(y_pred)
    y_pred_1D = y_pred.idxmax(axis=1)

    y_test_1D = pd.DataFrame(y_test)
    y_test_1D = y_test_1D.idxmax(axis=1)
    confused = confusion_matrix(y_test_1D, y_pred_1D)
    return confused, y_pred_1D


def confucianism_matrix(y_test, y_pred, true_labels, catch):
    def return_indices(concat_pd, y_pred_label, true_label, col_label='y_pred'):
        condition = (concat_pd.loc[:, col_label] == y_pred_label) & \
                    (concat_pd.loc[:, 'true_label'] == true_label)
        return concat_pd[condition].index

    first, y_pred_1D = confusion(y_test, y_pred)
    concat = np.concatenate((y_pred_1D.to_numpy().reshape(-1, 1), true_labels.reshape(-1, 1)), axis=1)
    concat_pd = pd.DataFrame(concat, columns=['y_pred', 'true_label'])
    concat_pd.loc[:, 'output'] = 0
    for true_label in range(1, catch + 1):
        false_positives = return_indices(concat_pd, 0, true_label)
        true_negatives = return_indices(concat_pd, 1, true_label)

        concat_pd.loc[false_positives, 'output'] = 0
        concat_pd.loc[true_negatives, 'output'] = true_label

    y_pred_true = concat_pd.loc[:, 'output']
    confused = confusion_matrix(
        true_labels,
        y_pred_true.to_numpy()
    )
    total_false_negatives = first[0, 1]
    confused[0, 0] -= total_false_negatives
    total_deacs = np.sum([confused[i, 0] + confused[i, i] for i in range(1, confused.shape[1])])
    for i in range(1, confused.shape[1]):
        confused[0, i] = total_false_negatives * ((confused[i, 0] + confused[i, i]) / total_deacs)
    return first, confused
