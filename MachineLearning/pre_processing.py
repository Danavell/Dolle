import numpy as np
import pandas as pd

import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler


def _return_catch_condition(agg, catch):
    if catch == 2:
        return (agg.loc[:, 'Label'] == 1) | \
               (agg.loc[:, 'Label 1 lagged'] == 1)
    elif catch == 3:
        return (agg.loc[:, 'Label'] == 1) | \
               (agg.loc[:, 'Label 1 lagged'] == 1) | \
               (agg.loc[:, 'Label 2 lagged'] == 1)
    elif catch == 4:
        return (agg.loc[:, 'Label'] == 1) | \
               (agg.loc[:, 'Label 1 lagged'] == 1) | \
               (agg.loc[:, 'Label 2 lagged'] == 1) | \
               (agg.loc[:, 'Label 3 lagged'] == 1)
    elif catch == 5:
        return (agg.loc[:, 'Label'] == 1) | \
               (agg.loc[:, 'Label 1 lagged'] == 1) | \
               (agg.loc[:, 'Label 2 lagged'] == 1) | \
               (agg.loc[:, 'Label 3 lagged'] == 1) | \
               (agg.loc[:, 'Label 4 lagged'] == 1)
    elif catch == 6:
        return (agg.loc[:, 'Label'] == 1) | \
               (agg.loc[:, 'Label 1 lagged'] == 1) | \
               (agg.loc[:, 'Label 2 lagged'] == 1) | \
               (agg.loc[:, 'Label 3 lagged'] == 1) | \
               (agg.loc[:, 'Label 4 lagged'] == 1) | \
               (agg.loc[:, 'Label 5 lagged'] == 1)
    else:
        raise Exception("Algorithm doesn't support catch > 6")


def _adjust_inputs(num_rows, pred):
    pred -= 1
    first_row = num_rows + pred
    return first_row, pred


def _adjust_inputs_catch(num_rows, pred, catch):
    pred -= 1
    first_row = num_rows + pred + catch - 1
    return first_row, pred


def _create_lags(agg, lags, pred):
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


def _create_lags_catch(agg, lags):
    output = pd.DataFrame()
    for lag in range(lags, -1, -1):
        for column in agg.columns:
            if column != 'JOBNUM':
                if lag == 0:
                    output.loc[:, column] = agg.loc[:, column]
                else:
                    output.loc[:, f'{column} {lag} lagged'] = agg[column].shift(lag)
    return output


def _flatten(agg, first_row, pred, catch=False):
    groupby = agg.groupby('JOBNUM')
    if catch:
        agg = groupby.apply(_create_lags_catch, first_row)
    else:
        agg = groupby.apply(_create_lags, first_row, pred)
    agg.dropna(inplace=True)
    return agg


def _scale(agg, scaler=None):
    if not isinstance(scaler, sklearn.preprocessing.data.MinMaxScaler):
        scaler = MinMaxScaler()
    scaled = scaler.fit_transform(agg)
    agg = pd.DataFrame(data=scaled, columns=agg.columns)
    return agg, scaler


def _train_test(data, split=0.8):
    i = round(len(data.index) * split)
    data = data.to_numpy()
    train = data[:i, :]
    test = data[i:, :]
    return train, test


def _train_test_non_deacs(sample_data, data, deacs, split=0.8):
    i = round(len(sample_data.index) * split)
    sample_data = sample_data.to_numpy()
    train = sample_data[:i, :]
    deacs_df = pd.DataFrame(deacs)
    sample_size = len(deacs_df.index) * 19
    test = data.sample(n=sample_size, random_state=42, axis=0)
    test = test.to_numpy()
    return train, test


def _concat_shuffle(deacs, non_deacs):
    data = np.concatenate([deacs, non_deacs], axis=0)
    data = pd.DataFrame(data)
    data = data.sample(frac=1, replace=False, random_state=42).to_numpy()
    return data


def _split_reshape(data, col, rows, columns, one_d):
    y = data[:, col]
    X = data[:, :col]
    if not one_d:
        X = X.reshape(-1, rows, columns, 1)
    return X, y


def _concat_shuffle_split_reshape(deacs, non_deacs, columns, rows, one_d=False, true_labels=False):
    col = columns * rows
    if true_labels:
        a = np.zeros((non_deacs.shape[0], non_deacs.shape[1] + 1))
        a[:, :-1] = non_deacs
        data = _concat_shuffle(deacs, a)
        X, y = _split_reshape(data, col, rows, columns, one_d)
        return X, y, data[:, col + 1]
    else:
        data = _concat_shuffle(deacs, non_deacs)
        return _split_reshape(data, col, rows, columns, one_d)


def _create_deacs_and_non_deacs(agg, first_row, pred, scaler=None, clustered=False):
    agg = _flatten(agg, first_row, pred)
    agg, scaler = _scale(agg, scaler)
    condition = agg.loc[:, 'Label'] == 1
    deacs = agg.loc[condition, :]
    non_deacs = agg.loc[~condition, :]
    if clustered:
        deacs.drop('Non Duplicate 0101', axis=1, inplace=True)
        non_deacs.drop('Non Duplicate 0101', axis=1, inplace=True)
    return deacs, non_deacs, scaler


def _labelize(data, label):
    data.loc[:, 'Label'] = label
    data_label = pd.DataFrame(data.loc[:, 'Label'])
    data = data.drop('Label', axis=1)
    return data, data_label


def _cols_to_drop(multiplier, num_columns, deacs, non_deacs):
    cols_to_drop = multiplier * num_columns
    deacs = deacs.iloc[:, :-cols_to_drop]
    non_deacs = non_deacs.iloc[:, :-cols_to_drop]
    return deacs, non_deacs


def _add_end_of_jobnums(data, catch):
    data = data.reset_index(drop=True)
    index = len(data.index) - 1
    for i in range(catch):
        data.loc[index, 'End of JOBNUM'] = i + 1
        index -= 1
    return data


def _create_deacs_and_non_deacs_catch(agg, first_row, skip, catch, num_columns,
                                      true_labels=True):
    # data = agg.copy()
    # agg['End of JOBNUM'] = 0
    # agg = agg.groupby('JOBNUM').apply(_add_end_of_jobnums, catch)
    # agg = agg.reset_index(drop=True)
    agg = _flatten(agg, first_row, pred=None, catch=True)
    agg, scaler = _scale(agg)

    if true_labels:
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
        true_label = agg.loc[:, 'True Label']
        true_label = true_label[true_label > 0]
        agg = agg.drop('True Label', axis=1)
    condition = _return_catch_condition(agg, catch)
    # end_of_JOBNUM = agg.loc[:, 'End of JOBNUM']
    # agg = agg.drop('End of JOBNUM', axis=1)

    deacs = agg.loc[condition, :]
    non_deacs = agg.loc[~condition, :]

    non_deacs, non_deacs_labels = _labelize(non_deacs, 0)
    deacs, deacs_labels = _labelize(deacs, 1)
    deacs, non_deacs = _cols_to_drop(catch, num_columns, deacs, non_deacs)

    if skip > 0:
        deacs, non_deacs = _cols_to_drop(skip, num_columns, deacs, non_deacs)

    cols = deacs.columns
    condition = cols.str.contains('Label')
                # | \
                # (cols.str.contains('End of JOBNUM'))
    cols_to_drop = cols[condition]
    deacs = deacs.drop(cols_to_drop, axis=1)
    non_deacs = non_deacs.drop(cols_to_drop, axis=1)

    deacs = pd.concat([deacs, deacs_labels], axis=1)
    non_deacs = pd.concat([non_deacs, non_deacs_labels], axis=1)
    deacs = deacs.dropna()
    non_deacs = non_deacs.dropna()

    if true_labels:
        deacs = pd.concat([deacs, true_label], axis=1)
        # condition = end_of_JOBNUM.index.isin(deacs.index)
        # end = round(end_of_JOBNUM[condition] * 3)
        # deacs['End of JOBNUM'] = end
        # deacs['Duplicate Index'] = deacs.index
    return deacs, non_deacs, scaler


def _cluster(data, cluster):
    if not isinstance(cluster, sklearn.cluster.k_means_.KMeans):
        cluster = KMeans(n_clusters=6, random_state=0).fit(data)
    else:
        cluster = cluster.fit(data)
    labels = cluster.labels_.reshape(-1, 1)
    return labels, cluster


def _cluster_add_labels(data, cluster=None):
    labels, cluster = _cluster(data, cluster)
    data = data.to_numpy()
    data = np.concatenate((data, labels), axis=1)
    return data, labels, cluster


def _cluster_add_labels_true(data, cluster=None):
    true_labels = data.loc[:, 'True Label']
    data = data.drop('True Label', axis=1)
    labels, cluster = _cluster(data, cluster)
    data = pd.concat([data, true_labels], axis=1)
    data = data.to_numpy()
    data = np.concatenate((data, labels), axis=1)
    return data, labels, cluster


def _train_test_split_categories(data):
    sample_size = round(len(data) * 0.8)
    indices = pd.Series(np.arange(sample_size))
    rnd_indices = indices.sample(frac=1, replace=False, random_state=42)
    train = data[rnd_indices.to_numpy()]

    data_indices = np.arange(len(data))
    mask = np.isin(data_indices, rnd_indices, invert=True)
    test = data[mask]
    return train, test


def _train_test_split_clustered(data, cluster=None, true_labels=False):
    data_train, data_test = None, None
    if true_labels:
        data, cluster_labels, cluster = _cluster_add_labels_true(data, cluster)
    else:
        data, cluster_labels, cluster = _cluster_add_labels(data, cluster)
    for i in range(len(cluster_labels)):
        mask = data[:, data.shape[1] - 1] == i
        fragment = data[mask]
        train, test = _train_test_split_categories(fragment)
        if isinstance(data_train, np.ndarray) and isinstance(data_test, np.ndarray):
            data_train = np.concatenate((data_train, train), axis=0)
            data_test = np.concatenate((data_test, test), axis=0)
        else:
            data_train = train
            data_test = test

    data_train = np.concatenate((data_train, data_train), axis=0)
    return data_train, data_test, cluster


def pre_process(agg, num_rows, skip=1, catch=1, one_d=True, non_deacs_cluster=None,
                deacs_cluster=None, train_test=True, true_labels=False):
    if catch == 0:
        raise Exception('catch cannot be equal to zero. You are trying to look in zero \n'
                        'rows for deactivations')
    elif catch == 1:
        num_columns = len(agg.columns) - 2
        first_row, skip = _adjust_inputs(num_rows, skip)
        if true_labels:
            raise Exception('catch must be > 1 if true_labels == True')
        deacs, non_deacs, scaler = _create_deacs_and_non_deacs(agg, first_row, skip)
    elif catch > 1:
        if not true_labels:
            raise Exception('true_labels cannot be False if catch > 1')
        num_columns = len(agg.columns) - 1
        first_row, skip = _adjust_inputs_catch(num_rows, skip, catch)
        deacs, non_deacs, scaler = _create_deacs_and_non_deacs_catch(
            agg, first_row, skip, catch, num_columns, true_labels=true_labels
        )
        num_columns -= 1
    else:
        raise Exception('catch cannot be less than zero. You are trying to look in less \n'
                        'than zero rows for deactivations')

    if train_test:
        non_deacs_train, non_deacs_test, non_deacs_cluster = _train_test_split_clustered(
            non_deacs, non_deacs_cluster, true_labels=False
        )
        deacs_train, deacs_test, deacs_cluster = _train_test_split_clustered(
            deacs, deacs_cluster, true_labels=true_labels
        )

        # Removes cluster labels
        non_deacs_train = non_deacs_train[:, :non_deacs_train.shape[1] - 1]
        non_deacs_test = non_deacs_test[:, :non_deacs_test.shape[1] - 1]
        deacs_train = deacs_train[:, :deacs_train.shape[1] - 1]
        deacs_test = deacs_test[:, :deacs_test.shape[1] - 1]

        diff = len(non_deacs_train) // len(deacs_train)
        final_diff = len(non_deacs_train) - (len(deacs_train) * diff)
        deacs_train_copy = deacs_train.copy()
        for _ in range(1, diff):
            deacs_train = np.concatenate((deacs_train, deacs_train_copy), axis=0)

        indices = pd.Series(np.arange(len(deacs_train_copy)))
        rnd_indices = indices.sample(n=final_diff, replace=False, random_state=42)
        deacs_sample = deacs_train[rnd_indices.to_numpy()]
        deacs_train = np.concatenate((deacs_train, deacs_sample), axis=0)

        if true_labels:
            deacs_train = deacs_train[:, :deacs_train.shape[1] - 1]
        X_train, y_train = _concat_shuffle_split_reshape(
            deacs_train, non_deacs_train, num_columns, num_rows, one_d=one_d,
        )
        test_output = _concat_shuffle_split_reshape(
            deacs_test, non_deacs_test, num_columns, num_rows, one_d=one_d, true_labels=true_labels
        )

        if true_labels:
            X_test, y_test, true_label = test_output
        else:
            X_test, y_test = test_output

        y_train = pd.get_dummies(y_train)
        y_test = pd.get_dummies(y_test)

        if catch > 1 and true_labels:
            return X_train, y_train, X_test, y_test, true_label, deacs_cluster, non_deacs_cluster, scaler
        else:
            return X_train, y_train, X_test, y_test, None, deacs_cluster, non_deacs_cluster, scaler
    else:
        X, y = _concat_shuffle_split_reshape(non_deacs, deacs, num_columns, num_rows)
        if catch > 1 and true_labels:
            return X, y, None
        return X, y


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
