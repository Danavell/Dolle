from abc import ABC, abstractmethod

import pandas as pd
from sklearn.decomposition import PCA

from machine_learning.PreProcess import Utilities as util


class PreProcessBase(ABC):
    def __init__(self, agg, num_rows, skip, scaler):
        self._agg = agg
        self._num_rows = num_rows
        self._skip = skip
        self._scaler = scaler
        self._num_columns = len(agg.columns) - 1
        self._deacs, self._non_deacs = None, None
        self._meta = dict()


class Catch(PreProcessBase):
    def __init__(self, agg, num_rows, skip, catch, scaler, true_label):
        super().__init__(agg, num_rows, skip, scaler)
        self._first_row, self._skip = util.adjust_inputs_catch(num_rows, skip, catch)
        self._catch = catch
        self._true_label = true_label
        self._label_group = None
        self._meta['catch'] = catch
        self._meta['JOBNUMs'] = pd.DataFrame(self._agg.loc[:, 'JOBNUM'].copy())
        self._meta['original labels'] = self._agg.loc[:, 'Label'].copy()

    @abstractmethod
    def _separate_deacs_and_non_deacs(self):
        self._agg = util.flatten(self._agg, self._first_row, self._skip, catch=True)
        self._agg, self._scaler = util.scale(self._agg, self._scaler)
        self._meta['scaler'] = self._scaler
        self._agg = util.add_labels(self._skip, self._catch, self._agg)

    def _base_catch_multi_separate_deacs_and_non_deacs(self):
        self._agg.loc[:, 'Label'] = self._agg.loc[:, 'True Label']
        self._agg = self._agg.drop('True Label', axis=1)
        condition = self._agg.loc[:, 'Label'] > 0
        self._deacs, self._non_deacs = util.split_deacs_and_non_deacs(self._agg, condition)
        self._deacs_labels, self._non_deacs_labels = util.separate_labels(self._deacs, self._non_deacs)

    def _base_catch_single_separate_deacs_and_non_deacs(self):
        self._true_labels = self._agg.loc[:, 'True Label']
        self._true_labels = self._true_labels[self._true_labels > 0]
        self._agg = self._agg.drop('True Label', axis=1)

        condition = util.return_catch_condition(self._agg, self._skip, self._catch)
        self._deacs, self._non_deacs = util.split_deacs_and_non_deacs(self._agg, condition)
        self._non_deacs, self._non_deacs_labels = util.labelize(self._non_deacs, 0)

    def _single_drop_columns(self):
        self._deacs, self._non_deacs, label_group = util.drop_original_labels(
            self._deacs, self._non_deacs
        )
        self._deacs, self._deacs_labels = util.labelize(self._deacs, 1)
        self._deacs, self._non_deacs = util.cols_to_drop(
            self._catch, self._num_columns, self._skip, self._deacs, self._non_deacs
        )


class CatchTTS(Catch):
    def __init__(self, agg, num_rows, skip, catch, split, scaler, true_label, start_deac, ltsm):
        super().__init__(agg, num_rows, skip, catch, scaler, true_label)
        self._split = split
        self._start_deac = start_deac
        self._deac_labels, self._non_deac_labels = None, None
        self._true_labels = None
        self._X_train, self._y_train, self._X_test, self._y_test = None, None, None, None
        self._ltsm = ltsm

    @abstractmethod
    def pre_process(self):
        self._num_columns -= 1
        deacs_train, deacs_test, self._meta = util.split_deacs(
            self._deacs, self._split, self._meta, self._true_label, self._start_deac
        )
        indices = util.return_test_indices_range(deacs_test)
        condition = self._non_deacs.index.isin(indices)
        self._assemble_x_test_y_test(deacs_test, condition)
        zeros = self._non_deacs.loc[~condition, :]
        self._assemble_x_train_y_train(zeros, deacs_train)
        data = self._meta['JOBNUMs'].loc[indices, :]
        for column in self._X_test.columns:
            data[column] = self._X_test.loc[:, column]
        self._meta['raw X_test'] = data.copy()
        data = data.dropna()
        self._meta['JOBNUMs'] = data.loc[:, 'JOBNUM']
        if self._ltsm:
            reversed_cols = self._X_train.columns[::-1]
            self._X_train = self._X_train.loc[:, reversed_cols]\
                .to_numpy()\
                .reshape(-1, self._num_rows, self._num_columns)

            self._X_test = self._X_test.loc[:, reversed_cols]\
                .to_numpy()\
                .reshape(-1, self._num_rows, self._num_columns)

    def _single_separate_deacs_and_non_deacs(self):
        super()._base_catch_single_separate_deacs_and_non_deacs()
        self._deacs, self._non_deacs, label_group = util.drop_original_labels(
            self._deacs, self._non_deacs
        )
        self._deacs, self._deacs_labels = util.labelize(self._deacs, 1)
        self._deacs, self._non_deacs = util.cols_to_drop(
            self._catch, self._num_columns, self._skip, self._deacs, self._non_deacs
        )
        labels = pd.concat([label_group, self._deacs_labels], axis=1)
        self._deacs = pd.concat([self._deacs, labels], axis=1)
        self._deacs['True Labels'] = self._true_labels
        self._non_deacs = pd.concat([self._non_deacs, self._non_deacs_labels], axis=1)
        self._deacs = self._deacs.dropna()
        self._non_deacs = self._non_deacs.dropna()

    def _multi_separate_deacs_and_non_deacs(self):
        super()._base_catch_multi_separate_deacs_and_non_deacs()
        self._deacs, self._non_deacs, label_group = util.drop_original_labels(
            self._deacs, self._non_deacs
        )
        self._deacs, self._non_deacs = util.cols_to_drop(
            self._catch, self._num_columns, self._skip, self._deacs, self._non_deacs
        )
        labels = pd.concat([label_group, self._deacs_labels], axis=1)
        self._deacs = pd.concat([self._deacs, labels], axis=1)
        self._non_deacs = pd.concat([self._non_deacs, self._non_deacs_labels], axis=1)

    def _assemble_x_test_y_test(self, deacs_test, condition):
        non_deacs_test = self._non_deacs.loc[condition, :]
        data = pd.concat([deacs_test, non_deacs_test], axis=0, sort=False)
        data = data.sort_index(axis=0)
        if self._true_label:
            true_label = data.iloc[:, -1].fillna(0)
            self._meta['True Label'] = true_label
            data = data.iloc[:, :-1]
        self._X_test = data.iloc[:, :-1]
        self._y_test = data.iloc[:, -1]
        self._y_test = pd.get_dummies(data.iloc[:, -1])

    def _assemble_x_train_y_train(self, zeros, deacs_train):
        categories = set(self._agg.loc[:, 'Label'])
        zeros = zeros.reset_index(drop=True)
        output = zeros.copy()
        for i in range(1, len(categories)):
            data = util.resize_concat_deacs(zeros, deacs_train, i)
            data = data.reset_index(drop=True)
            output = pd.concat([output, data], axis=0, sort=False)
        output = output.sample(frac=1, replace=False, axis=0)
        self._X_train = output.iloc[:, :-1]
        self._y_train = pd.get_dummies(output.iloc[:, -1])

    def return_values(self):
        return self._X_train, self._y_train, self._X_test, self._y_test, self._meta

#
# class Catch(PreProcessBase):
#     def __init__(self, agg, num_rows, skip, catch, scaler, true_label):
#         super().__init__(agg, num_rows, skip, scaler)
#         self._first_row, self._skip = util.adjust_inputs_catch(num_rows, skip, catch)
#         self._catch = catch
#         self._true_label = true_label
#         self._label_group = None
#         self._meta['catch'] = catch
#         self._meta['JOBNUMs'] = pd.DataFrame(self._agg.loc[:, 'JOBNUM'].copy())
#         self._meta['original labels'] = self._agg.loc[:, 'Label'].copy()
#
#     @abstractmethod
#     def _separate_deacs_and_non_deacs(self):
#         self._agg = util.flatten(self._agg, self._first_row, self._skip, catch=True)
#         self._agg, self._scaler = util.scale(self._agg, self._scaler)
#         self._meta['scaler'] = self._scaler
#         self._agg = util.add_labels(self._skip, self._catch, self._agg)
#
#     def _base_catch_multi_separate_deacs_and_non_deacs(self):
#         self._agg.loc[:, 'Label'] = self._agg.loc[:, 'True Label']
#         self._agg = self._agg.drop('True Label', axis=1)
#         condition = self._agg.loc[:, 'Label'] > 0
#         self._deacs, self._non_deacs = util.split_deacs_and_non_deacs(self._agg, condition)
#         self._deacs_labels, self._non_deacs_labels = util.separate_labels(self._deacs, self._non_deacs)
#
#     def _base_catch_single_separate_deacs_and_non_deacs(self):
#         self._true_labels = self._agg.loc[:, 'True Label']
#         self._true_labels = self._true_labels[self._true_labels > 0]
#         self._agg = self._agg.drop('True Label', axis=1)
#
#         condition = util.return_catch_condition(self._agg, self._catch)
#
#         """
#         Drop redundant columns before slicing condition is applied.
#         Necessary to do before splitting data into 0s and 1s. Particularly
#         if PCA will be applied to the data.
#         """
#         self._agg, self._label_group = util.drop_original_labels_v2(
#             self._agg, condition
#         )
#         self._agg = util.cols_to_drop_v2(
#             self._catch, self._num_columns, self._skip, self._agg
#         )
#
#         # pca = PCA(n_components=(len(self._agg.columns) // 3) * 2)
#         # principalComponents = pca.fit_transform(self._agg)
#         # self._agg = pd.DataFrame(data=principalComponents, index=self._agg.index)
#
#         self._deacs, self._non_deacs = util.split_deacs_and_non_deacs(self._agg, condition)
#         self._non_deacs, self._non_deacs_labels = util.labelize(self._non_deacs, 0)
#
#     def _single_drop_columns(self):
#         self._deacs, self._non_deacs, label_group = util.drop_original_labels(
#             self._deacs, self._non_deacs
#         )
#         self._deacs, self._deacs_labels = util.labelize(self._deacs, 1)
#         self._deacs, self._non_deacs = util.cols_to_drop(
#             self._catch, self._num_columns, self._skip, self._deacs, self._non_deacs
#         )
#
#
# class CatchTTS(Catch):
#     def __init__(self, agg, num_rows, skip, catch, split, scaler, true_label, start_deac):
#         super().__init__(agg, num_rows, skip, catch, scaler, true_label)
#         self._split = split
#         self._start_deac = start_deac
#         self._deac_labels, self._non_deac_labels = None, None
#         self._true_labels = None
#         self._X_train, self._y_train, self._X_test, self._y_test = None, None, None, None
#
#     @abstractmethod
#     def pre_process(self):
#         self._num_columns -= 1
#         deacs_train, deacs_test, self._meta = util.split_deacs(
#             self._deacs, self._split, self._meta, self._true_label, self._start_deac
#         )
#         indices = util.return_test_indices_range(deacs_test)
#         condition = self._non_deacs.index.isin(indices)
#         self._assemble_x_test_y_test(deacs_test, condition)
#         zeros = self._non_deacs.loc[~condition, :]
#         self._assemble_x_train_y_train(zeros, deacs_train)
#         data = self._meta['JOBNUMs'].loc[indices, :]
#         for column in self._X_test.columns:
#             data[column] = self._X_test.loc[:, column]
#         self._meta['raw X_test'] = data.copy()
#         data = data.dropna()
#         self._meta['JOBNUMs'] = data.loc[:, 'JOBNUM']
#
#     def _single_separate_deacs_and_non_deacs(self):
#         super()._base_catch_single_separate_deacs_and_non_deacs()
#         self._deacs, self._deacs_labels = util.labelize(self._deacs, 1)
#         labels = pd.concat([self._label_group, self._deacs_labels], axis=1)
#         self._deacs = pd.concat([self._deacs, labels], axis=1)
#         self._deacs['True Labels'] = self._true_labels
#         self._non_deacs = pd.concat([self._non_deacs, self._non_deacs_labels], axis=1)
#         self._deacs = self._deacs.dropna()
#         self._non_deacs = self._non_deacs.dropna()
#
#     def _multi_separate_deacs_and_non_deacs(self):
#         super()._base_catch_multi_separate_deacs_and_non_deacs()
#         self._deacs, self._non_deacs = util.cols_to_drop(
#             self._catch, self._num_columns, self._skip, self._deacs, self._non_deacs
#         )
#         labels = pd.concat([self._label_group, self._deacs_labels], axis=1)
#         self._deacs = pd.concat([self._deacs, labels], axis=1)
#         self._non_deacs = pd.concat([self._non_deacs, self._non_deacs_labels], axis=1)
#
#     def _assemble_x_test_y_test(self, deacs_test, condition):
#         non_deacs_test = self._non_deacs.loc[condition, :]
#         data = pd.concat([deacs_test, non_deacs_test], axis=0, sort=False)
#         data = data.sort_index(axis=0)
#         if self._true_label:
#             true_label = data.iloc[:, -1].fillna(0)
#             self._meta['True Label'] = true_label
#             data = data.iloc[:, :-1]
#         self._X_test = data.iloc[:, :-1]
#         self._y_test = data.iloc[:, -1]
#         self._y_test = pd.get_dummies(data.iloc[:, -1])
#
#     def _assemble_x_train_y_train(self, zeros, deacs_train):
#         num_categories = len(set(deacs_train.loc[:, 'Label'])) + 1
#         zeros = zeros.reset_index(drop=True)
#         output = zeros.copy()
#         for i in range(1, num_categories):
#             data = util.resize_concat_deacs(zeros, deacs_train, i)
#             data = data.reset_index(drop=True)
#             output = pd.concat([output, data], axis=0, sort=False)
#         output = output.sample(frac=1, replace=False, axis=0)
#         self._X_train = output.iloc[:, :-1]
#         self._y_train = pd.get_dummies(output.iloc[:, -1])
#
#     def return_values(self):
#         return self._X_train, self._y_train, self._X_test, self._y_test, self._meta


class CatchNoTTS(Catch):
    def __init__(self, agg, num_rows, skip, catch, scaler, true_label, ltsm):
        super().__init__(agg, num_rows, skip, catch, scaler, true_label)
        self._X, self._y = None, None
        self._ltsm = ltsm

    @abstractmethod
    def pre_process(self):
        self._X = pd.concat((self._deacs, self._non_deacs), axis=0).sort_index()
        self._y = pd.concat((self._non_deacs_labels, self._deacs_labels), axis=0).sort_index()
        if self._ltsm:
            self._X = self._X.loc[:, self._X.columns[::-1]] \
                .to_numpy() \
                .reshape(-1, self._num_rows, self._num_columns)

    def _single_separate_deacs_and_non_deacs(self):
        super()._base_catch_single_separate_deacs_and_non_deacs()
        super()._single_drop_columns()

    def return_values(self):
        return self._X, self._y, self._meta


class NoCatch(PreProcessBase):
    def __init__(self, agg, num_rows, skip, scaler):
        super().__init__(agg, num_rows, skip, scaler)
        self._num_columns = len(agg.columns) - 2
        self._first_row, self._skip = util.adjust_inputs(num_rows, skip)

    def _flatten_scale(self):
        self._agg = util.flatten(self._agg, self._first_row, self._skip, catch=False)
        self._agg, self._scaler = util.scale(self._agg, self._scaler)
        self._meta['scaler'] = self._scaler

