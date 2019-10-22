from MachineLearning.PreProcess import Utilities as util
from MachineLearning.PreProcess.BaseClasses import CatchTTS, CatchNoTTS, NoCatch

"""
//////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// CATCH + TRAIN TEST SPLIT //////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
"""


class CatchSingleBlobTTS(CatchTTS):
    def __init__(self, agg, num_rows, skip=1, catch=3, split=0.8, scaler=None, start_deac=None, ltsm=None):
        super().__init__(
            agg, num_rows, skip, catch, split, scaler, true_label=True, start_deac=start_deac, ltsm=ltsm
        )

    def pre_process(self):
        self._separate_deacs_and_non_deacs()
        super().pre_process()

    def _separate_deacs_and_non_deacs(self):
        super()._separate_deacs_and_non_deacs()
        self._agg = util.add_unique_deac_ids(self._agg, self._meta)
        super()._single_separate_deacs_and_non_deacs()


class CatchMultiCategoryTTS(CatchTTS):
    def __init__(self, agg, num_rows, skip=1, catch=3, split=0.8, scaler=None, start_deac=None, ltsm=None):
        super().__init__(
            agg, num_rows, skip, catch, split, scaler, true_label=False, start_deac=start_deac, ltsm=ltsm
        )

    def pre_process(self):
        self._separate_deacs_and_non_deacs()
        super().pre_process()

    def _separate_deacs_and_non_deacs(self):
        super()._separate_deacs_and_non_deacs()
        self._agg = util.add_unique_deac_ids(self._agg, self._meta)
        super()._multi_separate_deacs_and_non_deacs()


"""
/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// CATCH + NO TRAIN TEST SPLIT //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
"""


class CatchSingleBlobNoTTS(CatchNoTTS):
    def __init__(self, agg, num_rows, skip, catch, scaler=None):
        super().__init__(agg, num_rows, skip, catch, scaler, true_label=True)

    def pre_process(self):
        self._separate_deacs_and_non_deacs()
        super().pre_process()

    def _separate_deacs_and_non_deacs(self):
        super()._separate_deacs_and_non_deacs()
        self._agg = util.add_unique_deac_ids(self._agg, self._meta)
        super()._single_separate_deacs_and_non_deacs()


class CatchMultiCategoryNoTTS(CatchNoTTS):
    def __init__(self, agg, num_rows, skip, catch, scaler, true_label):
        super().__init__(agg, num_rows, skip, catch, scaler, true_label)

    def _separate_deacs_and_non_deacs(self):
        super()._separate_deacs_and_non_deacs()


"""
/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// NO CATCH + TRAIN TEST SPLIT //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
"""


class NoCatchTTS(NoCatch):
    def __init__(self, agg, num_rows, skip, scaler, ltsm=None):
        super().__init__(agg, num_rows, skip, scaler)
        self._X_train, self._y_train, self._X_test, self._y_test = None, None, None, None

    def _separate_deacs_and_non_deacs(self):
        condition = self._agg.loc[:, 'Label'] == 1
        self._deacs = self._agg.loc[condition, :]
        self._non_deacs = self._agg.loc[~condition, :]

    def pre_process(self):
        super()._flatten_scale()
        self._separate_deacs_and_non_deacs()


"""
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// NO CATCH + NO TRAIN TEST SPLIT //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
"""


class NoCatchNoTTS(NoCatch):
    def __init__(self, agg, num_rows, skip, scaler=None):
        super().__init__(agg, num_rows, skip, scaler)
        self._X, self._y = None, None

    def pre_process(self):
        super()._flatten_scale()
        self._X, self._y = self._agg.iloc[:, :-1], self._agg.iloc[:, -1]

    def return_values(self):
        return self._X, self._y, self._meta


"""
///////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// PRE PROCESS FACTORY FUNCTIONS //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
"""


def process_data(agg,
                 num_rows=2,
                 catch=4,
                 skip=1,
                 split=0.8,
                 scaler=None,
                 method='blob',
                 start_deac=None,
                 ltsm=None
                 ):

    if catch > 1:
        if method == 'blob':
            process = CatchSingleBlobTTS(
                agg, num_rows, skip, catch, split, scaler, ltsm=ltsm, start_deac=start_deac
            )
        elif method == 'multi':
            process = CatchMultiCategoryTTS(
                agg, num_rows, skip, catch, split, scaler, ltsm=ltsm, start_deac=start_deac
            )
        else:
            raise Exception()
    elif catch == 1:
        process = CatchSingleBlobTTS(
            agg, num_rows, skip, catch, split, scaler, ltsm=ltsm, start_deac=start_deac
        )
    else:
        raise Exception()

    process.pre_process()
    return process.return_values()


def no_train_test_split_factory(agg, num_rows=2, catch=4, skip=1, scaler=None, method='blob'):
    if catch > 1:
        if method == 'blob':
            process = CatchSingleBlobNoTTS(
                agg, num_rows, skip, catch, scaler,
            )
        elif method == 'multi':
            process = CatchMultiCategoryNoTTS(
                agg, num_rows, skip, catch, scaler
            )
    else:
        process = NoCatchTTS(
            agg, num_rows, skip, scaler
        )

    process.pre_process()
    return process.return_values()


# from abc import ABC, abstractmethod
# import pandas as pd
# from MachineLearning import utilities as util
#
#
# class PreProcessBase(ABC):
#     def __init__(self, agg, num_rows, skip, scaler):
#         self._agg = agg
#         self._num_rows = num_rows
#         self._skip = skip
#         self._scaler = scaler
#         self._num_columns = len(agg.columns) - 1
#         self._deacs, self._non_deacs = None, None
#         self._meta = dict()
#
#
# class Catch(PreProcessBase):
#     def __init__(self, agg, num_rows, skip, catch, scaler):
#         super().__init__(agg, num_rows, skip, scaler)
#         self._first_row, self._skip = util.adjust_inputs_catch(num_rows, skip, catch)
#         self._catch = catch
#         self._meta['catch'] = catch
#
#     @abstractmethod
#     def _separate_deacs_and_non_deacs(self):
#         self._agg = util.flatten(self._agg, self._first_row, self._skip, catch=True)
#         self._agg, self._scaler = util.scale(self._agg, self._scaler)
#         self._meta['scaler'] = self._scaler
#         self._agg = util.add_labels(self._skip, self._catch, self._agg)
#
#     def _single_separate_deacs_and_non_deacs(self):
#         true_labels = self._agg.loc[:, 'True Label']
#         true_labels = true_labels[true_labels > 0]
#         self.agg = self._agg.drop('True Label', axis=1)
#         condition = util.return_catch_condition(self._agg, self._catch)
#         self._deacs = self._agg.loc[condition, :]
#         self._non_deacs = self._agg.loc[~condition, :]
#         self._non_deacs, non_deacs_labels = util.labelize(self._non_deacs, 0)
#         self._deacs, deacs_labels = util.labelize(self._deacs, 1)
#         self._deacs, self._non_deacs = util.cols_to_drop(
#             self._catch, self._num_columns, self._skip, self._deacs, self._non_deacs
#         )
#         self._deacs = pd.concat([self._deacs, deacs_labels], axis=1)
#         self._deacs['True Labels'] = true_labels
#         self._non_deacs = pd.concat([self._non_deacs, non_deacs_labels], axis=1)
#         self._deacs = self._deacs.dropna()
#         self._non_deacs = self._non_deacs.dropna()
#
#     def _multi_separate_deacs_and_non_deacs(self):
#         self._agg.loc[:, 'Label'] = self._agg.loc[:, 'True Label']
#         self._agg = self._agg.drop('True Label', axis=1)
#         condition = self._agg.loc[:, 'Label'] > 0
#         self._deacs = self._agg.loc[condition, :]
#         self._non_deacs = self._agg.loc[~condition, :]
#         deacs_labels, non_deacs_labels = util.separate_labels(self._deacs, self._non_deacs)
#         self._deacs, self._non_deacs = util.cols_to_drop(
#             self._catch, self._num_columns, self._skip, self._deacs, self._non_deacs
#         )
#         self._deacs = pd.concat([self._deacs, deacs_labels], axis=1)
#         self._non_deacs = pd.concat([self._non_deacs, non_deacs_labels], axis=1)
#
#
# class CatchTTS(Catch):
#     def __init__(self, agg, num_rows, skip, catch, split, scaler, true_label, start_deac):
#         super().__init__(agg, num_rows, skip, catch, scaler)
#         self._split = split
#         self._true_label = true_label
#         self._start_deac = start_deac
#         self._X_train, self._y_train, self._X_test, self._y_test = None, None, None, None
#         self._meta['JOBNUMs'] = pd.DataFrame(self._agg.loc[:, 'JOBNUM'].copy())
#         self._meta['original labels'] = self._agg.loc[:, 'Label'].copy()
#
#     @abstractmethod
#     def pre_process(self):
#         self._num_columns -= 1
#         deacs_train, deacs_test, self._meta = util.split_deacs(
#             self._deacs, self._split, self._meta, self._true_label, self._start_deac
#         )
#         indices = util.return_test_indices_range(deacs_test)
#         self._meta['test indices'] = indices
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
#         categories = set(self._agg.loc[:, 'Label'])
#         zeros = zeros.reset_index(drop=True)
#         output = zeros.copy()
#         for i in range(1, len(categories)):
#             data = util.resize_concat_deacs(zeros, deacs_train, i)
#             data = data.reset_index(drop=True)
#             output = pd.concat([output, data], axis=0, sort=False)
#         output = output.sample(frac=1, replace=False, axis=0)
#         self._X_train = output.iloc[:, :-1]
#         self._y_train = pd.get_dummies(output.iloc[:, -1])
#
#     def return_values(self):
#         return self._X_train, self._y_train, self._X_test, self._y_test, self._meta
#
#
# class CatchSingleBlobTTS(CatchTTS):
#     def __init__(self, agg, num_rows, skip=1, catch=3, split=0.8, scaler=None, start_deac=None):
#         super().__init__(
#             agg, num_rows, skip, catch, split, scaler, true_label=True, start_deac=start_deac
#         )
#
#     def pre_process(self):
#         self._separate_deacs_and_non_deacs()
#         super().pre_process()
#
#     def _separate_deacs_and_non_deacs(self):
#         super()._separate_deacs_and_non_deacs()
#         super()._single_separate_deacs_and_non_deacs()
#
#
# class CatchMultiCategoryTTS(CatchTTS):
#     def __init__(self, agg, num_rows, skip=1, catch=3, split=0.8, scaler=None, start_deac=None):
#         super().__init__(
#             agg, num_rows, skip, catch, split, scaler, true_label=False, start_deac=start_deac
#         )
#
#     def pre_process(self):
#         self._separate_deacs_and_non_deacs()
#         super().pre_process()
#
#     def _separate_deacs_and_non_deacs(self):
#         super()._separate_deacs_and_non_deacs()
#         super()._multi_separate_deacs_and_non_deacs()
