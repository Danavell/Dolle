from models import machines
import os
from Functions.DataPreparation import SensorData as sd
from Functions.FeatureExtraction import Aggregates as A
from Functions.FeatureExtraction import SensorData as fsd
from Functions.STATS import STATS as st
import numpy as np
import pandas as pd


class BaseClass:
    def __init__(self,
                 work_table_path,
                 sensor_data_path,
                 stats=False,
                 base_data=False,
                 create_data=True,
                 ):
        if create_data:
            self.machine = machines.Machine1405()
            self.columns = self.machine.data_generation_columns
            path = sd.get_base_dolle_directory()
            self.work_table, self.sensor_data = sd.prepare_base_data(
                path,
                stats=stats,
                base_data=base_data,
                work_table_path=work_table_path,
                sensor_data_path=sensor_data_path
            )

    def save(self, *args):
        first_index = self.sensor_data.index[0]
        first_date = str(self.sensor_data.loc[first_index, 'Date'])
        last_index = len(self.sensor_data.index) - 1
        last_date = str(self.sensor_data.loc[last_index, 'Date'])

        path = sd.get_base_dolle_directory()
        path = os.path.join(path, 'csvs')
        folder = self.format_time(first_date, last_date)
        directory = os.path.join(path, folder)
        if not os.path.exists(directory):
            os.mkdir(directory)

        category = args[-1]
        cat_directory = os.path.join(directory, category)
        if not os.path.exists(cat_directory):
            os.mkdir(cat_directory)
            if len(args) == 1:
                sensor_path = os.path.join(cat_directory, 'sensor_data.csv')
                self.sensor_data.to_csv(sensor_path, sep=';')
                work_path = os.path.join(cat_directory, 'work_table.csv')
                self.work_table.to_csv(work_path, sep=';')
            elif len(args) > 1:
                save_dict = dict(args[0])
                for key in save_dict.keys():
                    if isinstance(save_dict[key], pd.DataFrame):
                        arg_path = os.path.join(cat_directory, f'{key}.csv')
                        save_dict[key].to_csv(arg_path, sep=';', index=False)
            else:
                pass
        else:
            print('DIRECTORY ALREADY EXISTS')

    @staticmethod
    def format_time(first_date, last_date):
        return f'{first_date.split()[0]}  ->  {last_date.split()[0]}'


class MLData(BaseClass):
    def __init__(self, work_table_path, sensor_data_path):
        super().__init__(work_table_path, sensor_data_path)
        self.sensor_data = fsd.feature_extraction_sensor_data(self.sensor_data, self.columns)
        self.sensor_data = fsd.calculate_pace(self.sensor_data, self.columns)

    def save(self, *args):
        super().save(*args)


class MLDataAggregate(MLData):
    def __init__(self, work_table_path, sensor_data_path):
        super().__init__(work_table_path, sensor_data_path)
        self.sensor_data = fsd.ffill_0102_per_0103(self.sensor_data)
        _, self.sensor_data = fsd.get_dummy_products(self.sensor_data)

        condition = (self.sensor_data.loc[:, 'CF/3D/3F/2B/12T'] == 1) | \
                    (self.sensor_data.loc[:, 'CF/3D/4F/4B/12T'] == 1) | \
                    (self.sensor_data.loc[:, 'SW/3D/3F/3B/12T'] == 1)
        self.sensor_data = self.sensor_data.loc[condition, :]

        self.sensor_data['0103 Group b-filled'] = sd.make_column_arange(self.sensor_data,
                                                                        'Non Duplicate 0103',
                                                                        groupby_col='JOBNUM')
        self.sensor_data = A.make_n_length_jam_durations(self.sensor_data)
        self.agg_CF_3D_3F_2B_12T, \
        self.agg_CF_3D_4F_4B_12T, \
        self.agg_SW_3D_3F_3B_12T, \
        self.agg_all_three = A.make_aggregates(self.sensor_data)

    def save(self):
        super().save({
            'sensor_data': self.sensor_data,
            'work_table': self.work_table,
            'agg_CF_3D_3F_2B_12T': self.agg_CF_3D_3F_2B_12T,
            'agg_CF_3D_4F_4B_12T': self.agg_CF_3D_4F_4B_12T,
            'agg_SW_3D_3F_3B_12T': self.agg_SW_3D_3F_3B_12T,
            'agg_all_three': self.agg_all_three
        }, 'MLDataAggregate')


class StatisticsData(BaseClass):
    def __init__(self, work_table_path, sensor_data_path):
        super().__init__(work_table_path, sensor_data_path, stats=True)
        self.sensor_data = fsd.create_non_duplicates(self.sensor_data, self.columns, phantoms=False)
        self.sensor_data = fsd.calculate_pace(self.sensor_data, self.columns)
        self.stats = st.generate_statistics(self.sensor_data.copy(),
                                            self.work_table.copy(),
                                            self.machine.generate_statistics
                                            )
        self.cols, self.stats = st.get_product_dummies(self.stats)
        strings_per_ladders = self.average_stat('Strings per Ladder')
        count_vs_expected = self.average_stat('0103 Count Vs Expected', drop=True)
        pace_in = self.average_stat('0102 Pace median(s)', drop=True, output_col='0102 Pace(s)')
        pace_out = self.average_stat('0103 Pace median(s)', drop=True, output_col='0103 Pace(s)')
        self.aggregate_stats = pd.concat([
            strings_per_ladders,
            count_vs_expected,
            pace_in,
            pace_out
        ], axis=1)

    def average_stat(self, target_col, drop=False, output_col=None):
        temp = []
        for column in self.cols:
            condition = self.stats[column] != 0
            filtered = self.stats.loc[condition, :].copy()
            temp.append([
                column, np.sum(filtered[target_col]) / len(filtered.index), np.median(filtered[target_col])
            ])

        if isinstance(output_col, str):
            target_col = output_col

        data = pd.DataFrame(temp, columns=['Product', f'Average {target_col}', f'Median {target_col}'])
        if drop:
            data.drop('Product', axis=1, inplace=True)
        return data

    def save(self, *args):
        super().save({
            'stats': self.stats,
            'agg_stats': self.aggregate_stats
        }, 'StatisticsData')


class BaseData(BaseClass):
    def __init__(self, work_table_path, sensor_data_path):
        super().__init__(work_table_path, sensor_data_path, base_data=True)

    def save(self):
        super().save('BaseData')


class StatsMlAggFacade(BaseClass):
    def __init__(self, work_table_path, sensor_data_path):
        super().__init__(work_table_path, sensor_data_path, create_data=False)
        stats_model = StatisticsData(work_table_path, sensor_data_path)
        ml_agg_model = MLDataAggregate(work_table_path, sensor_data_path)
        self.stats_work_table = stats_model.work_table
        self.stats_sensor_data = stats_model.sensor_data
        self.stats = stats_model.stats
        self.aggregate_stats = stats_model.aggregate_stats
        self.work_table = ml_agg_model.work_table
        self.sensor_data = ml_agg_model.sensor_data

        self.agg_all_three = ml_agg_model.agg_all_three

        self.agg_CF_3D_3F_2B_12T = ml_agg_model.agg_CF_3D_3F_2B_12T
        self.agg_CF_3D_3F_2B_12T = self.make_pace_sqrt_cube('CF/3D/3F/2B/12T',  self.agg_CF_3D_3F_2B_12T)
        self.agg_CF_3D_3F_2B_12T = fsd.pace_diff_column_sqrt_cube('CF/3D/3F/2B/12T',
                                                                  'Median 0103 Pace(s)',
                                                                  self.aggregate_stats.copy(),
                                                                  self.agg_CF_3D_3F_2B_12T,
                                                                  '0103 pace diff avg',
                                                                  '0103 Pace',
                                                                  sqrt=False)

        self.agg_CF_3D_4F_4B_12T = ml_agg_model.agg_CF_3D_4F_4B_12T
        self.agg_CF_3D_4F_4B_12T = self.make_pace_sqrt_cube('CF/3D/4F/4B/12T', self.agg_CF_3D_4F_4B_12T)
        self.agg_CF_3D_4F_4B_12T = fsd.pace_diff_column_sqrt_cube('CF/3D/4F/4B/12T',
                                                                  'Median 0103 Pace(s)',
                                                                  self.aggregate_stats.copy(),
                                                                  self.agg_CF_3D_4F_4B_12T,
                                                                  '0103 pace diff avg',
                                                                  '0103 Pace',
                                                                  sqrt=False)

        self.agg_SW_3D_3F_3B_12T = ml_agg_model.agg_SW_3D_3F_3B_12T
        self.agg_SW_3D_3F_3B_12T = self.make_pace_sqrt_cube('SW/3D/3F/3B/12T', self.agg_SW_3D_3F_3B_12T)
        self.agg_SW_3D_3F_3B_12T = fsd.pace_diff_column_sqrt_cube('SW/3D/3F/3B/12T',
                                                                  'Median 0103 Pace(s)',
                                                                  self.aggregate_stats.copy(),
                                                                  self.agg_SW_3D_3F_3B_12T,
                                                                  '0103 pace diff avg',
                                                                  '0103 Pace',
                                                                  sqrt=False)

    def make_pace_sqrt_cube(self,
                            product_string,
                            product,
                            agg_column='Median Strings per Ladder',
                            new_column='pace diff from avg',
                            old_column='Non Duplicate 0102',
                            sqrt=True):
        return fsd.pace_diff_column_sqrt_cube(
            product_string, agg_column,
            self.aggregate_stats, product, new_column, old_column, sqrt=sqrt
        )

    def save(self):
        super().save({
            'stats work table': self.stats_work_table,
            'stats_sensor_data': self.stats_sensor_data,
            'stats': self.stats,
            'aggregate stats': self.aggregate_stats,
            'ml work table': self.work_table,
            'ml sensor data': self.sensor_data,
            'agg_CF_3D_3F_2B_12T': self.agg_CF_3D_3F_2B_12T,
            'agg_CF_3D_4F_4B_12T': self.agg_CF_3D_4F_4B_12T,
            'agg_SW_3D_3F_3B_12T': self.agg_SW_3D_3F_3B_12T,
            'agg_all_three': self.agg_all_three
        }, 'Stats and MlAgg')


def model_factory(data_type,
                  work_table_path=r'WORK_TABLE.csv',
                  sensor_data_path=r'01-01-18 to 01-01-19/datacollection.csv'
                  ):
    if isinstance(data_type, str):

        if data_type.lower() == 'stats':
            return StatisticsData(work_table_path, sensor_data_path)

        elif data_type.lower() == 'ml':
            return MLData(work_table_path, sensor_data_path)

        elif data_type.lower() == 'ml_agg':
            return MLDataAggregate(work_table_path, sensor_data_path)

        elif data_type.lower() == 'base':
            return BaseData(work_table_path, sensor_data_path)

        elif data_type.lower() == 'stats & ml_agg':
            return StatsMlAggFacade(work_table_path, sensor_data_path)

    else:
        return ValueError('not a valid identifier')
