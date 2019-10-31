import pandas as pd

from utils.sensor_data import feature_extraction as fsd


class StatsFeatureExtractor:
    def __init__(self):
        self.data = None

    def feature_extraction(self, work_table, sensor_data, machine):
        columns = machine.data_generation_columns
        sensor_data = fsd.create_non_duplicates(sensor_data)
        sensor_data = fsd.calculate_pace(sensor_data, columns)

        columns = machine.generate_statistics
        stats = generate_statistics(sensor_data.copy(), work_table, columns)
        agg_dict = {
            'Product': 'first',
            'Job Length(s)': 'sum',
            'Strings per Ladder': ['median', 'mean'],
            '0103 Count Vs Expected': ['median', 'mean'],
            '0102 Pace median(s)': 'median',
            '0102 Pace avg(s)': 'mean',
            '0103 Pace median(s)': 'median',
            '0103 Pace avg(s)': 'mean'
        }
        self.data = dict()
        self.data['sensor_data'] = sensor_data
        self.data['work_table'] = work_table
        self.data['stats'] = stats
        self.data['agg_stats'] = stats.groupby('Product').agg(agg_dict)


def generate_statistics(data, work_table, columns):
    data_groupby = data.groupby('JOBREF')
    agg_dict_stats = data_groupby.agg(columns['data_agg_dict'])
    work_table = work_table[work_table['JOBNUM'].isin(set(data['JOBNUM']))]
    work_groupby = work_table.groupby('JOBREF')
    work_table_dict_stats = work_groupby.agg(columns['work_table_agg_dict'])
    stats = pd.concat([agg_dict_stats, work_table_dict_stats], axis=1)
    stats['No. 0104/hour'] = stats[('Non Duplicate 0104', 'sum')] * (3600 / stats['Seconds'])
    stats[r'% 0104'] = (stats[('0104 Alarm Time', 'sum')] / stats['Seconds']) * 100
    stats['No. 0105/hour'] = stats[('Non Duplicate 0105', 'sum')] * (3600 / stats['Seconds'])
    stats[r'% 0105'] = (stats[('0105 Alarm Time', 'sum')] / stats['Seconds']) * 100
    stats['No. 0106/hour'] = stats[('Non Duplicate 0106', 'sum')] * (3600 / stats['Seconds'])
    stats[r'% 0106'] = (stats[('0106 Alarm Time', 'sum')] / stats['Seconds']) * 100
    stats['No. Deactivations/hour'] = stats[('Non Duplicate 0101', 'sum')] * (3600 / stats['Seconds'])
    stats['0103 Count Vs Expected'] = stats[('Non Duplicate 0103', 'sum')] / stats['QTYGOOD']
    stats[r'% Down Time'] = (stats[('0101 Duration', 'sum')] / stats['Seconds']) * 100
    stats['Strings per Ladder'] = stats[('Non Duplicate 0102', 'sum')] / stats[('Non Duplicate 0103', 'sum')]
    stats = stats[columns['ordered_stats']]
    stats.columns = columns['stats_final columns']
    return stats


def get_product_dummies(data):
    agg = data.copy(deep=True)
    if 'NAME' in data.columns:
        col = 'NAME'
    if 'PRODUCT' in data.columns:
        col = 'PRODUCT'
    if 'Product' in data.columns:
        col = 'Product'
    cols = agg[col].str.split(':', expand=True)
    agg.drop(col, axis=1, inplace=True)
    agg['Product'] = cols[0].apply(lambda x: x[:2].lstrip().rstrip()) + '/' + cols[5].apply(lambda x: x.lstrip().rstrip())
    return pd.get_dummies(agg['Product']), pd.concat([agg, pd.get_dummies(agg['Product'])], axis=1)

