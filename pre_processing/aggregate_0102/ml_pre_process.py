from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd


def add_every_second(aggs):
    pace_0102 = aggs['0102 Pace'].astype(int).values[0]
    id = aggs['0102 ID'].values[0]
    output = pd.DataFrame(np.arange(pace_0102), [id for _ in range(pace_0102)])
    output.reset_index(drop=False, inplace=True)
    output.rename(columns={'index': '0102 ID'}, inplace=True)
    return output.merge(
        right=aggs[['Label', 'Time Delta', '0102 ID']], how='left',
        left_on='0102 ID', right_on='0102 ID'
    )


aggs = pd.read_csv(
    r'/home/james/Documents/Development/dolle_csvs/01-01-18 to 01-01-19/'
    r'MLAgg0102 1405: 3 CF, no overlaps/CF-3D-3F-2B-12T - single rows.csv',
    sep=';'
)
d = aggs.loc[aggs.loc[:, 'Time Delta'] >= 0, :]
a = np.percentile(d)

aggs.loc[:, 'next_0102 ID'] = aggs.groupby('JOBNUM').shift(-1)
aggs.loc[:, '0102 per 0103'] = aggs.groupby('JOBNUM').cumcount() + 1
condition = aggs.loc[:, 'next_0102 ID'] != aggs.loc[:, '0102 ID']
aggs = aggs.loc[condition, :].copy()

condition = (aggs.loc[:, '0102 Pace'] >= 25) & (aggs.loc[:, 'Label'] == 0)
num_non_deacs = len(aggs.loc[condition].index)

condition = aggs.loc[:, 'Label'] > 4
total_deacs = len(aggs.loc[condition].index)

condition = (aggs.loc[:, 'Label'] >= 1) & (aggs.loc[:, 'Time Delta'] >= 25)
num_deacs = len(aggs.loc[condition].index)

a = num_deacs / (num_non_deacs + num_deacs)
aggs.loc[:, '0102 ID GP'] = aggs.loc[:, '0102 ID']
data = aggs.groupby('0102 ID GP').apply(add_every_second)

condition = (aggs.loc[:, '0102 Pace'] >= 25) & (aggs.loc[:, 'Label'] == 0)
non_deacs = aggs.loc[condition, :]
ids = aggs.loc[condition, '0103 ID'].unique()
b = aggs.loc[aggs.loc[:, '0103 ID'].isin(aggs.loc[condition, '0103 ID'].unique()), :]


class MLPreProcess:
    def __init__(self, aggregate):
        self._aggs = aggregate

    def pre_process(self):
        condition = self._aggs.groupby('JOBNUM')['0102 ID'].shift(-1) != self._aggs.loc[:, '0102 ID']
        self._aggs = self._aggs.loc[condition, :].copy()

        """
        Calculates when a deactivation occurred. First splits deacs from non deacs
        """
        condition = self._aggs.loc[:, 'Label'] == 1
        non_deacs = self._aggs.loc[~condition, :].copy()
        deacs = self._aggs.loc[condition, :].copy()

        non_deacs = self._aggs.groupby('0102 ID').apply(add_every_second)

        #create previous and future jobnums
        #condition and slice to separate multi rows from single rows. Give multi_rows a different ID?
        #If train test split is to occur it must happen at the beginnning.
        pass

    def train_test_split(self, split):
        pass

    def no_train_test_split(self):
        pass