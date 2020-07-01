import pandas as pd
from os.path import join
from utils.sensor_data.feature_extraction import calc_rolling_time

path = r'C:\Development\DolleProject\dolle_csvs\28-02-16 to 2018-12-19\MLAgg0103 1405 - 1 SW, 3 CF, no overlaps'
filename = 'sensor_data.csv'
path = join(path, filename)

data = pd.read_csv(path, parse_dates=True, infer_datetime_format=True)
data['0103 ID'].fillna(0, inplace=True)

label = 'Downtime Label'
data[label] = 0
condition = (data['0103 ID'] > 0) & (data['0103 Pace'] > 60)
data.loc[condition, label] = 1


cols = [
    'JOBNUM', 'Date', 'Indgang 0101', 'Indgang 0102', 'Indgang 0103', 
    'Indgang 0104', 'Indgang 0105', 'Indgang 0106', '0103 ID'
]
a = data[cols].copy()
a['Date'] = pd.to_datetime(a['Date'])


def fill_time_gaps(group):
    group.reset_index(inplace=True, drop=True)
    first = group.loc[0, 'Date']
    last = group.loc[len(group) - 1, 'Date']
    i = pd.date_range(start=first, end=last, freq='S')
    output = pd.DataFrame(index=i)
    output = pd.merge(output, group.set_index('Date'), how='left', left_index=True, right_index=True)
    return output


b = a.groupby('JOBNUM').apply(fill_time_gaps)
b['Date'] = b.index.get_level_values(1)
b.reset_index(inplace=True, drop=True)
b = calc_rolling_time(b, groupby_cols=['JOBNUM', '0103 ID'])

