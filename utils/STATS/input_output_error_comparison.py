import pandas as pd
import datetime as dt
import numpy as np
from statisticss import sensor_data_stats as sds

#df_sensor_data = pd.read_csv("sensor_data_new_new.csv", sep=",", parse_dates = ['Date'], infer_datetime_format = True)
df_sensor_data_copy = pd.read_csv("sensor_data_new_new_copy.csv", sep=",", parse_dates = ['Date'], infer_datetime_format = True)
stats = pd.read_csv(r'stats.csv')
products = pd.read_csv(r'products.csv')
products.drop(columns=['Unnamed: 0'], axis=1, inplace=True)

stats.drop(columns=['CF/3D/3F/2B/12T',
       'CF/3D/4F/3B/12T', 'CF/3D/4F/4B/12T', 'CF/4D/3F/2B/12T',
       'CF/4D/3F/3B/11T', 'CF/4D/3F/3B/12T', 'CF/4D/4F/4B/12T',
       'SW/2D/4F/4B/3T', 'SW/2D/4F/4B/4T', 'SW/3D/3F/3B/11T',
       'SW/3D/3F/3B/12T', 'SW/3D/3G/3B/12T'], axis=1, inplace=True)
colname = ['CF/3D/3F/2B/12T', 'CF/3D/4F/3B/12T', 'CF/3D/4F/4B/12T',
       'CF/4D/3F/2B/12T', 'CF/4D/3F/3B/11T', 'CF/4D/3F/3B/12T',
       'CF/4D/4F/3B/12T', 'CF/4D/4F/4B/12T', 'SW/2D/4F/4B/3T',
       'SW/2D/4F/4B/4T', 'SW/3D/3F/3B/11T', 'SW/3D/3F/3B/12T',
       'SW/3D/3G/3B/12T']
stats[colname]=products[colname]

merged = pd.merge(df_sensor_data_copy, stats, on="JOBREF", how="left")

df_sensor_data = merged
condition_1 = (df_sensor_data['CF/3D/3F/2B/12T'] == 1) | \
              (df_sensor_data['CF/3D/4F/4B/12T'] == 1) | \
              (df_sensor_data['SW/3D/3F/3B/12T'] == 1)   
df_sensor_data = df_sensor_data.loc[condition_1]
df_sensor_data.drop(columns=['Unnamed: 0'], axis=1, inplace=True)

temp_a = df_sensor_data.loc[df_sensor_data['Non Duplicate 0102'] == 1]
temp_a['Indgang 0102 Duplicate'] = temp_a.groupby('JOBNUM').cumcount()

df_sensor_data['Indgang 0102 Duplicate'] = temp_a['Indgang 0102 Duplicate']
df_sensor_data['Indgang 0102 Duplicate'] = df_sensor_data['Indgang 0102 Duplicate'].fillna(method='ffill')

temp_b = df_sensor_data.loc[df_sensor_data['Non Duplicate 0103'] == 1]
temp_b['Indgang 0103 Duplicate'] = temp_b.groupby('JOBNUM').cumcount()

df_sensor_data['Indgang 0103 Duplicate'] = temp_b['Indgang 0103 Duplicate']
df_sensor_data['Indgang 0103 Duplicate'] = df_sensor_data['Indgang 0103 Duplicate'].fillna(method='ffill')

df_sensor_data.fillna(0, inplace = True)

# =============================================================================
# def asdf(sensor_data, condition):
#     f = sensor_data.loc[condition]
#     filtered_data = pd.DataFrame()
#     for _, row in f.iterrows():
#         start = row['Date']
#         stop = row['Date'].to_pydatetime() + dt.timedelta(minutes=20)
#         condition = sensor_data['Date'].between(start, stop, inclusive=True)
#         new_data = sensor_data.loc[condition, :].copy()
#         filtered_data = pd.concat([filtered_data, new_data], sort=True)
#     filtered_data = filtered_data.sort_values('Date').reset_index(drop=True)
# 
#     aggregation = { 'JOBNUM': 'first', 'Non Duplicate 0101' : 'sum' }
#     return filtered_data.groupby('JOBNUM').agg(aggregation)
# 
# 
# condition = (df_sensor_data['Indgang 0103 Duplicate'] == 1) & (df_sensor_data['Indgang 0102 Duplicate'] == 0)
# condition_2 = (df_sensor_data['Indgang 0103 Duplicate'] == 0) & (df_sensor_data['Indgang 0102 Duplicate'] >= 15) 
# 
# f_data = asdf(df_sensor_data, condition)
# f_data2 = asdf(df_sensor_data, condition_2)
# 
# =============================================================================

df_sensor_data_empty = df_sensor_data.loc[(df_sensor_data['Indgang 0103 Duplicate'] == 0) & \
                                          (df_sensor_data['Indgang 0102 Duplicate'] >= 15)]

setlist1 = set(df_sensor_data_empty['JOBNUM'])

df_sensor_data_copy = df_sensor_data['JOBNUM'].isin(setlist1)

df_sensor_data_empty_main = df_sensor_data.loc[df_sensor_data_copy]

df_sensor_data_empty_secondary = df_sensor_data[df_sensor_data_copy]
#f_2.drop(['Unnamed: 0'], axis = 1, inplace = True)
aggregation = { 'JOBNUM' : 'first', 'Date' : 'first'}
df_sensor_data_empty_secondary = df_sensor_data_empty_secondary.groupby('JOBNUM').agg(aggregation)
df_sensor_data_empty_secondary['Date_20'] = (df_sensor_data_empty_secondary['Date'] + dt.timedelta(minutes=1)).astype('datetime64[ns]')

#-------------------

df_sensor_data_full = df_sensor_data.loc[(df_sensor_data['Indgang 0103 Duplicate'] == 1) & \
                                         (df_sensor_data['Indgang 0102 Duplicate'] >= 0)]

setlist2 = set(df_sensor_data_full['JOBNUM'])

df_sensor_data_copy3 = df_sensor_data['JOBNUM'].isin(setlist2)

df_sensor_data_full_main = df_sensor_data.loc[df_sensor_data_copy3]

df_sensor_data_full_secondary = df_sensor_data[df_sensor_data_copy3]
#f_2.drop(['Unnamed: 0'], axis = 1, inplace = True)
aggregation = { 'JOBNUM' : 'first', 'Date' : 'first'}
df_sensor_data_full_secondary = df_sensor_data_full_secondary.groupby('JOBNUM').agg(aggregation)
df_sensor_data_full_secondary['Date_20'] = (df_sensor_data_full_secondary['Date'] + dt.timedelta(minutes=1)).astype('datetime64[ns]')

# =============================================================================
# result = pd.DataFrame()
# for _, row in f_3.iterrows():
#     condition = (f_2['Date_20'] > row['Date']) & (f_2['Date'] <= row['Date'])
#     temp = f_2.loc[condition]
#     result.append(temp)
# 
# =============================================================================

def filter_sensor_data(df_sensor_data_secondary, df_sensor_data_main):
    filtered_data = pd.DataFrame()
    for _, row in df_sensor_data_secondary.iterrows():
        start = row['Date']
        stop = row['Date_20']
        condition = df_sensor_data_main['Date'].between(start, stop, inclusive=True)
        new_data = df_sensor_data_main.loc[condition, :].copy()
        filtered_data = pd.concat([filtered_data, new_data], sort=True)
    filtered_data = filtered_data.sort_values('Date').reset_index(drop=True)
    return filtered_data

empty = filter_sensor_data(df_sensor_data_empty_secondary, df_sensor_data_empty_main)
full = filter_sensor_data(df_sensor_data_full_secondary, df_sensor_data_full_main)
# =============================================================================
# 
# condition_1 = (result['CF/3D/3F/2B/12T'] == 1) | (result['CF/3D/4F/4B/12T'] == 1) | (result['SW/3D/3F/3B/12T'] == 1) 
# result_new_full = result.loc[condition_1]
# result_new_empty = result_2.loc[condition_1]
# 
# =============================================================================

np.sum(empty['Non Duplicate 0101'])
np.sum(full['Non Duplicate 0101'])

len((empty['JOBNUM'])) * 60
len((full['JOBNUM'])) * 60
