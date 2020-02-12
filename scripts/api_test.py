from os import mkdir
from os.path import join, exists

import config_settings as cs
from utils.load_data import api


# original = 2441880132
#
#
# def get_number(original):
#     output = list()
#     for i in range(1, original + 1):
#         temp = original
#         if original % i == 0:
#             temp /= i
#             output.append(i)
#             for j in range(1, 6):
#                 current = i + j
#                 if temp % current == 0:
#                     temp /= current
#                     output.append(current)
#                 else:
#                     break
#             if len(output) == 5 and \
#                     output[0]*output[1]*output[2]*\
#                     output[3]*output[4] == original:
#                 break
#             else:
#                 output = list()
#     return output
#
#
# a = get_number(original)
#
#
#
import pandas as pd
# path = '/home/james/Development/DolleProject/dolle_csvs/2020-01-06 04:44:58 - 2020-01-21 18:33:13'
# path = join(path, 'sensor_data.csv')
# sensor_data = pd.read_csv(path, sep=',')
# sensor_data['Indgang 0101'] = abs(sensor_data['Indgang 0101'] - 1)
# sensor_data.to_csv(path, sep=',', index=False)

start = '2020-01-06'
# end = '2020-01-21'
# start = None
end = None
raw_sensor_data = api.get_sensor_data(start=start, end=end)
sensor_data = api.prep_sensor_data(raw_sensor_data)
sensor_data['Indgang 0101'] = abs(sensor_data['Indgang 0101'] - 1)

raw_erp = api.get_erp(start=start, end=end)
erp = api.prep_erp(raw_erp)
erp['Seconds'] = (erp['StopDateTime'] - erp['StartDateTime']).dt.total_seconds()
erp['QTYGOOD'] = 0

start = raw_sensor_data.loc[0, 'timestamp']
start = start.replace('T', ' ').split('.')[0]

end = raw_sensor_data.loc[len(raw_sensor_data) - 1, 'timestamp']
end = end.replace('T', ' ').split('.')[0]

path = join(cs.CSV_PATH, f'{start} - {end}')
if not exists(path):
    mkdir(path)
    print('making folder')
    sensor_data.to_csv(join(path, 'sensor_data.csv'), index=False, sep=',')
    erp.to_csv(join(path, 'work_table.csv'), index=False, sep=',')
else:
    print('folder already exists')
