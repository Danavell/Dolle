import pandas as pd
import numpy as np


def show_errors(sensor_data):
    error_indices = sensor_data.index[sensor_data['Non Duplicate 0101'] == 1]
    output = pd.DataFrame()
    for index in error_indices:
        sensor_data_fragment = sensor_data.loc[index-100: index, :]
        output = pd.concat([output, sensor_data_fragment], axis=0)
    return error_indices, output


def first_input_and_output_indices(data):
    data['Dup Index'] = data.index
    data['JOBNUM DUP'] = data['JOBNUM']
    last_0102 = _get_last_per_group(data, 2)
    last_0103 = _get_last_per_group(data, 3)

    merged = pd.merge(left=last_0102, right=last_0103, how='outer', left_index=True, right_index=True)
    merged = merged.fillna(-1).astype(np.int32)
    indices = merged.index[merged['JOBREF_x'] == -1]
    merged.loc[indices, 'JOBREF_x'] = merged.loc[indices, 'JOBREF_y']
    merged.drop('JOBREF_y', axis=1, inplace=True)
    merged.reset_index(drop=False, inplace=True)
    merged.columns = ['JOBNUM', 'JOBREF' 'Index First 0102', 'Index First 0103']


def _get_last_per_group(data, i):
    data = data.loc[data[f'010{i} Group'] == 1, :]
    return data.groupby(['JOBNUM', '010{i} Group'])[['JOBREF', 'JOBNUM DUP', 'Dup Index']]\
        .last()\
        .set_index('JOBNUM DUP', drop=True)
