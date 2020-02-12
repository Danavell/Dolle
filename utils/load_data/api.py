import datetime
import json
import re
import requests

import pandas as pd


url = 'http://92.43.176.119:3000/api/'


def authenticateToken(url):
    head = {'Authorization': 'token {}'.format('DASHBOARDKEY')}
    return requests.get(url, headers=head)


def test_date(date, end=False):
    regex = '^20[1-2][0-9]-[0-1][0-2]-[0-3][0-9]'
    full_date = f'{regex}T[0-2][0-3]:[0-5][0-9]:[0-5][0-9]$'
    if re.match(f'{regex}$', date):
        return f'{date}T23:59:59' if end else f'{date}T00:00:00'
    elif re.match(full_date, date):
        return date
    else:
        raise Exception(f'{date} does not fit the correct time format')


def create_url_ending(start, end, erp=False):
    if start is None and end is not None:
        raise Exception('Start is None but end is not')
    elif start is None and end is None:
        return f'productions' if erp else f'inputreadings'
    elif start is not None and end is None:
        end = datetime.datetime.now().date().__str__()

    ending = f'byinterval/{test_date(start)},{test_date(end, end=True)}'
    return f'productions{ending}' if erp else f'inputreadings{ending}'


def get_sensor_data(start=None, end=None, machineName='laddermachine1', url=url):
    conn = (url + machineName + '/' + create_url_ending(start, end))
    response = authenticateToken(conn)
    convert_to_json = json.dumps(response.json())
    parse_json = json.loads(convert_to_json)
    return pd.DataFrame(parse_json)


def get_erp(start=None, end=None, url=url):
    conn = (url + create_url_ending(start, end, erp=True))
    response = authenticateToken(conn)
    convert_to_json = json.dumps(response.json())
    parse_json = json.loads(convert_to_json)
    return pd.DataFrame(parse_json)


def prep_erp(work_table):
    work_table.rename(
        columns={
            "starttime": 'StartDateTime',
            "stoptime": 'StopDateTime',
            'machine_id': 'WRKCTRID',
            'name': 'NAME',
            'jobid': 'JOBREF',
            "from": 'planned_starttime',
            'to': 'planned_stoptime'},
        inplace=True
    )

    work_table.drop(
        columns=[
            'timestamp',
            '__v',
            'pack_group_id',
            '_id',
            'production_type',
            'c_stop',
            'c_start'
        ], inplace=True,
        axis=1
    )

    work_table = work_table.sort_values(by='StartDateTime')
    work_table = work_table.loc[work_table['active'] == 1, :]
    work_table = work_table.drop_duplicates().reset_index(drop=True)
    work_table.drop('active', axis=1, inplace=True)
    work_table['StartDateTime'] = pd.Series(
        pd.to_datetime(work_table['StartDateTime'])
    )
    work_table['StopDateTime'] = pd.Series(
        pd.to_datetime(work_table['StopDateTime'])
    )
    columns = ['WRKCTRID', 'JOBREF']
    work_table[columns] = work_table[columns].astype(int)

    condition = work_table['StopDateTime'] > work_table['StartDateTime']
    return work_table.loc[condition, :].reset_index(drop=True)


def prep_sensor_data(raw_data):
    columns = [f'Indgang 010{i}' for i in range(1, 7)]
    data = pd.DataFrame(index=raw_data.index, columns=columns)

    for i in range(1, len(columns)+1):
        condition = (raw_data.loc[:, 'port'] == f'010{i}') & (raw_data.loc[:, 'value'] == True)
        indices = data.loc[condition].index
        data.loc[indices, f'Indgang 010{i}'] = 1

    data = data.fillna(0)
    data['Date'] = pd.to_datetime(raw_data.loc[:, 'timestamp'])
    data = data[['Date'] + columns]
    return data
