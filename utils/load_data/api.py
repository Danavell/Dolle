import requests
import json
import pandas as pd

dateFormat = "%Y-%m-%dT%H:%M:%S"
dolleCoreApi = "http://theAPI/api/"
urlEnd = "/inputreadingsbyinterval/"


def authenticateToken(url):
    head = {'Authorization': 'token {}'.format('DASHBOARDKEY')}
    response = requests.get(url, headers=head)
    return response


def get_sensor_data(machineName='laddermachine1'):
    url = (r'http://192.168.222.17:3000/api/' + machineName + "/inputreadings")
    response = authenticateToken(url)
    convert_to_json = json.dumps(response.json())
    parse_json = json.loads(convert_to_json)
    return pd.DataFrame(parse_json)


def get_erp():
    url = (r'http://192.168.222.17:3000/api/productions')
    response = authenticateToken(url)
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

    # work_table.drop(
    #     columns=[
    #         '__v',
    #         'pack_group_id',
    #         '_id',
    #         'production_type',
    #         'c_stop',
    #         'c_start'
    #     ], inplace=True,
    #     axis=1
    # )

    work_table = work_table.sort_values(
        by=['StartDateTime', 'timestamp']
    )
    work_table = work_table.loc[work_table['active'] == 1, :]
    work_table = work_table.drop_duplicates().reset_index(drop=True)

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
