import requests
import json
import pandas as pd

data = requests.get(r'http://192.168.222.17:3000/api/laddermachine1/inputreadings').json()
convert_to_json = json.dumps(data)
parse_json = json.loads(convert_to_json)
data = pd.DataFrame(parse_json)

columns = list()

for i in range(1, 7):
    column = f'Indgang 010{i}'
    columns.append(column)
    data.loc[:, column] = 0
    condition = (data.loc[:, 'port'] == f'010{i}') & (data.loc[:, 'value'] == True)
    indices = data.loc[condition].index
    data.loc[indices, f'Indgang 010{i}'] = 1

data.loc[:, 'timestamp'] = pd.to_datetime(data.loc[:, 'timestamp'])
data = data.loc[:, ['timestamp'] + columns]

