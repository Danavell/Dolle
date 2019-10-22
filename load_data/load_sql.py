import pyodbc
import pandas as pd
import numpy as np
import datetime


def _conn():
    return pyodbc.connect('Driver={SQL Server};Server=92.43.176.119;Database=Indusoft;UID=UCN-DDD01;PWD=rQdj7KmG8p;')


def work_table(start, stop, machine=1405):
    work_table = pd.read_sql_query(
        'SELECT p.NAME, j.JOBREF, j.QTYGOOD, j.WRKCTRID'
        'j.CORRSTARTTIME AS start_time, '
        'j.CORRSTOPTIME AS stop_time, '
        'j.CORRSTARTDATE AS start_date, '
        'j.CORRSTOPDATE AS stop_date '
        'FROM JMGSTAMPTRANS AS j LEFT JOIN PRODTABLE AS p ON p.COLLECTREFPRODID = j.JOBREF '
        'WHERE j.WRKCTRID == :machine '
        'AND start_time >= :start AND stop_time <= :stop',
        con=_conn(),
        parse_dates={'start_date': '%Y/%M/%d',
                     'stop_date': '%Y/%M/%d',
                     'start_time': '%H:%M:%S',
                     'stop_time': '%H:%M:%S'},
        params={'start': start, 'stop': stop, 'machine': machine})

    work_table['StartDateTime'] = to_date_time(work_table, work_table['start_date'], work_table['start_time'])
    work_table['StopDateTime'] = to_date_time(work_table, work_table['stop_date'], work_table['stop_time'])
    work_table['Seconds'] = work_table['stop_time'] - work_table['start_time']
    work_table['Seconds'] = work_table['Seconds'].dt.total_seconds().fillna(0).astype(int)
    return work_table


def get_sensor_data(columns, start, stop):
    sensor_data = pd.read_sql_query("SELECT [Date], [Time],[Indgang 0101],[Indgang 0102],"
                                    "[Indgang 0103],[Indgang 0104],[Indgang 0105] ,[Indgang 0106] "
                                    "FROM [Indusoft].[dbo].[DataCollection] "
                                    "WHERE [Date] BETWEEN :start AND :stop",
                                    parse_dates={'Date': '%Y/%M/%d %H:%M:%S'},
                                    params={'start': start, 'stop': stop},
                                    con=_conn(),
                                    )

    time_regex = r'^(2[0-3]|[01]?[0-9]):([1-5]?[0-9]):([1-5]?[0-9])$'

    a = r'^(2[0][0-9][0-9])/([01][1-9])/([0-3][01]|[0-2][0-9])([\s])(2[0-3]|[01][0-9]):([0-5][0-9]):([0-5][0-9])$'

    condition = sensor_data['Time'].str.match(time_regex)
    errors = sensor_data.loc[condition, :].copy()
    if len(errors.index) > 0:
        sensor_data = sensor_data.loc[~condition, :]
        errors['Date'] = errors['Date'].datetime.date
        errors['Date'] = pd.datetime.combine(errors['Date'], errors['Time'])
        sensor_data = pd.concat([sensor_data, errors])
        sensor_data = sensor_data.sort_values('Date').drop('Time')
    return sensor_data


def to_date_time(data, dates, times):
    return data.apply(lambda x: pd.datetime.combine(dates, times), axis=1)


def to_time(time):
    time = time.apply(lambda x: pd.to_datetime(str(datetime.timedelta(seconds=x))))
    return pd.to_datetime(time.values[0], format='%H:%M:%S').dt.time


def to_date(date):
    return pd.to_datetime(date.values[0], format='%Y/%M/%d').dt.date


def convert_to_float(d):
    return d.str.split(',').str.join('.').astype('float')
