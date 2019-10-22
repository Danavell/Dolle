class Machine1405:
    def __init__(self):
        self.machine_id = 1405
        self.product_reg_ex = '^CF|^SW'
        self.previous_1_to_6 = [f'previous_010{i}' for i in range(1, 7)]
        self.next_4_to_6 = [f'next_010{i}' for i in range(4, 7)]
        self.current_1_to_6 = [f'Indgang 010{X}' for X in range(1, 7)]
        self.non_duplicates_1_to_6 = [f'Non Duplicate 010{X}' for X in range(1, 7)]
        self.data_generation_columns = {
            'init_raw_sensor_columns': merge(['Date', 'Time'], ['Indgang 010{}'.format(i) for i in range(1, 7)]),
            # 'raw_sensor_dtypes': self._dtypes,
            'init_sample_work_table': ['JOBREF', 'StartDate', 'StartTime', 'Seconds', 'StopDate', 'StopTime', 'SysQtyGood', 'WrkCtrId'],
            'init_work_prepared': ['NAME', 'WRKCTRID', 'JOBREF', 'QTYGOOD', 'StartDateTime', 'StopDateTime', 'Seconds'],
            'init_product_table': ['Name', 'ProdId'],
            'previous_shifted': merge(['previous_Date'], self.previous_1_to_6),
            'previous_1_to_6': self.previous_1_to_6,
            'gen_shifted_columns_1': merge(['Date'], self.current_1_to_6),
            'next_shifted': merge(['next_Date', 'next_0101'], self.next_4_to_6),
            'remove_nans_and_floats': merge(merge(self.previous_1_to_6, merge(['next_0101'], self.next_4_to_6)),
                                              self.non_duplicates_1_to_6),
            'init_columns': merge(['Non Duplicate 0103'], merge(self.previous_1_to_6,
                           merge(['f_0101'], self.next_4_to_6))),
            '_gen_shifted_columns_-1': merge(['Date', 'Indgang 0101'], [f'Indgang 010{i}' for i in range(4, 7)]),
            'convert_to_seconds': merge(['0101 Down Time'], [f'010{i} Alarm Time' for i in range(4, 7)]),
            'columns_to_keep': [
                        'Date', 'JOBREF', 'JOBNUM',
                        'Indgang 0101', 'Non Duplicate 0101', '0101 Down Time',
                        'Indgang 0102', 'Non Duplicate 0102', '0102 Pace',
                        'Indgang 0103', 'Non Duplicate 0103', '0103 Pace',
                        'Indgang 0104', 'Non Duplicate 0104', '0104 Alarm Time',
                        'Indgang 0105', '0105 Alarm Time',
                        'Indgang 0106', '0106 Alarm Time',
                    ]
        }

        self.generate_statistics = {
            'data_agg_dict': {
                'Date': ['first', 'last'],
                'JOBREF': 'first',
                'Non Duplicate 0101': 'sum',
                '0101 Duration': 'sum',
                'Non Duplicate 0102': 'sum',
                '0102 Pace': ['mean', 'median', 'std'],
                'Non Duplicate 0103': 'sum',
                'Non Duplicate 0104': 'sum',
                'Non Duplicate 0105': 'sum',
                'Non Duplicate 0106': 'sum',
                '0103 Pace': ['mean', 'median', 'std'],
                '0104 Alarm Time': 'sum',
                '0105 Alarm Time': 'sum',
                '0106 Alarm Time': 'sum',
            },

            'work_table_agg_dict': {
                'QTYGOOD': 'sum',
                'Seconds': 'sum',
                'NAME': 'first',
            },
            'ordered_stats': [
                ('JOBREF', 'first'),
                ('Date', 'first'),
                ('Date', 'last'),
                'Seconds',
                'No. Deactivations/hour',
                ('0101 Duration', 'sum'),
                '% Down Time',
                'No. 0104/hour',
                '% 0104',
                ('0104 Alarm Time', 'sum'),
                'No. 0105/hour',
                '% 0105',
                'No. 0106/hour',
                '% 0106',
                ('Non Duplicate 0102', 'sum'),
                ('Non Duplicate 0103', 'sum'),
                'QTYGOOD',
                '0103 Count Vs Expected',
                'Strings per Ladder',
                ('0105 Alarm Time', 'sum'),
                ('0106 Alarm Time', 'sum'),
                ('0102 Pace', 'mean'),
                ('0102 Pace', 'median'),
                ('0102 Pace', 'std'),
                ('0103 Pace', 'mean'),
                ('0103 Pace', 'median'),
                ('0103 Pace', 'std'),
                'NAME',
            ],
            'stats_final columns': [
                'JOBREF',
                'Start Time',
                'Stop Time',
                'Job Length(s)',
                'No. Deactivations/hour',
                'Down Time(s)',
                '% Down Time',
                'No. 0104/hour',
                '% 0104',
                '0104 Alarm Sum(s)',
                'No. 0105/hour',
                '% 0105',
                'No. 0106/hour',
                '% 0106',
                '0102 Sum',
                '0103 Sum',
                'Expected 0103',
                '0103 Count Vs Expected',
                'Strings per Ladder',
                '0105 Alarm Sum(s)',
                '0106 Alarm Sum(s)',
                '0102 Pace avg(s)',
                '0102 Pace median(s)',
                '0102 std',
                '0103 Pace avg(s)',
                '0103 Pace median(s)',
                '0103 std',
                'Product',
            ],
        }


def merge(first, second):
    return [item for sublist in [first, second] for item in sublist]

