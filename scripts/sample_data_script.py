from load_data import load_csv
from models import machines
import os

machine = machines.Machine1405()
columns = machine.data_generation_columns

current_directory = os.getcwd()
data_path = os.path.join(current_directory, r'csvs\sample_data\Ladder machine 1.csv')
work_path = os.path.join(current_directory, r'csvs\sample_data\jmgStampTrans.csv')
prod_path = os.path.join(current_directory, r'csvs\sample_data\ProdTable.csv')


data = load_csv.read_sample_data(data_path, columns)
# work_table = csv.read_sample_work_table(work_path, prod_path,
#                                         machine.machine_id,
#                                         machine.product_reg_ex,
#                                         columns)
# data = sensor_data.filter_sensor_data(work_table, data)
#
# data = sensor_data.create_non_duplicates(data, columns=columns, reduced=2)
# non_cum_data = sensor_data.non_cumulative_data(data, columns=columns)
#
# stats = sensor_data_stats.generate_statistics(non_cum_data, work_table, machine.generate_statistics)
# cols, stats_prods = sensor_data.feature_extraction(stats)
# string_num = sensor_data_stats.generate_string_num(stats_prods, cols)

# non_cum_data = sensor_data.sensor_groupings(data)
# # cum_data = sensor_data.sensor_groupings(data)
# cum_data = sensor_data.cumulative_data(data, columns=columns)


# features, labels, joined = pp.aggregate_pace_in(non_cum_data)
# features.to_csv(r'C:\Users\1067332\Desktop\features.csv', sep=';', index=False)
# labels.to_csv(r'C:\Users\1067332\Desktop\labels.csv', sep=';', index=False)

# data_path = os.path.join(current_directory, r'csvs\sample_data\sample_non_duplicate.csv')
# data = pd.read_csv(data_path,
#                    sep=';',
#                    parse_dates=['Date'],
#                    infer_datetime_format=True)
# data = sensor_data.sensor_groupings(data)
# data = sensor_data.cumulative_data(data)
# data = sensor_data.non_cumulative_data(data, columns=columns)
#
# stats = sensor_data_stats.generate_statistics(data, work_table, machine.generate_statistics)
# cols, stats = sensor_data.feature_extraction(stats)



# data_0103 = data.loc[data['Non Duplicate 0103'] == 1, :]
# data['0103 Group'] = data_0103.groupby('JOBNUM').cumcount()+1
# data['0103 Group'] = data.groupby('JOBNUM')['0103 Group'].fillna(method='bfill')
#
# data_0103 = data.loc[data['Non Duplicate 0102'] == 1, :]
# data['0102 Group'] = data_0103.groupby('JOBNUM').cumcount()+1
# data['0102 Group'] = data.groupby('JOBNUM')['0102 Group'].fillna(method='bfill')

#stats = sensor_data_stats.generate_statistics(new_data, work_table, machine.generate_statistics)
#cols, stats = sensor_data.feature_extraction(stats)

#da.products_dummies(stats)
# stats_2 = sds.generate_statistics(new_data, work_table, machine.generate_statistics, job_ref=False)

