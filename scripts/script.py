import pandas as pd
import os

current_directory = os.getcwd()
directory = os.path.join(current_directory, r'csvs\01-01-18 to 01-01-19')

data_path = os.path.join(directory, r'datacollection.csv')

data = pd.read_csv(data_path, ';')

# data_path = os.path.join(directory, r'datacollection.csv')
# work_path = os.path.join(directory, r'WORK_TABLE.csv')
#
# machine = machines.Machine1405()
# columns = machine.data_generation_columns
#
# data = csv.read_sample_data(data_path, columns)
# work_table = csv.read_work_table(work_path,
#                                 machine.machine_id,
#                                 machine.product_reg_ex,
#                                 columns)
# data = sensor_data.filter_sensor_data(work_table, data)
# data = sensor_data.create_non_duplicates(data, columns=columns, reduced=2)
#
# new_data = sensor_data.non_cumulative_data(data, columns)

# stats = sensor_data_stats.generate_statistics(new_data, work_table, machine.generate_statistics)
# cols, stats = sensor_data.feature_extraction(stats)
# string_num = sensor_data_stats.generate_string_num(stats, cols)

# non_cum_data = sensor_data.sensor_groupings(data)
# cum_data = sensor_data.sensor_groupings(data)
# cum_data = sensor_data.cumulative_data(data, columns=columns)
#
#
# features, labels, joined = pp.aggregate_pace_in(non_cum_data)

# da.products_dummies(stats)
# stats_2 = sensor_data_stats.generate_statistics(new_data, work_table, machine.generate_statistics, job_ref=False)

# work_table = csv.read_sample_work_table(work_path, prod_path,
#                                        machine.machine_id,
#                                        machine.product_reg_ex,
#                                        machine.data_generation_columns)
