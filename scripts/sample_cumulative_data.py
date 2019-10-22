import pandas as pd
import os

current_directory = os.getcwd()
data_path = os.path.join(current_directory, r'csvs\sample_data\sample_non_cumulative.csv')
data = pd.read_csv(data_path, sep=';')

