# import numpy as np
# import pandas as pd
# import os
# from statisticss import sensor_stats
#
# current_directory = os.getcwd()
# data_path = os.path.join(current_directory, r'csvs\sample_data\sample_non_cumulative.csv')
# data = pd.read_csv(data_path,
#                    sep=';',
#                    parse_dates=['Date'],
#                    infer_datetime_format=True,
#                    )
#
#
# data_0103 = data.loc[data['Non Duplicate 0103'] == 1, :]
# data['0103 Group'] = data_0103.groupby('JOBNUM').cumcount()+1
# data['0103 Group'] = data.groupby('JOBNUM')['0103 Group'].fillna(method='bfill')
#
# data_0103 = data.loc[data['Non Duplicate 0102'] == 1, :]
# data['0102 Group'] = data_0103.groupby('JOBNUM').cumcount()+1
# data['0102 Group'] = data.groupby('JOBNUM')['0102 Group'].fillna(method='bfill')
#

# importing pandas as pd
import pandas as pd

# Let's create the dataframe
df = pd.DataFrame({'Date': ['10/2/2011', '12/2/2011', '13/2/2011', '14/2/2011'],
                   'Event': ['Music', 'Poetry', 'Theatre', 'Comedy'],
                   'Cost': [10000, 5000, 15000, 2000]})


def Insert_row(row_number, df, row_value):
    # Starting value of upper half
    start_upper = 0

    # End value of upper half
    end_upper = row_number

    # Start value of lower half
    start_lower = row_number

    # End value of lower half
    end_lower = df.shape[0]

    # Create a list of upper_half index
    upper_half = [*range(start_upper, end_upper, 1)]

    # Create a list of lower_half index
    lower_half = [*range(start_lower, end_lower, 1)]

    # Increment the value of lower half by 1
    lower_half = [x.__add__(1) for x in lower_half]

    # Combine the two lists
    index_ = upper_half + lower_half

    # Update the index of the dataframe
    df.index = index_

    # Insert a row at the end
    df.loc[row_number] = row_value

    # Sort the index labels
    df = df.sort_index()

    # return the dataframe
    return df


import numpy as np
# Let's create a row which we want to insert
row_number = 2
row_value = list(np.arange(8))

if row_number > df.index.max() + 1:
    print("Invalid row_number")
else:

    # Let's call the function and insert the row
    # at the second position
    agg = Insert_row(row_number, agg, row_value)

    agg = Insert_row(2, agg, agg.loc[0, :])
    # Print the updated dataframe
    print()