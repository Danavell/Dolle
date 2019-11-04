def drop_first_rows(data):
    """
    The first row of many JOBNUMs contain strange readings that are unrepresentative of
    the data as a whole, implying that they should be dropped
    """
    indices = data.loc[data.loc[:, '0103 ID'] == 1].index
    return data.drop(indices, axis=0)
