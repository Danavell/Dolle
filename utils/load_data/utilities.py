def convert_to_float(d):
    return d.str.split(',').str.join('.').astype('float')

