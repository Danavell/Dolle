import pandas as pd


def get_dummy_products(data):
    cols = data.loc[:, 'NAME'].str.split(':', expand=True).dropna()
    data.loc[:, 'NAME'] = cols[0].apply(lambda x: x[:2].lstrip().rstrip()) + '/' + cols[5].apply(lambda x: x.lstrip().rstrip())

    return pd.concat([data, pd.get_dummies(data.loc[:, 'NAME'])], axis=1)


