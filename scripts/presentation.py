import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from MachineLearning import MLModels as m
from MachineLearning.PreProcess.PreProcess import CatchMultiCategoryTTS, process_data, CatchSingleBlobNoTTS, NoCatchNoTTS
from MachineLearning.STATS import ml_stats, confused

from sklearn.metrics import confusion_matrix

aggregate_path = r'/home/james/Documents/Development/Dolle/csvs/2018-02-16 -> ' \
                 r'2018-12-19/MLAgg/agg_all_three.csv'

agg_cols_to_use = [
    'Non Duplicate 0102', 'Sum 0102 Jam >= 20', '0103 Pace',
    '0104 Alarm Time', '0105 Alarm Time', '0106 Alarm Time', 'Label'
]

agg = pd.read_csv(aggregate_path, sep=';', usecols=agg_cols_to_use)

condition = agg.loc[:, 'Label'] == 1
y = agg.loc[:, 'Label']
agg = agg.drop('Label', axis=1)
scaler = StandardScaler()
scaled = scaler.fit_transform(agg)
agg = pd.DataFrame(data=scaled, index=agg.index)

pca = PCA(n_components=4)
principalComponents = pca.fit_transform(agg)
agg = pd.DataFrame(data=principalComponents, index=agg.index)

deacs = agg.loc[condition, :]
non_deacs = agg.loc[~condition, :]
# non_deacs = non_deacs.sample(n=len(deacs.index))
data = (deacs, non_deacs)
colors = ("red", "blue")
groups = ("Deactivations", "Functioning")

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# ax.scatter(deacs.iloc[:, 0], deacs.iloc[:, 1], alpha=1, c='red', s=30)
# ax.set_xlabel('PCA Column 1')
# ax.set_ylabel('PCA Column 2')
# plt.title('Deactivation and Non Deactivations PCA Scatter')
# plt.show()

alpha = 1
for data, color, group in zip(data, colors, groups):
    x, y = data.iloc[:, 0], data.iloc[:, 1]
    # if color == 'blue':
    #     alpha = 0.4
    # else:
    #     alpha = 1
    ax.scatter(x, y, alpha=alpha, c=color, s=30, label=group)

ax.set_xlabel('PCA Column 1')
ax.set_ylabel('PCA Column 2')
plt.title('Deactivation and Non Deactivations PCA Scatter')
plt.legend(loc=2)
plt.show()
