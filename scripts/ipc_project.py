import matplotlib.pyplot as plt
import numpy as np

from sklearn import metrics

from scripts import pre_process_test as p
from machine_learning.MLModels import DolleLSTM, DolleNeural1D


def paths(folder):
    aggregate_path = rf'/home/james//Documents/DolleProject/dolle_csvs/{folder}' \
                     r'/MLAgg0103 1405: 1 SW, 3 CF, no overlaps/SW-3D-3F-3B-12T.csv'

    sensor_path = rf'/home/james//Documents/DolleProject/dolle_csvs/{folder}' \
                  r'/MLAgg0103 1405: 1 SW, 3 CF, no overlaps/sensor_data.csv'
    return aggregate_path, sensor_path


# for _ in range(1):
#     catch = 2
#
#     folder = '28-02-16 to 2018-12-19'
#     aggregate_path, sensor_path = paths(folder)
#     X_train, y_train, meta = p.pre_process_no_tts(
#         agg_path=aggregate_path,
#         sensor_path=sensor_path,
#         catch=catch,
#     )
#
#     folder = 'new_data'
#     aggregate_path, sensor_path = paths(folder)
#     X_test, y_test, meta = p.pre_process_no_tts(
#         agg_path=aggregate_path,
#         sensor_path=sensor_path,
#         catch=catch,
#         balance_classes=True,
#         _scaler=meta['scaler']
#     )
#
#     deac_times = meta['deac_times']
#     test_idx = meta['test_idx']
#
#     model = DolleLSTM()
#     history = model.fit(
#         X_train, y_train, X_test, y_test, epochs=10, class_weights={0: 1, 1: 0.2, 2: 0.2}
#     )
#
#     plt.plot(history.history['loss'], label='train')
#     plt.plot(history.history['val_loss'], label='test')
#     plt.legend()
#     plt.show()
#
#     y_pred = model.predict(X_test)
#     y_pred_1D, y_test_1D = p.flatten_idxmax_y(y_pred, y_test)
#     ConfusionMatrix = confusion_matrix(y_test_1D, y_pred_1D)
#     t, time_matrix = p.calc_time_until_deac(catch, test_idx, y_pred_1D, deac_times)


for _ in range(1):
    catch = 2

    folder = '28-02-16 to 2018-12-19'
    aggregate_path, sensor_path = paths(folder)
    X_train, y_train, X_test, y_test, meta = p.pre_process(
        catch=catch,
        balance_test=False,
        agg_path=aggregate_path,
        sensor_path=sensor_path,
        three_d=True
    )

    deac_times = meta['deac_times']
    test_idx = meta['test_idx']

    model = DolleNeural1D()
    history = model.fit(
        X_train, y_train, X_test, y_test, epochs=50, class_weights={0: 1, 1: 0.2}
    )

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    y_pred = model.predict(X_test)

    y_pred_1D, y_test_1D = p.flatten_idxmax_y(y_pred, y_test)
    ConfusionMatrix = metrics.confusion_matrix(y_test_1D, y_pred_1D)
    c = ConfusionMatrix.reshape(
        1, ConfusionMatrix.shape[0], ConfusionMatrix.shape[1]
    )
    t, time_matrix = p.calc_time_until_deac(catch, test_idx, y_pred_1D, deac_times)

    fpr, tpr, thresholds = metrics.roc_curve(y_test_1D, y_pred_1D, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    if 'confused' in locals() and 'times' in locals():
        confused = np.concatenate((confused, c), axis=0)
        times = np.concatenate((times, t), axis=0)
    else:
        confused = c
        times = t

#
# path = r'/home/james/Documents/DolleProject/'
# np.save(f'{path}confused_cube', confused)
# np.save(f'{path}future_matrix', times)

