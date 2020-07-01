import matplotlib.pyplot as plt

from os.path import join
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, LSTM


class KerasBase:
    def __init__(self):
        self.model = Sequential()
        self._output_shape = None
        self._early_stopping = None
        self._mcp_save = None
        self._reduce_lr_loss = None

    def compile(self, output_shape):
        self._output_shape = output_shape
        self._early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')

        BASE_DIR = r'C:\Development\DolleProject\dolle_csvs'
        DATE_FOLDER = r'28-02-16 to 2018-12-19'
        self._mcp_save = ModelCheckpoint(
            join(join(BASE_DIR, DATE_FOLDER), 'mdl_wts.hdf5'),
            save_best_only=True, monitor='val_loss', mode='min'
        )
        self._reduce_lr_loss = ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min'
        )

        """compile model using accuracy to measure model performance"""
        if self._output_shape == 2:
            loss_metric = 'binary_crossentropy'
        elif self._output_shape > 2:
            loss_metric = 'categorical_crossentropy'
        else:
            raise ValueError()
        self.model.compile(
            optimizer='adam', loss=loss_metric, metrics=['accuracy']
        )

    def run(self, class_weights, epochs, X_train, y_train, X_test, y_test):
        return self.model.fit(
            X_train, y_train, epochs=epochs, class_weight=class_weights,
            callbacks=[self._early_stopping, self._mcp_save, self._reduce_lr_loss],
            validation_data=(X_test, y_test)
        )

    def save(self, path=r'/home/james/Documents/model.model'):
        self.model.save(path)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_best_weights(self, X_test, path=None):
        if not path:
            raise ValueError('No weights provided')
        self.model.load_weights(path)
        return self.model.predict(X_test)

    def summary(self):
        return self.model.summary()


class DolleNeural1D(KerasBase):
    def __init__(self):
        super().__init__()

    def fit(self, X_train, y_train, X_test=None, y_test=None, epochs=100, class_weights=None):
        output_shape = y_train.shape[1]
        shape = X_train.shape[1]
        input_shape = (shape, )
        shape = round(shape / 2)
        self.model.add(Dense(shape, activation='relu', input_shape=input_shape))
        while True:
            self.model.add(Dropout(0.2, (shape, )))
            shape = round(shape / 2)
            if shape > 2:
                self.model.add(Dense(shape, activation='relu'))
            else:
                self.model.add(Dense(output_shape, activation='softmax'))
                break

        super().compile(output_shape)

        """train the model"""
        if not class_weights:
            class_weights = {0: 1, 1: 1}
        return super().run(class_weights, epochs, X_train, y_train, X_test, y_test)


class DolleLSTM(KerasBase):
    def fit(self, X_train, y_train, X_test=None, y_test=None, input_shape=(3, 6), epochs=100, class_weights=None):
        output_shape = y_train.shape[1]
        self.model.add(LSTM(
            3, input_shape=input_shape, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        )
        self.model.add(LSTM(3, return_sequences=False))
        self.model.add(Dense(output_shape, activation='sigmoid'))

        # fit network
        earlyStopping = EarlyStopping(monitor='loss', patience=20, verbose=1, mode='min')
        mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='loss', mode='min')
        reduce_lr_loss = ReduceLROnPlateau(
            monitor='loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min'
        )

        if not class_weights:
            class_weights = {i: 1 for i in range(output_shape)}

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        val = (X_test, y_test) if X_train is not None and X_test is not None else None
        return self.model.fit(
            X_train, y_train, epochs=epochs, batch_size=32,
            validation_data=val, verbose=1,
            shuffle=False, callbacks=[
                earlyStopping, mcp_save, reduce_lr_loss
            ],
            class_weight=class_weights
        )


class DolleConv2D(KerasBase):
    def __init__(self, num_rows, num_rows_ahead, epochs, output_cols=2):
        super().__init__()
        self.n = num_rows
        if self.n % 2 != 0:
            raise Exception(f'num_rows must be a multiple of 2. Got {self.n}')
        self.na = num_rows_ahead
        self.epochs = epochs
        self.output_cols = output_cols

    def fit(self, X_train, y_train):
        try:
            """add model layers"""
            self.model.add(Conv2D(64, kernel_size=2, activation='relu', input_shape=(self.n, 6, 1)))
            self.model.add(Conv2D(32, kernel_size=2, activation='relu'))
            self.model.add(Flatten())
            # self.model.add(Dense(500, activation='relu'))
            self.model.add(Dense(self.output_cols, activation='softmax'))

            """compile model using accuracy to measure model performance"""
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            """train the model"""
            return self.model.fit(X_train, y_train, epochs=self.epochs)

        except:
            raise Exception('Invalid data was passed to the model')


def create_model_fit_predict(X_train, y_train, X_test, y_test, class_weights=None):
    model = DolleNeural1D()
    history = model.fit(X_train, y_train, X_test=X_test, y_test=y_test, class_weights=class_weights)

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    y_pred = model.predict_best_weights(X_test, r'/home/james/Documents/Development/Dolle/machine_learning/mdl_wts.hdf5')
    return model, y_pred