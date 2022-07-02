import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from classifier import Model
from helper_functions import calculate_metrics


class DNNClassifier(Model):
    def __init__(self, db: pd.DataFrame, n_layers: int, weight_decay: float, dense_size: list,
                 activation_function: list, learning_rate: float, drop_rate: list,
                 batch_size: int,  n_epochs: int, optimizer: str = 'adam') -> None:
        """
        :param db: cell ephys features dataframe.
        """
        self.wd = weight_decay
        self.lr = learning_rate
        self.dr = drop_rate
        self.af = activation_function
        self.opt = optimizer
        db = self._preprocess_data(db)
        super(DNNClassifier, self).__init__(data=db, num_layers=n_layers, num_neurons=dense_size,
                                            batch_size=batch_size, n_epochs=n_epochs)

    @staticmethod
    def _preprocess_data(df):
        db = df.dropna(axis=1, how='all')
        db = db.dropna(axis=0)
        irrelevant_columns = ['layer', 'structure_area_abbrev', 'sampling_rate', 'mean_clipped', 'file_name']
        db = db.drop([x for x in irrelevant_columns if x in df.columns], axis=1)
        db['dendrite_type'] = pd.Categorical(db['dendrite_type'])
        db['dendrite_type'] = db['dendrite_type'].cat.codes
        return db

    def _create_model(self) -> Sequential:
        """
        :return: creates a logistic regression classifier
        """
        model = Sequential()
        for i in range(self._num_layers):
            model.add(BatchNormalization())
            model.add(Dense(self._num_nodes[i], activation=self.af[i],
                            kernel_regularizer=l2(self.wd), bias_regularizer=l2(self.wd)))
            model.add(Dropout(self.dr[i]))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def train_and_test(self) -> pd.DataFrame:
        """
        :return: results of the logistic regression classifier on the testing data.
        """
        df = self.data
        y = df.pop('dendrite_type')
        y = y.values.astype(np.float32)
        x = df.values.astype(np.float32)

        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.9, random_state=1)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.78, random_state=1)

        if self.opt == 'adam':
            opt = Adam(learning_rate=self.lr)
        if self.opt == 'sgd':
            opt = SGD(learning_rate=self.lr)
        if self.opt == 'rmsprop':
            opt = RMSprop(learning_rate=self.lr)

        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        history = self.model.fit(x_train, y_train, epochs=self.n_epochs, batch_size=self._batch_size,
                                 validation_data=(x_val, y_val), verbose=1)
        loss, acc = self.model.evaluate(x_test, y_test, verbose=2)

        # plot history
        plt.plot(history.history['accuracy'], label='train_accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.show()

        print('--------------------------------------------------------------')
        print("DNN Summary: ")
        self.model.summary()
        print('--------------------------------------------------------------')
        print("Accuracy: " + str(acc))
        return acc

    @staticmethod
    def save_results(results: pd.DataFrame, path: str, name: str) -> None:
        pass


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataframe_path = dir_path + '\\data\\dataframe'
    dataframe_name = 'extracted_mean_ephys_data.csv'
    data = pd.read_csv(dataframe_path + '\\' + dataframe_name)
    dnnclf = DNNClassifier(db=data, n_layers=4, weight_decay=0.01, dense_size=[27, 64, 128, 16],
                           activation_function=['swish', 'swish', 'swish', 'swish'], learning_rate=0.00005,
                           drop_rate=[0, 0.1, 0.2, 0.3], batch_size=32, n_epochs=200, optimizer='adam')
    dnnclf.train_and_test()

    # results_path = dir_path + '\\results\\logistic_regression'
    # model_name = 'logistic_regression.csv'


