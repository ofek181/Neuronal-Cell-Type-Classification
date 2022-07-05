import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop
from keras.utils import to_categorical
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from classifier import Model
from helper_functions import calculate_metrics

n_classes = 2

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")


# TODO add calculate_metrics

class DNNClassifier(Model):
    def __init__(self, db: pd.DataFrame, n_layers: int, weight_decay: float, dense_size: list,
                 activation_function: list, learning_rate: float, drop_rate: list,
                 batch_size: int,  n_epochs: int, optimizer: str = 'adam') -> None:
        self.wd = weight_decay
        self.lr = learning_rate
        self.dr = drop_rate
        self.af = activation_function
        self.opt = optimizer
        db = self.preprocess_data(db)
        super(DNNClassifier, self).__init__(data=db, num_layers=n_layers, num_neurons=dense_size,
                                            batch_size=batch_size, n_epochs=n_epochs)

    def _create_model(self) -> Sequential:
        model = Sequential()
        for i in range(self._num_layers):
            model.add(BatchNormalization())
            model.add(Dense(self._num_nodes[i], activation=self.af[i],
                            kernel_regularizer=l2(self.wd), bias_regularizer=l2(self.wd)))
            model.add(Dropout(self.dr[i]))
        model.add(Dense(n_classes, activation='softmax'))
        return model

    def train_and_test(self) -> pd.DataFrame:
        x_train, y_train, x_val, y_val, x_test, y_test = self.split_train_val_test(self.data)

        opt = Adam(learning_rate=self.lr)
        if self.opt == 'sgd':
            opt = SGD(learning_rate=self.lr)
        if self.opt == 'rmsprop':
            opt = RMSprop(learning_rate=self.lr)

        # compile model
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics="accuracy")
        # fit model
        history = self.model.fit(x_train, y_train, epochs=self.n_epochs, batch_size=self._batch_size,
                                 validation_data=(x_val, y_val), verbose=0)
        # plot history
        self.plot_history(history)

        # test the model
        loss, acc = self.test(x_test, y_test)

        # print summary
        # print('--------------------------------------------------------------')
        # print("DNN Summary: ")
        # self.model.summary()
        return acc

    def retrain_for_domain_adaptation(self, new_domain_data: pd.DataFrame, lr: float, n_epochs: int, bs: int) -> None:
        data = self.preprocess_data(new_domain_data)
        x_train, y_train, x_val, y_val, x_test, y_test = self.split_train_val_test(data)

        # set new learning rate
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)

        # fit the new model on the new data
        history = self.model.fit(x_train, y_train, epochs=n_epochs,
                                 batch_size=bs, validation_data=(x_val, y_val), verbose=0)

        # plot history
        self.plot_history(history)

        # test the model
        loss, acc = self.test(x_test, y_test)
        return acc

    def test(self, x_test, y_test) -> tuple:
        # calculate test loss and accuracy
        loss, acc = self.model.evaluate(x_test, y_test, verbose=0)
        print('====================================================')
        print("Accuracy: " + str(acc))
        # plot confusion matrix
        y_pred = self.model.predict(x_test)
        matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
        plt.figure()
        label_names = ['aspiny', 'spiny']
        s = sns.heatmap(matrix / np.sum(matrix), annot=True, fmt='.2%',
                    cmap='Blues', xticklabels=label_names, yticklabels=label_names)
        s.set(xlabel='Predicted label', ylabel='True label')
        plt.draw()
        return loss, acc

    @staticmethod
    def preprocess_data(df):
        db = df.dropna(axis=1, how='all')
        db = db.dropna(axis=0)
        irrelevant_columns = ['layer', 'structure_area_abbrev', 'sampling_rate', 'mean_clipped', 'file_name']
        db = db.drop([x for x in irrelevant_columns if x in df.columns], axis=1)
        db['dendrite_type'] = pd.Categorical(db['dendrite_type'])
        db['dendrite_type'] = db['dendrite_type'].cat.codes
        return db

    @staticmethod
    def split_train_val_test(data: pd.DataFrame) -> tuple:
        scaler = StandardScaler()
        y = data.pop('dendrite_type')
        y = y.values.astype(np.float32)
        y = to_categorical(y, num_classes=n_classes)
        x = data.values.astype(np.float32)
        x = scaler.fit_transform(x)
        x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.85, random_state=42)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.7, random_state=42)
        return x_train, y_train, x_val, y_val, x_test, y_test

    @staticmethod
    def plot_history(history) -> None:
        plt.figure()
        plt.plot(history.history['accuracy'], label='train_accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.draw()

    @staticmethod
    def save_results(results: pd.DataFrame, path: str, name: str) -> None:
        pass


def main():
    # get directories and data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataframe_path_mouse = dir_path + '\\data\\dataframe\\mouse'
    dataframe_path_human = dir_path + '\\data\\dataframe\\human'
    dataframe_name = 'extracted_mean_ephys_data.csv'
    data_mouse = pd.read_csv(dataframe_path_mouse + '\\' + dataframe_name)
    data_human = pd.read_csv(dataframe_path_human + '\\' + dataframe_name)

    # train on mouse data
    print("=================================================")
    print("Mouse training:")
    dnnclf = DNNClassifier(db=data_mouse, n_layers=4, weight_decay=0.01, dense_size=[27, 64, 128, 32],
                           activation_function=['swish', 'swish', 'swish', 'swish'], learning_rate=0.00005,
                           drop_rate=[0, 0.1, 0.2, 0.3], batch_size=16, n_epochs=1000, optimizer='adam')
    dnnclf.train_and_test()

    # test human data on pretrained mouse network
    print("=================================================")
    print("Human test on mouse network:")
    data_human_test = dnnclf.preprocess_data(data_human)
    _, _, _, _, x_test_human, y_test_human = dnnclf.split_train_val_test(data_human_test)
    dnnclf.test(x_test_human, y_test_human)

    # classify human data with domain adaptation on mouse network
    print("=================================================")
    print("Domain adaptation from mouse data to human data:")
    dnnclf.retrain_for_domain_adaptation(data_human, lr=0.00005, n_epochs=1000, bs=16)

    # test retrained model with mouse data again
    print("=================================================")
    print("Test retrained model with mouse data:")
    data_mouse_test = dnnclf.preprocess_data(data_mouse)
    _, _, _, _, x_test_mouse, y_test_mouse = dnnclf.split_train_val_test(data_mouse_test)
    dnnclf.test(x_test_mouse, y_test_mouse)

    # classify human data with its own complete network
    print("=================================================")
    print("Human training:")
    dnnclf = DNNClassifier(db=data_human, n_layers=4, weight_decay=0.01, dense_size=[27, 64, 128, 32],
                           activation_function=['swish', 'swish', 'swish', 'swish'], learning_rate=0.00005,
                           drop_rate=[0, 0.1, 0.2, 0.3], batch_size=16, n_epochs=1000, optimizer='adam')
    dnnclf.train_and_test()

    # test human model with mouse data
    print("=================================================")
    print("Test human model with mouse data:")
    data_mouse_test = dnnclf.preprocess_data(data_mouse)
    _, _, _, _, x_test_mouse, y_test_mouse = dnnclf.split_train_val_test(data_mouse_test)
    dnnclf.test(x_test_mouse, y_test_mouse)

    # show matplotlib graphs
    plt.show()


if __name__ == '__main__':
    main()

# TODO define different figures for each plot
