import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model, Input
from keras.layers import Layer
from keras.layers import Dense, Flatten, BatchNormalization, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam, SGD, RMSprop
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import os
import matplotlib.pyplot as plt

n_classes = 2
n_domains = 2


# Gradient Reversal Layer
class GradientReversal(Layer):
    @tf.custom_gradient
    def grad_reverse(self, x):
        y = tf.identity(x)

        def custom_grad(dy):
            return tf.negative(dy)
        return y, custom_grad

    def __init__(self, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        return self.grad_reverse(inputs)


class DANNClassifier(Model):
    def __init__(self, db: pd.DataFrame, n_layers: int, weight_decay: float, dense_size: list,
                 activation_function: list, learning_rate: float, drop_rate: list,
                 batch_size: int,  n_epochs: int, optimizer: str = 'adam', lamda: float = 1.0) -> None:
        super(DANNClassifier, self).__init__()
        self.wd = weight_decay
        self.lr = learning_rate
        self.dr = drop_rate
        self.af = activation_function
        self.opt = optimizer
        self.lamda = lamda
        self._num_layers = n_layers
        self._num_nodes = dense_size
        self._batch_size = batch_size
        self.n_epochs = n_epochs
        self.data = self.preprocess_data(db)
        self.model = self._create_model()

    def _create_model(self):
        # Feature extractor
        n_label_columns = 2
        inputs = Input(shape=(self.data.shape[1]-n_label_columns,), name='input')
        x = BatchNormalization()(inputs)
        x = Dense(self._num_nodes[0], activation=self.af[0],
                  kernel_regularizer=l2(self.wd), bias_regularizer=l2(self.wd))(x)
        x = Dropout(self.dr[0])(x)
        for i in range(1, self._num_layers):
            x = BatchNormalization()(x)
            x = Dense(self._num_nodes[i], activation=self.af[i],
                      kernel_regularizer=l2(self.wd), bias_regularizer=l2(self.wd))(x)
            x = Dropout(self.dr[i])(x)

        # Label predictor
        label = Dense(n_classes, activation='softmax', name='l_pred')(x)

        # Domain predictor
        flipped_grad = GradientReversal()(x)
        domain = Dense(n_domains, activation='softmax', name='d_pred')(flipped_grad)

        # Define model using Keras' functional API
        model = Model(inputs=inputs, outputs=[label, domain])

        return model

    def train_and_test(self) -> pd.DataFrame:
        # Split for train and test
        # x_train, y_train_domain, y_train_label, x_val, y_val_domain,\
        # y_val_label, x_test, y_test_domain, y_test_label = self.split_train_val_test(self.data)
        x_train, y_train, x_val, y_val, x_test, y_test = self.split_train_val_test()

        # Get optimizer
        opt = Adam(learning_rate=self.lr, decay=self.lr/self.n_epochs)
        if self.opt == 'sgd':
            opt = SGD(learning_rate=self.lr, decay=self.lr/self.n_epochs)
        if self.opt == 'rmsprop':
            opt = RMSprop(learning_rate=self.lr, decay=self.lr/self.n_epochs)

        # Compile model
        self.model.compile(loss={'l_pred': 'categorical_crossentropy','d_pred': 'categorical_crossentropy'},
                           optimizer=opt, metrics="accuracy")
        # fit model
        history = self.model.fit(x_train, {'l_pred': y_train[:, 0, :], 'd_pred': y_train[:, 1, :]},
                                 validation_data=(x_val, {'l_pred': y_val[:, 0, :], 'd_pred': y_val[:, 1, :]}),
                                 epochs=self.n_epochs, batch_size=self._batch_size, verbose=1)
        # plot history
        self.plot_history(history)

        # test the model
        loss, acc = self.test(x_test, y_test)

        # print summary
        # print('--------------------------------------------------------------')
        # print("DNN Summary: ")
        # self.model.summary()
        return acc

    def test(self, x_test, y_test) -> tuple:
        # calculate test loss and accuracy
        cce = tf.keras.losses.CategoricalCrossentropy()
        predictions = self.model.predict(x_test)
        l_pred = predictions[0]
        d_pred = predictions[1]
        l_true = np.argmax(y_test[:, 0, :], axis=1)
        d_true = np.argmax(y_test[:, 1, :], axis=1)
        loss = cce(y_test[:, 0, :], l_pred).numpy()
        l_pred = np.argmax(l_pred, axis=1)
        d_pred = np.argmax(d_pred, axis=1)
        l_acc = accuracy_score(l_true, l_pred)
        d_acc = accuracy_score(d_true, d_pred)
        print('====================================================')
        print("Label Accuracy: " + str(l_acc))
        print("Domain Accuracy: " + str(d_acc))
        # # plot confusion matrix
        # y_pred = self.model.predict(x_test)
        # matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
        # plt.figure()
        # label_names = ['aspiny', 'spiny']
        # s = sns.heatmap(matrix / np.sum(matrix), annot=True, fmt='.2%',
        #             cmap='Blues', xticklabels=label_names, yticklabels=label_names)
        # s.set(xlabel='Predicted label', ylabel='True label')
        # plt.draw()
        return loss, l_acc

    @staticmethod
    def _create_label_predictor() -> list:
        label_predictor = [Dense(units=6, activation='relu'),
                           Dense(units=n_classes, activation='softmax')]
        return label_predictor

    @staticmethod
    def _create_domain_predictor() -> list:
        domain_predictor = [Dense(units=6, activation='relu'),
                            Dense(units=n_domains, activation='softmax')]
        return domain_predictor

    @staticmethod
    def preprocess_data(df):
        db = df.dropna(axis=1, how='all')
        db = db.dropna(axis=0)
        irrelevant_columns = ['layer', 'structure_area_abbrev', 'sampling_rate', 'mean_clipped', 'file_name']
        db = db.drop([x for x in irrelevant_columns if x in df.columns], axis=1)
        db['dendrite_type'] = pd.Categorical(db['dendrite_type'])
        db['dendrite_type'] = db['dendrite_type'].cat.codes
        db['organism'] = pd.Categorical(db['organism'])
        db['organism'] = db['organism'].cat.codes
        return db

    def split_train_val_test(self) -> tuple:
        scaler = StandardScaler()

        y_label = self.data.pop('dendrite_type')
        y_label = y_label.values.astype(np.float32)
        y_label = to_categorical(y_label, num_classes=n_classes)

        y_domain = self.data.pop('organism')
        y_domain = y_domain.values.astype(np.float32)
        y_domain = to_categorical(y_domain, num_classes=n_domains)

        y = np.zeros((len(y_label), n_classes, n_domains))
        y[:, 0, :] = y_label
        y[:, 1, :] = y_domain

        x = self.data.values.astype(np.float32)
        x = scaler.fit_transform(x)

        x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.85, random_state=42)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.7, random_state=42)
        return x_train, y_train, x_val, y_val, x_test, y_test

    @staticmethod
    def plot_history(history) -> None:
        plt.figure()
        plt.plot(history.history['l_pred_accuracy'], label='label train accuracy')
        plt.plot(history.history['val_l_pred_accuracy'], label='label val accuracy')
        plt.plot(history.history['d_pred_accuracy'], label='domain train accuracy')
        plt.plot(history.history['val_d_pred_accuracy'], label='domain val accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.draw()


def get_data() -> pd.DataFrame:
    # get directories and data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataframe_path_mouse = dir_path + '/data/dataframe/mouse'
    dataframe_path_human = dir_path + '/data/dataframe/human'
    dataframe_name = 'extracted_mean_ephys_data.csv'
    data_mouse = pd.read_csv(dataframe_path_mouse + '/' + dataframe_name)
    data_mouse['organism'] = 0
    data_human = pd.read_csv(dataframe_path_human + '/' + dataframe_name)
    data_human['organism'] = 1
    data = data_mouse.append(data_human, ignore_index=True)
    return data


def main():
    data = get_data()
    DANN = DANNClassifier(db=data, n_layers=4, weight_decay=0.1, dense_size=[27, 64, 128, 32],
                          activation_function=['swish', 'swish', 'swish', 'swish'], learning_rate=0.001,
                          drop_rate=[0, 0.2, 0.2, 0.2], batch_size=16, n_epochs=100, optimizer='adam')
    DANN.train_and_test()

    # show matplotlib graphs
    plt.show()


if __name__ == '__main__':
    main()
