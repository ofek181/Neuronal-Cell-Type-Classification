import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from glob import glob
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop
from keras.utils import to_categorical
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from gpu_check import get_device

# get directories
dir_path = os.path.dirname(os.path.realpath(__file__))
data = pd.read_csv(dir_path + '/data/mouse/single_spike_data.csv')
results_path = dir_path + '/results/single_spike_tabular'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Cancel randomness for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
tf.random.set_seed(1)
np.random.seed(1)
random.seed(1)


callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]


class TabularAnalyzer:
    def __init__(self, n_layers: int, weight_decay: float, dense_size: list,
                 activation_function: list, learning_rate: float, drop_rate: list,
                 batch_size: int, n_epochs: int, optimizer: str = 'adam', n_classes: int = 5) -> None:
        """
        :param n_layers: number of layer in the model.
        :param weight_decay: l2 regularization values.
        :param dense_size: size of the dense layers.
        :param activation_function: activation function in each dense layer.
        :param learning_rate: learning rate during training.
        :param drop_rate: dropout rate.
        :param batch_size: batch size during training and testing.
        :param n_epochs: number of epochs during training.
        :param optimizer: optimizer used (adam, sgd or rmsprop).
        :param n_classes: number of classes in the data.
        """
        self.class_names = {}
        self.x_train, self.x_test, self.y_train, self.y_test = self.preprocess_data()
        self.n_layers = n_layers
        self.weight_decay = weight_decay
        self.dense_size = dense_size
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.drop_rate = drop_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.n_classes = n_classes
        self.model = self._create_model()

    def _create_model(self) -> Sequential:
        """
        :return: a sequential keras model.
        """
        model = Sequential()
        for i in range(self.n_layers):
            model.add(BatchNormalization())
            model.add(Dense(self.dense_size[i], activation=self.activation_function[i],
                            kernel_regularizer=l2(self.weight_decay), bias_regularizer=l2(self.weight_decay)))
            model.add(Dropout(self.drop_rate[i]))
        model.add(Dense(self.n_classes, activation='softmax'))
        return model

    def train_and_test(self) -> float:
        """
        :return: trains and tests a neural network.
        """
        # Assign optimizer
        opt = Adam(learning_rate=self.learning_rate, decay=self.learning_rate / self.n_epochs)
        if self.optimizer == 'sgd':
            opt = SGD(learning_rate=self.learning_rate, decay=self.learning_rate / self.n_epochs)
        if self.optimizer == 'rmsprop':
            opt = RMSprop(learning_rate=self.learning_rate, decay=self.learning_rate / self.n_epochs)

        # Compile model
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics="accuracy")
        # Fit model
        history = self.model.fit(self.x_train, self.y_train, epochs=self.n_epochs, batch_size=self.batch_size,
                                 validation_data=(self.x_test, self.y_test), verbose=0, callbacks=callbacks)
        # Plot history
        self.plot_history(history)

        # Test model
        accuracy = self.test(self.x_test, self.y_test)
        return accuracy

    def test(self, x_test, y_test) -> float:
        """
        :param x_test: testing data.
        :param y_test: true labels of the testing data.
        :return: loss and accuracy of the model on the testing data.
        """
        # Calculate test loss and accuracy
        predictions = self.model.predict(x_test, verbose=0)
        y_pred, y_test = np.argmax(predictions, axis=1), np.argmax(y_test, axis=1)
        accuracy = accuracy_score(y_test, y_pred)

        print('==============================================')
        print("Accuracy: " + str(accuracy))

        def reverse_labels(tup: tuple) -> list:
            return [self.class_names[x] for x in tup]
        y_true_labeled, y_pred_labeled = reverse_labels(tuple(y_test)), reverse_labels(tuple(y_pred))

        # plot confusion matrix
        matrix = confusion_matrix(y_true_labeled, y_pred_labeled)
        df_cm = pd.DataFrame(matrix, columns=np.unique(y_true_labeled), index=np.unique(y_true_labeled))
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        plt.figure(1)
        cmap = sns.cubehelix_palette(light=0.9, as_cmap=True)
        cm_normalized = df_cm.div(df_cm.sum(axis=0), axis=1)
        sns.heatmap(cm_normalized, cbar=False, annot=True, cmap=cmap, square=True, fmt='.1%', annot_kws={'size': 10})
        plt.title('Transgenic Lines FeedForward Neural Network Confusion Matrix')
        plt.tight_layout()
        plt.draw()
        return accuracy

    def preprocess_data(self) -> tuple:
        data['transgenic_line'] = pd.Categorical(data['transgenic_line'])
        self.class_names = dict(enumerate(data['transgenic_line'].cat.categories))
        n = 0
        features_tabular, labels_tabular = [], []
        db = data.dropna(axis=1, how='all')  # there is no reason for NAN values in this data
        db = db.dropna(axis=0)
        irrelevant_columns = ['transgenic_line', 'neurotransmitter', 'dendrite_type', 'reporter_status', 'layer']
        db = db.drop([x for x in irrelevant_columns if x in db.columns], axis=1, errors='ignore')
        directories = ['glutamatergic', 'htr3a', 'pvalb', 'sst', 'vip']
        for idx, directory in enumerate(directories):
            files_time = glob(dir_path + '/data/single_spike/mouse/' + directory + '/*')
            # tabular features analyzed from the raw data.
            names = [f[f.rfind('/') + 1:f.rfind('.')] for f in files_time]
            for name in names:
                value = db[db['file_name'] == name]
                tmp = value.drop('file_name', axis=1)
                features_tabular.append(tmp.values.flatten())
                labels_tabular.append(idx)
                if value.empty:  # the signal is not spiny nor aspiny
                    del features_tabular[n]
                    del labels_tabular[n]
                    n -= 1
                n += 1
        # normalize each column
        features_tabular = np.stack(features_tabular, axis=0)
        features_tabular = normalize(features_tabular, axis=0, norm='max')
        # one hot encoding
        labels_tabular = to_categorical(np.array(labels_tabular), num_classes=5)
        # split into train and test
        x_train, x_test, y_train, y_test = train_test_split(features_tabular, labels_tabular,
                                                            test_size=0.2,
                                                            random_state=7,
                                                            shuffle=True)
        return x_train, x_test, y_train, y_test

    @staticmethod
    def plot_history(history) -> None:
        """
        :param history: history of the training process.
        :return: plots the training process over the number of epochs.
        """
        plt.figure(2)
        plt.plot(history.history['accuracy'], label='train_accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.draw()

    @staticmethod
    def save_results(results: pd.DataFrame, path: str, name: str) -> None:
        pass


def grid_search(data: pd.DataFrame) -> TabularAnalyzer:
    layers = [2, 3]
    l2s = [0.01, 0.0001]
    denses = [[10, 10, 10], [12, 8, 6], [11, 7, 5]]
    activations = [['relu', 'relu', 'relu']]
    lrs = [0.001]
    drops = [[0.3, 0.3, 0.3], [0.1, 0.1, 0.1]]
    bss = [32]
    epochs = [100]
    optims = 'adam'
    for layer in layers:
        for l2 in l2s:
            for dense in denses:
                for activation in activations:
                    for lr in lrs:
                        for drop in drops:
                            for bs in bss:
                                for epoch in epochs:
                                    for optim in optims:
                                        clf = TabularAnalyzer(n_layers=layer, weight_decay=l2,
                                                              dense_size=dense, activation_function=activation,
                                                              learning_rate=lr, drop_rate=drop, batch_size=bs,
                                                              n_epochs=epoch, optimizer=optim)
                                        accuracy = clf.train_and_test()
                                        print(accuracy)
                                        # if accuracy > 0.9:
                                        #     return clf
                                        plt.close(1)
                                        plt.close(2)


# def train(data: pd.DataFrame) -> DNNClassifier:
#     """
#     :param data: data to be trained on
#     :return: a trained DNNClassifier model
#     """
#     clf = DNNClassifier(db=data, n_layers=4, weight_decay=0.0001, dense_size=[20, 256, 256, 64],
#                         activation_function=['swish', 'swish', 'swish', 'swish'], learning_rate=0.001,
#                         drop_rate=[0.5, 0.5, 0.5, 0.5], batch_size=32, n_epochs=1024, optimizer='adam')
#     clf.train_and_test()
#     return clf


def main():
    clf = grid_search(data)
    clf.model.save(filepath=results_path + '/model')
    plt.show()
    # print("==============================================")
    # print("Training:")
    # clf = train(data)
    # plt.show()


if __name__ == '__main__':
    # device = get_device()
    # with tf.device(device):
    main()
