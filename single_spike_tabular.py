import warnings
warnings.filterwarnings('ignore')
import os
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
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from gpu_check import get_device
from copy import deepcopy
from helper_functions import calculate_metrics_multiclass

# get directories
dir_path = os.path.dirname(os.path.realpath(__file__))
data = pd.read_csv(dir_path + '/data/mouse/single_spike_data.csv')
results_path = dir_path + '/results/single_spike_tabular'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
        self.model, self.best_model = self._create_model(), None
        self.history = None

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

    def train(self):
        """
        :return: trains and tests a neural network.
        """
        # Assign optimizer
        opt = Adam(learning_rate=self.learning_rate, decay=self.learning_rate / self.n_epochs)
        if self.optimizer == 'sgd':
            opt = SGD(learning_rate=self.learning_rate, decay=self.learning_rate / self.n_epochs)
        if self.optimizer == 'rmsprop':
            opt = RMSprop(learning_rate=self.learning_rate, decay=self.learning_rate / self.n_epochs)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001,
                                                          patience=10, restore_best_weights=True)

        # Compile model
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics="accuracy")
        # Fit model
        self.history = self.model.fit(self.x_train, self.y_train, epochs=self.n_epochs, batch_size=self.batch_size,
                                      validation_data=(self.x_test, self.y_test), verbose=0, callbacks=early_stopping)

    def test(self):
        # calculate test loss and accuracy
        predictions = self.model.predict(self.x_test, verbose=0)
        y_pred, y_test = np.argmax(predictions, axis=1), np.argmax(self.y_test, axis=1)

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
        plt.title('Single spike tabular classification')
        plt.tight_layout()
        plt.savefig(results_path + "/confusion_matrix.png")
        plt.draw()

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
            files = glob(dir_path + '/data/single_spike/mouse/' + directory + '/*')
            files = sorted(files)
            # tabular features analyzed from the raw data.
            names = [f[f.rfind('/') + 1:f.rfind('.')] for f in files]
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
                                                            random_state=42,
                                                            shuffle=True)
        return x_train, x_test, y_train, y_test

    def plot_history(self) -> None:
        plt.figure(2)
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Accuracy vs. Epoch')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train acc', 'val acc'], loc='upper left')
        plt.draw()
        plt.savefig(results_path + "/accuracy_epoch.png")
        plt.show()

        plt.figure(3)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Loss vs. Epoch')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train loss', 'val loss'], loc='upper left')
        plt.draw()
        plt.savefig(results_path + "/loss_epoch.png")
        plt.show()

    def save_model(self):
        """
        save the best model to path.
        """
        self.model.save(filepath=results_path + '/model')

    def load_model(self) -> None:
        """
        load the best model from path.
        """
        self.best_model = tf.keras.models.load_model(filepath=results_path + '/model')


def grid_search() -> TabularAnalyzer:
    layers = [2, 3, 4]
    l2s = [0.001, 0.01]
    denses = [[10, 10, 10, 10], [11, 8, 6, 6], [11, 8, 7, 5]]
    activations = [['relu', 'relu', 'relu', 'relu'], ['selu', 'selu', 'selu', 'selu']]
    lrs = [0.001]
    drops = [[0.3, 0.3, 0.3, 0.3], [0.1, 0.1, 0.1, 0.1]]
    bss = [32]
    epochs = [100]
    optims = ['adam', 'sgd', 'rmsprop']
    best_f1 = 0
    best_clf = TabularAnalyzer(n_layers=0, weight_decay=0,
                               dense_size=[], activation_function=[],
                               learning_rate=0, drop_rate=[], batch_size=0,
                               n_epochs=0, optimizer='adam')
    n = len(layers) * len(l2s) * len(denses) * len(activations) * len(lrs) \
        * len(drops) * len(bss) * len(epochs) * len(optims)
    i = 0
    print("Number of loops: {}".format(n))
    for layer in layers:
        for reg in l2s:
            for dense in denses:
                for activation in activations:
                    for lr in lrs:
                        for drop in drops:
                            for bs in bss:
                                for epoch in epochs:
                                    for optim in optims:
                                        i += 1
                                        print("Iteration: {}, out of {}".format(i, n))
                                        clf = TabularAnalyzer(n_layers=layer, weight_decay=reg,
                                                              dense_size=dense, activation_function=activation,
                                                              learning_rate=lr, drop_rate=drop, batch_size=bs,
                                                              n_epochs=epoch, optimizer=optim)
                                        clf.train()
                                        y_prob = clf.model.predict(x=clf.x_test, verbose=0)
                                        y_pred = np.argmax(y_prob, axis=1)
                                        y_true = np.argmax(clf.y_test, axis=1)
                                        accuracy, f1, precision, recall, roc_auc = calculate_metrics_multiclass(y_true,
                                                                                                                y_pred,
                                                                                                                y_prob)
                                        if f1 > best_f1:
                                            best_clf = deepcopy(clf)
                                            best_f1 = f1
                                        plt.close(1)
                                        plt.close(2)
    return best_clf


def train_model():
    clf = grid_search()
    clf.model.summary()
    clf.plot_history()
    clf.test()
    y_pred_proba = clf.model.predict(x=clf.x_test,  verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(clf.y_test, axis=1)
    accuracy, f1, precision, recall, roc_auc = calculate_metrics_multiclass(y_true, y_pred, y_pred_proba)
    print("Accuracy: {}\nF1: {}\nPrecision: {}\nRecall: {}\nAUC: {}".format(accuracy, f1, precision, recall, roc_auc))
    clf.save_model()
    plt.show()


def test_model():
    clf_loaded = TabularAnalyzer(n_layers=0, weight_decay=0,
                                 dense_size=[], activation_function=[],
                                 learning_rate=0, drop_rate=[], batch_size=0,
                                 n_epochs=0, optimizer='adam')
    clf_loaded.load_model()
    y_pred_proba = clf_loaded.best_model.predict(x=clf_loaded.x_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(clf_loaded.y_test, axis=1)
    accuracy, f1, precision, recall, roc_auc = calculate_metrics_multiclass(y_true, y_pred, y_pred_proba)
    print("Accuracy: {}\nF1: {}\nPrecision: {}\nRecall: {}\nAUC: {}".format(accuracy, f1, precision, recall, roc_auc))


def main():
    train_model()
    test_model()


if __name__ == '__main__':
    device = get_device()
    with tf.device(device):
        main()
