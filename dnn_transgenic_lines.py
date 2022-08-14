import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
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
from classifier import Model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from gpu_check import get_device

n_classes = 5

# get directories
dir_path = os.path.dirname(os.path.realpath(__file__))
dataframe = dir_path + '/data/dataframe/mouse/ephys_transgenic_data.csv'
data_mouse = pd.read_csv(dataframe)
results_mouse = dir_path + '/results/MLP/transgenic/mouse'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# # Cancel randomness for reproducibility
# os.environ['PYTHONHASHSEED'] = '0'
# tf.random.set_seed(1)
# np.random.seed(1)
# random.seed(1)


callbacks = [tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)]


class DNNClassifier(Model):
    def __init__(self, db: pd.DataFrame, n_layers: int, weight_decay: float, dense_size: list,
                 activation_function: list, learning_rate: float, drop_rate: list,
                 batch_size: int,  n_epochs: int, optimizer: str = 'adam') -> None:
        """
        :param db: dataframe for training and testing.
        :param n_layers: number of layer in the model.
        :param weight_decay: l2 regularization values.
        :param dense_size: size of the dense layers.
        :param activation_function: activation function in each dense layer.
        :param learning_rate: learning rate during training.
        :param drop_rate: dropout rate.
        :param batch_size: batch size during training and testing.
        :param n_epochs: number of epochs during training.
        :param optimizer: optimizer used (adam, sgd or rmsprop).
        """
        self.wd = weight_decay
        self.lr = learning_rate
        self.dr = drop_rate
        self.af = activation_function
        self.opt = optimizer
        db = self.preprocess_data(db)
        super(DNNClassifier, self).__init__(data=db, num_layers=n_layers, num_neurons=dense_size,
                                            batch_size=batch_size, n_epochs=n_epochs)

    def _create_model(self) -> Sequential:
        """
        :return: a sequential keras model.
        """
        model = Sequential()
        for i in range(self._num_layers):
            model.add(BatchNormalization())
            model.add(Dense(self._num_nodes[i], activation=self.af[i],
                            kernel_regularizer=l2(self.wd), bias_regularizer=l2(self.wd)))
            model.add(Dropout(self.dr[i]))
        model.add(Dense(n_classes, activation='softmax'))
        return model

    def train_and_test(self) -> float:
        """
        :return: trains and tests a neural network.
        """
        # Split into train, val and test
        x_train, y_train, x_val, y_val, x_test, y_test = self.split_train_val_test(self.data)

        # Assign optimizer
        opt = Adam(learning_rate=self.lr, decay=self.lr/self.n_epochs)
        if self.opt == 'sgd':
            opt = SGD(learning_rate=self.lr, decay=self.lr/self.n_epochs)
        if self.opt == 'rmsprop':
            opt = RMSprop(learning_rate=self.lr, decay=self.lr/self.n_epochs)

        # Compile model
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics="accuracy")
        # Fit model
        history = self.model.fit(x_train, y_train, epochs=self.n_epochs, batch_size=self._batch_size,
                                 validation_data=(x_val, y_val), verbose=0, callbacks=callbacks)
        # Plot history
        self.plot_history(history)

        # Test model
        accuracy = self.test(x_test, y_test)
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

        # TODO there must be a better way to do this
        label_names = ['Exc', 'Ndnf', 'Pvalb', 'Sst', 'Vip']
        y_true_labeled, y_pred_labeled, i = [], [], 0
        for _ in np.nditer(y_test):
            y_true_labeled.append(label_names[y_test[i]])
            y_pred_labeled.append(label_names[y_pred[i]])
            i += 1

        # plot confusion matrix
        matrix = confusion_matrix(y_true_labeled, y_pred_labeled)
        df_cm = pd.DataFrame(matrix, columns=np.unique(y_true_labeled), index=np.unique(y_true_labeled))
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        plt.figure()
        cmap = sns.cubehelix_palette(light=0.9, as_cmap=True)
        sns.heatmap(df_cm, cbar=False, annot=True, cmap=cmap, square=True, fmt='.0f', annot_kws={'size': 10})
        plt.title('Actual vs Predicted')
        plt.tight_layout()
        plt.draw()
        return accuracy

    @staticmethod
    def preprocess_data(df) -> pd.DataFrame:
        """
        :param df: raw dataframe.
        :return: processed dataframe.
        """
        db = df.dropna(axis=1, how='all')
        db = db.dropna(axis=0)
        irrelevant_columns = ['dendrite_type', 'layer', 'mean_clipped', 'file_name',
                              'mean_threshold_index', 'mean_peak_index', 'mean_trough_index', 'mean_upstroke_index',
                              'mean_downstroke_index', 'mean_fast_trough_index']
        db = db.drop([x for x in irrelevant_columns if x in df.columns], axis=1)
        db['transgenic_line'] = pd.Categorical(db['transgenic_line'])
        db['transgenic_line'] = db['transgenic_line'].cat.codes
        return db

    @staticmethod
    def split_train_val_test(data: pd.DataFrame) -> tuple:
        """
        :param data: processed dataset.
        :return: data split into train, val and test.
        """
        scaler = StandardScaler()
        y = data.pop('transgenic_line')
        y = y.values.astype(np.float32)
        y = to_categorical(y, num_classes=n_classes)
        x = data.values.astype(np.float32)
        x = scaler.fit_transform(x)
        x_train, x_val, y_train, y_val = train_test_split(x, y, stratify=y, train_size=0.85)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, stratify=y_train, train_size=0.7)
        return x_train, y_train, x_val, y_val, x_test, y_test

    @staticmethod
    def plot_history(history) -> None:
        """
        :param history: history of the training process.
        :return: plots the training process over the number of epochs.
        """
        plt.figure()
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


def train(data: pd.DataFrame) -> DNNClassifier:
    """
    :param data: data to be trained on
    :return: a trained DNNClassifier model
    """
    clf = DNNClassifier(db=data, n_layers=6, weight_decay=0.0001, dense_size=[20, 128, 128, 128, 64, 32],
                        activation_function=['swish', 'swish', 'swish', 'swish', 'swish', 'swish'], learning_rate=0.1,
                        drop_rate=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], batch_size=64, n_epochs=1024, optimizer='adam')
    clf.train_and_test()
    return clf


def main():
    # train on mouse data
    print("==============================================")
    print("Training:")
    dnnclf = train(data_mouse)
    plt.show()


if __name__ == '__main__':
    device = get_device()
    with tf.device(device):
        main()
