import os
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
from helper_functions import calculate_metrics


dir_path = os.path.dirname(os.path.realpath(__file__))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
n_classes = 2


callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]


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

    def train_and_test(self) -> tuple:
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
        accuracy, f1, precision, recall, roc_auc = self.test(x_test, y_test)
        return accuracy, f1, precision, recall, roc_auc

    def test(self, x_test, y_test) -> tuple:
        """
        :param x_test: testing data.
        :param y_test: true labels of the testing data.
        :return: loss and accuracy of the model on the testing data.
        """
        # Calculate test loss and accuracy
        predictions = self.model.predict(x_test, verbose=0)
        y_pred, y_test = np.argmax(predictions, axis=1), np.argmax(y_test, axis=1)
        accuracy, f1, precision, recall, roc_auc = calculate_metrics(y_test, y_pred)

        print('====================================================')
        print("Accuracy: " + str(accuracy))
        print("F1 Score: " + str(f1))
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("ROC AUC: " + str(roc_auc))

        # Plot confusion matrix
        matrix = confusion_matrix(y_test, y_pred)
        plt.figure()
        label_names = ['aspiny', 'spiny']
        s = sns.heatmap(matrix / np.sum(matrix), annot=True, fmt='.2%',
                        cmap='Blues', xticklabels=label_names, yticklabels=label_names)
        s.set(xlabel='Predicted label', ylabel='True label')
        plt.draw()
        return accuracy, f1, precision, recall, roc_auc

    @staticmethod
    def preprocess_data(df) -> pd.DataFrame:
        """
        :param df: raw dataframe.
        :return: processed dataframe.
        """
        db = df.dropna(axis=1, how='all')
        db = db.dropna(axis=0)
        irrelevant_columns = ['layer', 'structure_area_abbrev', 'sampling_rate', 'mean_clipped', 'file_name']
        db = db.drop([x for x in irrelevant_columns if x in df.columns], axis=1)
        db['dendrite_type'] = pd.Categorical(db['dendrite_type'])
        db['dendrite_type'] = db['dendrite_type'].cat.codes
        return db

    @staticmethod
    def split_train_val_test(data: pd.DataFrame) -> tuple:
        """
        :param data: processed dataset.
        :return: data split into train, val and test.
        """
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
        """
        :param history: history of the training process.
        :return: plots the training process over the number of epochs.
        """
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


def train(data: pd.DataFrame) -> DNNClassifier:
    """
    :param data: data to be trained on
    :return: a trained DNNClassifier model
    """
    clf = DNNClassifier(db=data, n_layers=4, weight_decay=0.01, dense_size=[27, 64, 128, 32],
                        activation_function=['swish', 'swish', 'swish', 'swish'], learning_rate=0.00005,
                        drop_rate=[0, 0.1, 0.2, 0.3], batch_size=16, n_epochs=1024, optimizer='adam')
    clf.train_and_test()
    return clf


def main():
    # get directories
    dataframe_name = 'extracted_mean_ephys_data.csv'
    dataframe_path_mouse = dir_path + '/data/dataframe/mouse'
    dataframe_path_human = dir_path + '/data/dataframe/human'
    data_mouse = pd.read_csv(dataframe_path_mouse + '/' + dataframe_name)
    data_human = pd.read_csv(dataframe_path_human + '/' + dataframe_name)

    # train on mouse data
    print("=================================================")
    print("Mouse training:")
    dnnclf = train(data_mouse)
    # test human data on trained mouse network
    print("=================================================")
    print("Human test on mouse network:")
    human_test = dnnclf.preprocess_data(data_human)
    scaler = StandardScaler()
    y = human_test.pop('dendrite_type')
    y = y.values.astype(np.float32)
    y = to_categorical(y, num_classes=n_classes)
    x = human_test.values.astype(np.float32)
    x = scaler.fit_transform(x)
    accuracy_h, f1_h, precision_h, recall_h, roc_auc_h = dnnclf.test(x, y)

    # train on human data
    print("=================================================")
    print("Human training:")
    dnnclf = train(data_human)
    # test human data on trained mouse network
    print("=================================================")
    print("Mouse test on human network:")
    mouse_test = dnnclf.preprocess_data(data_mouse)
    scaler = StandardScaler()
    y = mouse_test.pop('dendrite_type')
    y = y.values.astype(np.float32)
    y = to_categorical(y, num_classes=n_classes)
    x = mouse_test.values.astype(np.float32)
    x = scaler.fit_transform(x)
    accuracy_m, f1_m, precision_m, recall_m, roc_auc_m = dnnclf.test(x, y)

    # show matplotlib graphs
    plt.show()


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        print("Name:", gpu.name, "  Type:", gpu.device_type)
    print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))

    cpus = tf.config.list_physical_devices('CPU')
    for cpu in cpus:
        print("Name:", cpu.name, "  Type:", cpu.device_type)

    device = input("Enter device (such as /device:GPU:0 or /device:CPU:0): ")
    with tf.device(device):
        main()
