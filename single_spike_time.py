import os
import optuna
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from scipy import signal
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Input, MaxPooling1D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from helper_functions import calculate_metrics_multiclass

from gpu_check import get_device
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('INFO')

# get directories
file_path = os.path.dirname(os.path.realpath(__file__))
data_path = file_path + '/data/single_spike/mouse/'
results_path = file_path + '/results/single_spike_time'

plt.style.use(file_path + '/plot_style.txt')

class_names = {0: 'Glutamatergic', 1: 'Htr3a+|Vip-', 2: 'Pvalb+', 3: 'Sst+', 4: 'Vip+'}

class TimeAnalyzer:
    def __init__(self) -> None:
        self.x_train, self.x_test, self.y_train, self.y_test = self.process_data()
        self.model, self.best_model = self.create_model(), None
        self.history, self.best_history = None, None

    @staticmethod
    def process_data(test_size: float = 0.2):
        # read the data
        directories = ['glutamatergic', 'htr3a', 'pvalb', 'sst', 'vip']
        features = []
        labels = []
        for idx, directory in enumerate(directories):
            files = glob(data_path + directory + '/*')
            files = sorted(files)
            arrays = [np.load(f) for f in files]
            for array in arrays:
                if len(array) == 600:
                    array = signal.decimate(array, 4)

                features.append(array)
                labels.append(idx)

        # normalize signal between 0 and 1
        features = np.stack(features, axis=0)
        features = (features - np.min(features)) / (np.max(features) - np.min(features))

        # one hot encoding
        labels = to_categorical(np.array(labels), num_classes=5)

        # split into train and test
        return train_test_split(features, labels, test_size=test_size, random_state=42, shuffle=True)

    @staticmethod
    def create_model(n_filters: tuple = (32, 32, 32),
                     n_units: tuple = (128, 64),
                     n_conv1d_layers: int = 3,
                     n_dense_layers: int = 2,
                     input_size: int = 150,
                     kernel_size: int = 3,
                     stride_size: int = 1,
                     pool_size: int = 2,
                     n_classes: int = 5,
                     activation: str = 'relu',
                     padding: str = 'same',):
        # check that each Conv1D layer has a specified number of filters
        assert len(n_filters) >= n_conv1d_layers

        # check that each Dense layer has a specified number of neurons
        assert len(n_units) >= n_dense_layers

        # input layer
        x = Input(shape=(input_size, 1))
        inputs = x

        for layer in range(n_conv1d_layers):
            x = Conv1D(filters=n_filters[layer], kernel_size=kernel_size, strides=stride_size,
                       activation=activation, padding=padding)(x)
            x = MaxPooling1D(pool_size, padding=padding)(x)

        # flatten layer
        x = Flatten()(x)

        for layer in range(n_dense_layers):
            x = Dense(units=n_units[layer], activation=activation)(x)

        # output layer
        outputs = Dense(n_classes, activation="softmax")(x)

        return Model(inputs=inputs, outputs=outputs)

    def train(self, optimizer: str = 'adam', batch_size: int = 16,
                       epochs: int = 50):
        # compile the model
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        # create checkpoint so that the best epoch is saved
        checkpoint_filepath = '/tmp/checkpoint'
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=15)

        # compute class weight
        y_integers = np.argmax(self.y_train, axis=1)
        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_integers), y=y_integers)
        d_class_weights = dict(enumerate(class_weights))

        # fit the model
        history = self.model.fit(x=self.x_train, y=self.y_train,
                                 batch_size=batch_size, epochs=epochs,
                                 validation_data=(self.x_test, self.y_test), shuffle=True,
                                 callbacks=[checkpoint, early_stopping], class_weight=d_class_weights, verbose=0)

        # The best model weights are loaded into the model
        self.model.load_weights(checkpoint_filepath)
        return history

    def callback(self, study, trial):
        if study.best_trial == trial:
            self.best_model = self.model
            self.best_history = self.history

    def objective(self, trial):
        n_filters = trial.suggest_categorical("n_filters", ([64, 64, 64],
                                                            [128, 128, 128],
                                                            [32, 64, 128]))
        n_units = trial.suggest_categorical("n_units", ([128, 128, 128],
                                                        [64, 64, 64],
                                                        [32, 32, 32],
                                                        [128, 64, 32],
                                                        [256, 128, 64]))
        n_conv1d_layers = trial.suggest_categorical("n_conv1d_layers", (1, 2, 3))
        n_dense_layers = trial.suggest_categorical("n_dense_layers", (1, 2))
        kernel_size = trial.suggest_categorical("kernel_size", (2, 3, 4))
        stride_size = trial.suggest_categorical("stride_size", (1, 2))
        pool_size = trial.suggest_categorical("pool_size", (1, 2))
        activation = trial.suggest_categorical("activation", ('relu', 'selu', 'swish', 'sigmoid', 'tanh'))
        optimizer = trial.suggest_categorical("optimizer", ('adam', 'sgd', 'rmsprop'))
        batch_size = trial.suggest_categorical("batch_size", (16, 32, 64))
        epochs = trial.suggest_categorical("epochs", (50, 100, 200))

        self.model = self.create_model(n_filters=n_filters,
                                       n_units=n_units,
                                       n_conv1d_layers=n_conv1d_layers,
                                       n_dense_layers=n_dense_layers,
                                       kernel_size=kernel_size,
                                       stride_size=stride_size,
                                       pool_size=pool_size,
                                       activation=activation)

        self.history = self.train(optimizer=optimizer, batch_size=batch_size, epochs=epochs)
        y_pred_proba = self.model.predict(x=self.x_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        accuracy, f1, precision, recall, roc_auc = calculate_metrics_multiclass(y_true, y_pred, y_pred_proba)
        return f1

    def optimize(self, n_trials: int, n_jobs: int = 1):
        study = optuna.create_study(study_name='1dconv_time', direction='maximize')
        study.optimize(func=self.objective, n_trials=n_trials, n_jobs=n_jobs, callbacks=[self.callback])
        best_n_filters = study.best_params['n_filters']
        best_n_units = study.best_params['n_units']
        best_n_conv1d_layers = study.best_params['n_conv1d_layers']
        best_n_dense_layers = study.best_params['n_dense_layers']
        best_kernel_size = study.best_params['kernel_size']
        best_stride_size = study.best_params['stride_size']
        best_pool_size = study.best_params['pool_size']
        best_activation = study.best_params['activation']
        best_optimizer = study.best_params['optimizer']
        best_batch_size = study.best_params['batch_size']
        best_epochs = study.best_params['epochs']

        y_pred_proba = self.best_model.predict(x=self.x_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        accuracy, f1, precision, recall, roc_auc = calculate_metrics_multiclass(y_true, y_pred, y_pred_proba)

        print("********* Trial Finished *********")
        print("best_n_filters: {}".format(best_n_filters))
        print("best_n_units {}".format(best_n_units))
        print("best_n_conv1d_layers: {}".format(best_n_conv1d_layers))
        print("best_n_dense_layers: {}".format(best_n_dense_layers))
        print("best_kernel_size: {}".format(best_kernel_size))
        print("best_stride_size: {}".format(best_stride_size))
        print("best_pool_size : {}".format(best_pool_size))
        print("best_activation : {}".format(best_activation))
        print("best_optimizer: {}".format(best_optimizer))
        print("best_batch_size : {}".format(best_batch_size))
        print("best_epochs : {}".format(best_epochs))

        with open(os.path.join(results_path, 'hyperparameters.txt'), "w") as file:
            file.write(
                "best_n_filters: {}\n"
                "best_n_units {}\n"
                "best_n_conv1d_layers: {}\n"
                "best_n_dense_layers: {}\n"
                "best_kernel_size: {}\n"
                "best_stride_size: {}\n"
                "best_pool_size : {}\n"
                "best_activation : {}\n"
                "best_optimizer: {}\n"
                "best_batch_size: {}\n"
                "best_epochs: {}\n"
                "Test accuracy: {}\n"
                "Test f1: {}\n"
                "Test precision: {}\n"
                "Test recall: {}\n"
                "Test roc auc: {}".format(best_n_filters, best_n_units, best_n_conv1d_layers,
                                                 best_n_dense_layers, best_kernel_size, best_stride_size,
                                                 best_pool_size, best_activation, best_optimizer, best_batch_size,
                                                 best_epochs, accuracy, f1, precision, recall, roc_auc))

    @staticmethod
    def plot_cm(y_pred, y_true):
        def reverse_labels(tup: tuple) -> list:
            return [class_names[x] for x in tup]

        y_true_labeled, y_pred_labeled = reverse_labels(tuple(y_true)), reverse_labels(tuple(y_pred))
        matrix = confusion_matrix(y_true_labeled, y_pred_labeled)
        df_cm = pd.DataFrame(matrix, columns=np.unique(y_true_labeled), index=np.unique(y_true_labeled))
        plt.figure()
        cmap = sns.cubehelix_palette(light=0.9, as_cmap=True)
        cm_normalized = df_cm.div(df_cm.sum(axis=0), axis=1)
        sns.heatmap(cm_normalized, cbar=False, annot=True, cmap=cmap, square=True, fmt='.1%',
                    annot_kws={'size': 10})
        plt.title('CNN Raw Signal')
        plt.tight_layout()
        plt.draw()
        plt.savefig(results_path + "/confusion_matrix.png")
        plt.show()

    def save_model(self):
        """
        save the best model to path.
        """
        self.best_model.save(filepath=results_path + '/model')

    def load_model(self) -> None:
        """
        load the best model from path.
        """
        self.best_model = tf.keras.models.load_model(filepath=results_path + '/model')

    def plot_history(self) -> None:
        plt.figure(2)
        plt.plot(self.best_history.history['accuracy'])
        plt.plot(self.best_history.history['val_accuracy'])
        plt.title('Accuracy vs. Epoch')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train acc', 'val acc'], loc='upper left')
        plt.draw()
        plt.savefig(results_path + "/accuracy_epoch.png")
        plt.show()

        plt.figure(3)
        plt.plot(self.best_history.history['loss'])
        plt.plot(self.best_history.history['val_loss'])
        plt.title('Loss vs. Epoch')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train loss', 'val loss'], loc='upper left')
        plt.draw()
        plt.savefig(results_path + "/loss_epoch.png")
        plt.show()


def train_model():
    model = TimeAnalyzer()
    model.optimize(n_trials=100)
    y_pred_proba = model.best_model.predict(x=model.x_test,  verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(model.y_test, axis=1)
    accuracy, f1, precision, recall, roc_auc = calculate_metrics_multiclass(y_true, y_pred, y_pred_proba)
    print("Accuracy: {}\nF1: {}\nPrecision: {}\nRecall: {}\nAUC: {}".format(accuracy, f1, precision, recall, roc_auc))
    model.plot_cm(y_pred, y_true)
    model.plot_history()
    model.save_model()


def test_model():
    model = TimeAnalyzer()
    model.load_model()
    model.best_model.summary()
    y_pred_proba = model.best_model.predict(x=model.x_test,  verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(model.y_test, axis=1)
    accuracy, f1, precision, recall, roc_auc = calculate_metrics_multiclass(y_true, y_pred, y_pred_proba)
    print("Accuracy: {}\nF1: {}\nPrecision: {}\nRecall: {}\nAUC: {}".format(accuracy, f1, precision, recall, roc_auc))
    model.plot_cm(y_pred, y_true)


def main():
    train_model()
    test_model()


if __name__ == '__main__':
    device = get_device()
    with tf.device(device):
        main()
