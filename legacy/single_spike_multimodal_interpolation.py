import os
import optuna
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import tensorflow.python.keras.callbacks
from scipy.interpolate import interp1d
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Input, MaxPooling1D, concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import normalize

from gpu_check import get_device
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('INFO')

# get directories
file_path = os.path.dirname(os.path.realpath(__file__))
data_path_time = file_path + '/data/single_spike/mouse/'
data_path_fft = file_path + '/data/single_spike_fft/mouse/'
data_path_tabular = file_path + '/data/mouse/single_spike_data.csv'
results_path = file_path + '/results/single_spike_multimodal'

plt.style.use(file_path + '/plot_style.txt')

class_names = {0: 'Glutamatergic', 1: 'Htr3a+|Vip-', 2: 'Pvalb+', 3: 'Sst+', 4: 'Vip+'}


class SingleSpikeAnalyzer:
    """
    Uses a single action potential provoked by a short square stimulus to classify the neuron type.
    """
    def __init__(self) -> None:
        self.x_train_time, self. x_test_time, self.y_train_time, self.y_test_time,\
            self.x_train_fft, self.x_test_fft, self.y_train_fft, self.y_test_fft,\
            self.x_train_tab, self.x_test_tab, self.y_train_tab, self.y_test_tab = self.process_data()
        self.model, self.best_model = None, None
        self.history, self.best_history = None, None

    @staticmethod
    def process_data(test_size: float = 0.2, seed: int = 42) -> tuple:
        """
        Reads the time series data, the FFT data and tabular data, normalizes each domain and splits into train/test.
        """
        directories = ['glutamatergic', 'htr3a', 'pvalb', 'sst', 'vip']
        features_time, labels_time = [], []
        features_fft, labels_fft = [], []
        features_tabular, labels_tabular = [], []
        data_tabular = pd.read_csv(data_path_tabular)
        db = data_tabular.dropna(axis=1, how='all')  # there is no reason for NAN values in this data
        db = db.dropna(axis=0)
        irrelevant_columns = ['transgenic_line', 'neurotransmitter', 'dendrite_type', 'reporter_status', 'layer']
        db = db.drop([x for x in irrelevant_columns if x in db.columns], axis=1, errors='ignore')
        n = 0
        for idx, directory in enumerate(directories):
            # raw signal, action potential voltage vs time in milliseconds.
            files_time = glob(data_path_time + directory + '/*')
            arrays = [np.load(f) for f in files_time]
            for i, array in enumerate(arrays):
                if len(array) == 150:
                    x = np.arange(0, 150)
                    f = interp1d(x, array)
                    x_new = np.linspace(0, 149, 600)
                    array_new = f(x_new)
                    array = array_new


                features_time.append(array)
                labels_time.append(idx)

            # FFT signal of the original sequence.
            files_fft = glob(data_path_fft + directory + '/*')
            arrays = [np.load(f) for f in files_fft]
            for array in arrays:
                if len(array) == 600:  # sampled at 200khz (before 2016)
                    freq = np.fft.fftfreq(array.size, d=1. / 200000)
                    max_freq = int(freq.size / 2)
                    array = np.abs(array[0: max_freq])
                else:  # sampled at 50khz (after 2016)
                    freq = np.fft.fftfreq(array.size, d=1. / 50000)
                    max_freq = int(freq.size / 2)
                    array = np.abs(array[0: max_freq])

                    x = np.arange(0, 75)
                    f = interp1d(x, array)
                    x_new = np.linspace(0, 74, 300)
                    array_new = f(x_new)
                    array = array_new

                features_fft.append(array)
                labels_fft.append(idx)

            # tabular features analyzed from the raw data.
            names = [f[f.rfind('/') + 1:f.rfind('.')] for f in files_time]
            for name in names:
                value = db[db['file_name'] == name]
                tmp = value.drop('file_name', axis=1)
                features_tabular.append(tmp.values.flatten())
                labels_tabular.append(idx)
                if value.empty:  # the signal is not spiny nor aspiny
                    del features_time[n]
                    del labels_time[n]
                    del features_fft[n]
                    del labels_fft[n]
                    del features_tabular[n]
                    del labels_tabular[n]
                    n -= 1
                n += 1

        # normalize signal between 0 and 1
        features_time = np.stack(features_time, axis=0)
        features_time = (features_time - np.min(features_time)) / (np.max(features_time) - np.min(features_time))

        # one hot encoding
        labels_time = to_categorical(np.array(labels_time), num_classes=5)

        # split into train and test
        x_train_time, x_test_time, y_train_time, y_test_time = train_test_split(features_time,
                                                                                labels_time,
                                                                                test_size=test_size,
                                                                                random_state=seed,
                                                                                shuffle=True)

        # normalize signal between 0 and 1
        features_fft = np.stack(features_fft, axis=0)
        features_fft = (features_fft - np.min(features_fft)) / (np.max(features_fft) - np.min(features_fft))

        # one hot encoding
        labels_fft = to_categorical(np.array(labels_fft), num_classes=5)

        # split into train and test
        x_train_fft, x_test_fft, y_train_fft, y_test_fft = train_test_split(features_fft,
                                                                            labels_fft,
                                                                            test_size=test_size,
                                                                            random_state=seed,
                                                                            shuffle=True)

        # normalize each column
        features_tabular = np.stack(features_tabular, axis=0)
        features_tabular = normalize(features_tabular, axis=0, norm='max')

        # one hot encoding
        labels_tabular = to_categorical(np.array(labels_tabular), num_classes=5)

        # split into train and test
        x_train_tab, x_test_tab, y_train_tab, y_test_tab = train_test_split(features_tabular,
                                                                            labels_tabular,
                                                                            test_size=test_size,
                                                                            random_state=seed,
                                                                            shuffle=True)

        return x_train_time, x_test_time, y_train_time, y_test_time,\
            x_train_fft, x_test_fft, y_train_fft, y_test_fft,\
            x_train_tab, x_test_tab, y_train_tab, y_test_tab

    @staticmethod
    def create_model(n_filters_t: tuple = (32, 32, 32),
                     n_filters_f: tuple = (32, 32, 32),
                     n_units_t: tuple = (128, 64),
                     n_units_f: tuple = (128, 64),
                     n_units_tab: tuple = (10, 10),
                     n_conv1d_layers_t: int = 3,
                     n_conv1d_layers_f: int = 3,
                     n_dense_layers_t: int = 2,
                     n_dense_layers_f: int = 2,
                     n_dense_layers_tab: int = 2,
                     input_size_t: int = 600,
                     input_size_f: int = 300,
                     input_size_tab: int = 11,
                     kernel_size_t: int = 3,
                     kernel_size_f: int = 3,
                     stride_size_t: int = 1,
                     stride_size_f: int = 1,
                     pool_size_t: int = 2,
                     pool_size_f: int = 2,
                     concatenate_size: int = 5,
                     n_classes: int = 5,
                     activation_t: str = 'relu',
                     activation_f: str = 'relu',
                     activation_tab: str = 'relu',
                     padding: str = 'same') -> tensorflow.keras.models.Model:
        # check that each Conv1D layer has a specified number of filters
        assert len(n_filters_t) >= n_conv1d_layers_t
        assert len(n_filters_f) >= n_conv1d_layers_f

        # check that each Dense layer has a specified number of neurons
        assert len(n_units_t) >= n_dense_layers_t
        assert len(n_units_f) >= n_dense_layers_f
        assert len(n_units_tab) >= n_dense_layers_tab

        # time domain model
        x_t = Input(shape=(input_size_t, 1))
        input_t = x_t
        for layer in range(n_conv1d_layers_t):
            x_t = Conv1D(filters=n_filters_t[layer], kernel_size=kernel_size_t, strides=stride_size_t,
                         activation=activation_t, padding=padding)(x_t)
            x_t = MaxPooling1D(pool_size_t, padding=padding)(x_t)
        # flatten layer
        x_t = Flatten()(x_t)
        for layer in range(n_dense_layers_t):
            x_t = Dense(units=n_units_t[layer], activation=activation_t)(x_t)
        # output layer
        output_t = Dense(concatenate_size, activation=activation_t)(x_t)
        # model_t = Model(inputs=input_t, outputs=output_t)

        # frequency domain model
        x_f = Input(shape=(input_size_f, 1))
        input_f = x_f
        for layer in range(n_conv1d_layers_f):
            x_f = Conv1D(filters=n_filters_f[layer], kernel_size=kernel_size_f, strides=stride_size_f,
                         activation=activation_f, padding=padding)(x_f)
            x_f = MaxPooling1D(pool_size_f, padding=padding)(x_f)
        # flatten layer
        x_f = Flatten()(x_f)
        for layer in range(n_dense_layers_f):
            x_f = Dense(units=n_units_f[layer], activation=activation_f)(x_f)
        # output layer
        output_f = Dense(concatenate_size, activation=activation_f)(x_f)
        # model_f = Model(inputs=input_f, outputs=output_f)

        # tabular domain model
        x_tab = Input(shape=(input_size_tab, 1))
        input_tab = x_tab
        x_tab = Flatten()(x_tab)
        for layer in range(n_dense_layers_tab):
            x_tab = Dense(units=n_units_tab[layer], activation=activation_tab)(x_tab)
        output_tab = Dense(concatenate_size, activation=activation_f)(x_tab)
        # model_tab = Model(inputs=input_tab, outputs=output_tab)

        # combine the output of the three models
        combined = concatenate([output_t, output_f, output_tab])
        out = Dense(10, activation="relu")(combined)
        out = Dense(n_classes, activation="softmax")(out)
        model = Model(inputs=[input_t, input_f, input_tab], outputs=out)
        return model

    def train_and_test(self, optimizer: str = 'adam', batch_size: int = 16,
                       epochs: int = 50) -> tensorflow.keras.callbacks.History:
        # compile the model
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        # create checkpoint so that the best epoch is saved
        checkpoint_filepath = '/tmp/checkpoint'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        # compute class weight
        y_integers = np.argmax(self.y_train_time, axis=1)
        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_integers), y=y_integers)
        d_class_weights = dict(enumerate(class_weights))

        # fit the model
        history = self.model.fit(x=[self.x_train_time, self.x_train_fft, self.x_train_tab], y=self.y_train_time,
                                 batch_size=batch_size, epochs=epochs, validation_data=([self.x_test_time,
                                                                                         self.x_test_fft,
                                                                                         self.x_test_tab],
                                                                                        self.y_test_time),
                                 shuffle=True, callbacks=[model_checkpoint_callback],
                                 class_weight=d_class_weights, verbose=0)

        # The best model weights are loaded into the model
        self.model.load_weights(checkpoint_filepath)
        return history

    def callback(self, study, trial):
        if study.best_trial == trial:
            self.best_model = self.model
            self.best_history = self.history

    def objective(self, trial):
        n_filters_t = trial.suggest_categorical("n_filters_t", ([256, 256, 256],
                                                                [128, 128, 128],
                                                                [64, 64, 64],
                                                                [32, 64, 128]))
        n_filters_f = trial.suggest_categorical("n_filters_f", ([256, 256, 256],
                                                                [128, 128, 128],
                                                                [64, 64, 64],
                                                                [32, 64, 128]))
        n_units_t = trial.suggest_categorical("n_units_t", ([256, 128, 64],
                                                            [128, 64, 32],
                                                            [64, 32, 16]))
        n_units_f = trial.suggest_categorical("n_units_f", ([256, 128, 64],
                                                            [128, 64, 32],
                                                            [64, 32, 16]))
        n_units_tab = trial.suggest_categorical("n_units_tab", ([11, 11], [10, 10]))
        n_conv1d_layers_t = trial.suggest_categorical("n_conv1d_layers_t", (1, 2, 3))
        n_conv1d_layers_f = trial.suggest_categorical("n_conv1d_layers_f", (1, 2, 3))
        n_dense_layers_t = trial.suggest_categorical("n_dense_layers_t", (1, 2, 3))
        n_dense_layers_f = trial.suggest_categorical("n_dense_layers_f", (1, 2, 3))
        n_dense_layers_tab = trial.suggest_categorical("n_dense_layers_tab", (1, 2))
        kernel_size_t = trial.suggest_categorical("kernel_size_t", (2, 3, 4, 5, 6))
        kernel_size_f = trial.suggest_categorical("kernel_size_f", (2, 3, 4, 5, 6))
        stride_size_t = trial.suggest_categorical("stride_size_t", (1, 2, 3))
        stride_size_f = trial.suggest_categorical("stride_size_f", (1, 2, 3))
        pool_size_t = trial.suggest_categorical("pool_size_t", (1, 2))
        pool_size_f = trial.suggest_categorical("pool_size_f", (1, 2))
        activation_t = trial.suggest_categorical("activation_t", ('relu', 'selu', 'swish', 'sigmoid', 'tanh'))
        activation_f = trial.suggest_categorical("activation_f", ('relu', 'selu', 'swish', 'sigmoid', 'tanh'))
        activation_tab = trial.suggest_categorical("activation_tab", ('relu', 'selu', 'swish', 'sigmoid', 'tanh'))
        concatenate_size = trial.suggest_categorical("concatenate_size", (5, 10))
        optimizer = trial.suggest_categorical("optimizer", ('adam', 'sgd', 'rmsprop'))
        batch_size = trial.suggest_categorical("batch_size", (32, 64))
        epochs = trial.suggest_categorical("epochs", (50, 100, 150, 200))

        self.model = self.create_model(n_filters_t=n_filters_t,
                                       n_filters_f=n_filters_f,
                                       n_units_t=n_units_t,
                                       n_units_f=n_units_f,
                                       n_units_tab=n_units_tab,
                                       n_conv1d_layers_t=n_conv1d_layers_t,
                                       n_conv1d_layers_f=n_conv1d_layers_f,
                                       n_dense_layers_t=n_dense_layers_t,
                                       n_dense_layers_f=n_dense_layers_f,
                                       n_dense_layers_tab=n_dense_layers_tab,
                                       kernel_size_t=kernel_size_t,
                                       kernel_size_f=kernel_size_f,
                                       stride_size_t=stride_size_t,
                                       stride_size_f=stride_size_f,
                                       pool_size_t=pool_size_t,
                                       pool_size_f=pool_size_f,
                                       activation_t=activation_t,
                                       activation_f=activation_f,
                                       activation_tab=activation_tab,
                                       concatenate_size=concatenate_size)

        self.history = self.train_and_test(optimizer=optimizer, batch_size=batch_size, epochs=epochs)
        return max(self.history.history['val_accuracy'])

    def optimize(self, n_trials: int, n_jobs: int = 1):
        study = optuna.create_study(study_name='single_spike_multimodal', direction='maximize')
        study.optimize(func=self.objective, n_trials=n_trials, n_jobs=n_jobs, callbacks=[self.callback])
        best_n_filters_t = study.best_params['n_filters_t']
        best_n_filters_f = study.best_params['n_filters_f']
        best_n_units_t = study.best_params['n_units_t']
        best_n_units_f = study.best_params['n_units_f']
        best_n_units_tab = study.best_params['n_units_tab']
        best_n_conv1d_layers_t = study.best_params['n_conv1d_layers_t']
        best_n_conv1d_layers_f = study.best_params['n_conv1d_layers_f']
        best_n_dense_layers_t = study.best_params['n_dense_layers_t']
        best_n_dense_layers_f = study.best_params['n_dense_layers_f']
        best_n_dense_layers_tab = study.best_params['n_dense_layers_tab']
        best_kernel_size_t = study.best_params['kernel_size_t']
        best_kernel_size_f = study.best_params['kernel_size_f']
        best_stride_size_t = study.best_params['stride_size_t']
        best_stride_size_f = study.best_params['stride_size_f']
        best_pool_size_t = study.best_params['pool_size_t']
        best_pool_size_f = study.best_params['pool_size_f']
        best_activation_t = study.best_params['activation_t']
        best_activation_f = study.best_params['activation_f']
        best_activation_tab = study.best_params['activation_tab']
        best_concatenate_size = study.best_params['concatenate_size']
        best_optimizer = study.best_params['optimizer']
        best_batch_size = study.best_params['batch_size']
        best_epochs = study.best_params['epochs']

        results = self.best_model.evaluate(x=[self.x_test_time, self.x_test_fft, self.x_test_tab],
                                           y=self.y_test_time, verbose=2)

        print("********* Trial Finished *********")
        print("Time Sequence Neural Network: ")
        print("best_n_filters: {}".format(best_n_filters_t))
        print("best_n_units {}".format(best_n_units_t))
        print("best_n_conv1d_layers: {}".format(best_n_conv1d_layers_t))
        print("best_n_dense_layers: {}".format(best_n_dense_layers_t))
        print("best_kernel_size: {}".format(best_kernel_size_t))
        print("best_stride_size: {}".format(best_stride_size_t))
        print("best_pool_size : {}".format(best_pool_size_t))
        print("best_activation : {}".format(best_activation_t))
        print("FFT Neural Network: ")
        print("best_n_filters: {}".format(best_n_filters_f))
        print("best_n_units {}".format(best_n_units_f))
        print("best_n_conv1d_layers: {}".format(best_n_conv1d_layers_f))
        print("best_n_dense_layers: {}".format(best_n_dense_layers_f))
        print("best_kernel_size: {}".format(best_kernel_size_f))
        print("best_stride_size: {}".format(best_stride_size_f))
        print("best_pool_size : {}".format(best_pool_size_f))
        print("best_activation : {}".format(best_activation_f))
        print("Tabular Neural Network: ")
        print("best_n_units {}".format(best_n_units_tab))
        print("best_n_dense_layers: {}".format(best_n_dense_layers_tab))
        print("best_activation : {}".format(best_activation_tab))
        print("General Parameters: ")
        print("best_out_size: {}".format(best_concatenate_size))
        print("best_optimizer: {}".format(best_optimizer))
        print("best_batch_size : {}".format(best_batch_size))
        print("best_epochs : {}".format(best_epochs))

        with open(os.path.join(results_path, 'hyperparameters.txt'), "w") as file:
            file.write(
                "best_n_filters_t: {}\n"
                "best_n_filters_f: {}\n"
                "best_n_units_t: {}\n"
                "best_n_units_f: {}\n"
                "best_n_units_tab: {}\n"
                "best_n_conv1d_layers_t: {}\n"
                "best_n_conv1d_layers_f: {}\n"
                "best_n_dense_layers_t: {}\n"
                "best_n_dense_layers_f: {}\n"
                "best_n_dense_layers_tab: {}\n"
                "best_kernel_size_t: {}\n"
                "best_kernel_size_f: {}\n"
                "best_stride_size_t: {}\n"
                "best_stride_size_f: {}\n"
                "best_pool_size_t: {}\n"
                "best_pool_size_f: {}\n"
                "best_activation_t: {}\n"
                "best_activation_f: {}\n"
                "best_activation_tab: {}\n"
                "best_out_size: {}\n"
                "best_optimizer: {}\n"
                "best_batch_size: {}\n"
                "best_epochs: {}\n"
                "Validation loss: {}\n"
                "Validation accuracy: {}".format(best_n_filters_t, best_n_filters_t,  best_n_units_t, best_n_units_f,
                                                 best_n_units_tab, best_n_conv1d_layers_t, best_n_conv1d_layers_f,
                                                 best_n_dense_layers_t, best_n_dense_layers_f, best_n_dense_layers_tab,
                                                 best_kernel_size_t, best_kernel_size_f,
                                                 best_stride_size_t, best_stride_size_f,
                                                 best_pool_size_t, best_pool_size_f,
                                                 best_activation_t, best_activation_f, best_activation_tab,
                                                 best_concatenate_size, best_optimizer, best_batch_size,
                                                 best_epochs, results[0], results[1]))

    def plot_results(self):
        # plot the confusion matrix
        y_pred = np.argmax(self.best_model.predict([self.x_test_time, self.x_test_fft, self.x_test_tab]), axis=1)
        y_true = np.argmax(self.y_test_time, axis=1)

        def reverse_labels(tup: tuple) -> list:
            return [class_names[x] for x in tup]

        y_true_labeled, y_pred_labeled = reverse_labels(tuple(y_true)), reverse_labels(tuple(y_pred))
        matrix = confusion_matrix(y_true_labeled, y_pred_labeled)
        df_cm = pd.DataFrame(matrix, columns=np.unique(y_true_labeled), index=np.unique(y_true_labeled))
        # df_cm.index.name = 'Actual'
        # df_cm.columns.name = 'Predicted'
        plt.figure(1)
        cmap = sns.cubehelix_palette(light=0.9, as_cmap=True)
        cm_normalized = df_cm.div(df_cm.sum(axis=0), axis=1)
        sns.heatmap(cm_normalized, cbar=False, annot=True, cmap=cmap, square=True, fmt='.1%', annot_kws={'size': 10})
        plt.title('Multimodal Classification')
        plt.tight_layout()
        plt.draw()
        plt.savefig(results_path + "/confusion_matrix.png")
        plt.show()

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

    def save_model(self):
        """
        save the best model to path.
        """
        self.best_model.save(filepath=results_path + '/model')


def main():
    model = SingleSpikeAnalyzer()
    model.optimize(n_trials=100)
    model.plot_results()
    model.save_model()


if __name__ == '__main__':
    # device = get_device()
    # with tf.device(device):
    main()
