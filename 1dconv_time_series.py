import os
import numpy as np
import tensorflow as tf
from glob import glob
from scipy import signal
from tensorflow.keras.layers import Concatenate, Conv1D, Dense, Flatten, Input, MaxPooling1D, Reshape
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

from gpu_check import get_device
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('INFO')

# get directories
file_path = os.path.dirname(os.path.realpath(__file__))
data_path = file_path + '/data/single_spike/mouse/'


class TimeAnalyzer:
    def __init__(self) -> None:
        # read the data
        directories = ['glutamatergic', 'htr3a', 'pvalb', 'sst', 'vip']
        features = []
        labels = []
        for idx, directory in enumerate(directories):
            files = glob(data_path + directory + '/*')
            arrays = [np.load(f) for f in files]
            for array in arrays:
                if len(array) == 600:
                    array = signal.decimate(array, 4)

                features.append(array)
                labels.append(idx)

        # convert to a tensor
        features = np.array(features)
        labels = to_categorical(np.array(labels), num_classes=5)

        # split into train, validation and test
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(features,
                                                                                labels,
                                                                                test_size=0.2,
                                                                                random_state=42,
                                                                                shuffle=True)
        self.model = self._create_model()

    @staticmethod
    def _create_model(n_filters: tuple = (32, 32, 32),
                      n_units: tuple = (128, 64),
                      n_conv1d_layers: int = 3,
                      n_dense_layers: int = 2,
                      input_size: int = 150,
                      kernel_size: int = 3,
                      pool_size: int = 2,
                      n_classes: int = 5,
                      activation: str = 'relu',
                      padding: str = 'valid',):
        # check that each Conv1D layer has a specified number of filters
        assert len(n_filters) == n_conv1d_layers

        # check that each Dense layer has a specified number of neurons
        assert len(n_units) == n_dense_layers

        # input layer
        x = Input(shape=(input_size, 1))
        inputs = x

        for layer in range(n_conv1d_layers):
            x = Conv1D(filters=n_filters[layer], kernel_size=kernel_size, activation=activation, padding=padding)(x)
            x = MaxPooling1D(pool_size)(x)

        # flatten layer
        x = Flatten()(x)

        for layer in range(n_dense_layers):
            x = Dense(units=n_units[layer], activation=activation)(x)

        # output layer
        outputs = Dense(n_classes, activation="softmax")(x)

        return Model(inputs=inputs, outputs=outputs)


def main():
    model = TimeAnalyzer()
    # read the data
    directories = ['glutamatergic', 'htr3a', 'pvalb', 'sst', 'vip']
    features = []
    labels = []
    for idx, directory in enumerate(directories):
        files = glob(data_path + directory + '/*')
        arrays = [np.load(f) for f in files]
        for array in arrays:
            if len(array) == 600:
                array = signal.decimate(array, 4)

            features.append(array)
            labels.append(idx)

    # convert to a tensor
    features = np.array(features)
    labels = to_categorical(np.array(labels), num_classes=5)
    # features = tf.convert_to_tensor(features, dtype=float)
    # labels = tf.convert_to_tensor(tf.one_hot(labels, 5))

    # split into train, validation and test
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, shuffle=True)

    # # define the model
    # model = create_model()
    #
    # # compile the model
    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    #
    # # fit the model
    # history = model.fit(x=x_train,
    #                     y=y_train,
    #                     batch_size=8,
    #                     epochs=10,
    #                     validation_data=(x_test, y_test),
    #                     shuffle=True)


if __name__ == '__main__':
    # device = get_device()
    # with tf.device(device):
    #     main()
    main()
