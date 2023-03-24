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


def create_model(kernel_size: int = 3, activation: str = 'relu', padding: str = 'valid'):
    # input layer
    inp = Input(shape=(150, 1))

    # conv layer 1
    x = Conv1D(filters=32, kernel_size=kernel_size, activation=activation, padding=padding)(inp)

    # max pooling 1
    x = MaxPooling1D(2)(x)

    # conv layer 2
    x = Conv1D(filters=32, kernel_size=kernel_size, activation=activation, padding=padding)(x)

    # max pooling 2
    x = MaxPooling1D(2)(x)

    # conv layer 3
    x = Conv1D(filters=16, kernel_size=kernel_size, activation=activation, padding=padding)(x)

    # max pooling 3
    x = MaxPooling1D(2)(x)

    # flatten layer
    x = Flatten()(x)

    # fully connected layer 1
    x = Dense(100, activation=activation)(x)

    # fully connected layer 2
    x = Dense(50, activation=activation)(x)

    # output layer
    out = Dense(5, activation="softmax")(x)

    return Model(inputs=inp, outputs=out)


def main():
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

    # define the model
    model = create_model()

    # compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # fit the model
    history = model.fit(x=x_train,
                        y=y_train,
                        batch_size=8,
                        epochs=10,
                        validation_data=(x_test, y_test),
                        shuffle=True)


if __name__ == '__main__':
    # device = get_device()
    # with tf.device(device):
    #     main()
    main()
