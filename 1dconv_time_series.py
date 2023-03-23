import os
import numpy as np
import tensorflow as tf
from glob import glob
from scipy import signal
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from gpu_check import get_device
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('INFO')

# get directories
file_path = os.path.dirname(os.path.realpath(__file__))
data_path = file_path + '/data/single_spike/mouse/'


def main():
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
    features = np.array(features)
    labels = np.array(labels)
    features = tf.convert_to_tensor(features, dtype=float)
    labels = tf.convert_to_tensor(tf.one_hot(labels, 5))

    # define the model
    model = Sequential()
    model.add(Dense(150, input_shape=(150,)))
    model.add(Dense(50, input_shape=(150,)))
    model.add(Dense(20, input_shape=(150,)))
    model.add(Dense(5))

    # compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='categorical_crossentropy')

    # fit the model
    history = model.fit(features, labels, epochs=10, validation_split=0.2, shuffle=True)


if __name__ == '__main__':
    # device = get_device()
    # with tf.device(device):
    #     main()
    main()
