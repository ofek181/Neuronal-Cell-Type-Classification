import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications import VGG16
from keras.layers import Dense, Flatten, Conv2D, Dropout, GlobalAveragePooling2D, BatchNormalization
from keras.utils import to_categorical
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.regularizers import l2
from helper_functions import calculate_metrics
from consts import GAF_IMAGE_SIZE
dir_path = os.path.dirname(os.path.realpath(__file__))
gaf_path = dir_path + '\\data\\images\\gaf'

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")


# class CustomCallback(Callback):
#     def __init__(self, model, x_test, y_test):
#         super().__init__()
#         self.model = model
#         self.x_test = x_test
#         self.y_test = y_test
#
#     def on_epoch_end(self, epoch, logs=None):
#         y_pred = self.model.predict(self.x_test)
#         print('y predicted: ', y_pred)


class ConvNet:
    def __init__(self, weight_decay: float, dense_size: list, drop_rate: float, data_file: str, labels_file: str) -> None:
        """
        :param db: cell ephys features dataframe.
        """
        self._load_data(data_file, labels_file)
        self.weight_decay = weight_decay
        self.dense_size = dense_size
        self.drop_rate = drop_rate
        self.model = self._create_model()

    def _load_data(self, data_file: str, labels_file: str):
        x = np.load(data_file).astype(int) / 255
        y = np.load(labels_file).astype(int)
        idx_of_sparsely_spiny = np.where(y == 2)
        x = np.delete(x, idx_of_sparsely_spiny, axis=0)
        y = np.delete(y, idx_of_sparsely_spiny)
        y = to_categorical(y, num_classes=2)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x, y, train_size=0.9, random_state=1)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_train, self.y_train,
                                                                                train_size=0.78, random_state=1)

    def _create_model(self) -> Sequential:
        """
        :return: creates a CNN classifier
        """
        base_model = VGG16(input_shape=(GAF_IMAGE_SIZE, GAF_IMAGE_SIZE, 3), weights='imagenet', include_top=False)
        base_model.trainable = False

        norm = BatchNormalization()
        flatten = Flatten()

        dense1 = Dense(self.dense_size[0], activation='relu',
                       kernel_regularizer=l2(self.weight_decay),
                       bias_regularizer=l2(self.weight_decay))
        dense2 = Dense(self.dense_size[1], activation='relu',
                       kernel_regularizer=l2(self.weight_decay),
                       bias_regularizer=l2(self.weight_decay))
        drop_layer = Dropout(self.drop_rate)

        prediction = Dense(2, activation='softmax')

        model = Sequential([base_model, norm, flatten, dense1, dense2, drop_layer, prediction])
        # model.summary()

        return model

    def train_and_test(self, lr: float, n_epochs: int, batch_size: int, optim: str = 'adam', moment: float = 0) -> pd.DataFrame:
        """
        :return: results of the logistic regression classifier on the testing data.
        """
        # compile model
        if optim == 'adam':
            opt = Adam(learning_rate=lr, decay=lr/n_epochs)
        if optim == 'sgd':
            opt = SGD(learning_rate=lr, momentum=moment)
        if optim == 'rmsprop':
            opt = RMSprop(learning_rate=lr, momentum=moment)

        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        # fit model
        history = self.model.fit(self.x_train, self.y_train, epochs=n_epochs,
                                 batch_size=batch_size, validation_data=(self.x_val, self.y_val))

        # history = self.model.fit(self.x_train, self.y_train,
        #                          epochs=n_epochs, callbacks=[CustomCallback(self.model, self.x_val, self.y_val)],
        #                          batch_size=batch_size, validation_data=(self.x_val, self.y_val))

        # evaluate model
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=2)
        print(acc)

        # # plot history
        # plt.plot(history.history['accuracy'], label='accuracy')
        # plt.plot(history.history['val_accuracy'], label='val_accuracy')
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        # plt.ylim([0.5, 1])
        # plt.legend(loc='lower right')
        # plt.show()

        # accuracy, f1, precision, recall, roc_auc = calculate_metrics(validation_generator.classes, pred)
        # stats.append([accuracy, f1, precision, recall, roc_auc])
        # results = pd.DataFrame(stats, columns=['Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC AUC'])

        results = acc
        return results

    def save_results(self, results: pd.DataFrame, path: str, name: str) -> None:
        """
        :param results: results on the testing data.
        :param path: path to save file.
        :param name: name of the file.
        :return: None.
        """
        self.model.save(path)
        results.to_csv(os.path.join(path, name))


if __name__ == '__main__':
    lrs = [0.005, 0.001, 0.0001, 0.00001]
    batches = [64, 32]
    opts = ['adam', 'rmsprop', 'sgd']
    mmnts = [0.2, 0.5, 0.8]
    wds = [1, 0.5, 0.1, 0.01, 0.001]
    denses = [[4096, 2000], [4096, 1000], [1024, 512]]
    drops = [0.5, 0.3, 0.1]
    data = gaf_path + '\\images.npy'
    labels = gaf_path + '\\labels.npy'
    for lr in lrs:
        for batch in batches:
            for drop in drops:
                for mmnt in mmnts:
                    for opt in opts:
                        for wd in wds:
                            for dense in denses:
                                print("Dense Size: {}, Weight Decay: {}, Optimizer: {}, Moment: {}, Drop Rate: {}, Batch Size: {}, Learning Rate: {}".format(dense, wd, opt, mmnt, drop, batch, lr))
                                cnn = ConvNet(weight_decay=wd, dense_size=dense, drop_rate=drop, data_file=data, labels_file=labels)
                                res = cnn.train_and_test(lr=lr, n_epochs=10, batch_size=batch, optim=opt, moment=mmnt)
                                print("Accuracy: {}".format(res))


# TODO understand why the network only outputs 1 label instead of 2
