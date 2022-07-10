import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, LayerNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.python.client import device_lib
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.regularizers import l2
from helper_functions import calculate_metrics
from consts import GAF_IMAGE_SIZE
dir_path = os.path.dirname(os.path.realpath(__file__))
npy_path = dir_path + '/data/images/npy/mouse'

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# TODO there might be something wrong with the data

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

callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]


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

        norm = LayerNormalization()
        gap = GlobalAveragePooling2D()
        dense1 = Dense(self.dense_size[0], activation='relu',
                       kernel_regularizer=l2(self.weight_decay),
                       bias_regularizer=l2(self.weight_decay))
        drop_layer = Dropout(self.drop_rate)
        dense2 = Dense(self.dense_size[1], activation='relu',
                       kernel_regularizer=l2(self.weight_decay),
                       bias_regularizer=l2(self.weight_decay))
        prediction = Dense(2, activation='softmax')

        model = Sequential([base_model, norm, gap, dense1, drop_layer, dense2, prediction])
        # model.summary()

        return model

    def train_and_test(self, lr: float, n_epochs: int, batch_size: int, optim: str = 'adam', moment: float = 0) -> pd.DataFrame:
        """
        :return: results of the logistic regression classifier on the testing data.
        """
        # compile model
        if optim == 'adam':
            # opt = Adam(learning_rate=lr, decay=lr/n_epochs)
            opt = Adam(learning_rate=lr)
        if optim == 'sgd':
            opt = SGD(learning_rate=lr, momentum=moment)
        if optim == 'rmsprop':
            opt = RMSprop(learning_rate=lr, momentum=moment)

        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        # fit model

        history = self.model.fit(self.x_train, self.y_train,
                                 epochs=n_epochs, callbacks=callbacks,
                                 batch_size=batch_size, validation_data=(self.x_val, self.y_val))

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

    @staticmethod
    def save_results(results: pd.DataFrame, acc: float, path: str, n_run: int,
                     lr: float, batch_size: int, optimizer: str, drop_rate: float,
                     moment: float, wd: float, dense_size: list, epch: int) -> None:
        results.loc[n_run] = [acc, lr, batch_size, drop_rate, moment, optimizer, wd, dense_size, epch]
        results.to_csv(os.path.join(path, 'CNN_results.csv'), index=True)


def main():
    results_path = dir_path + '/results/DANN'
    column_names = ["Accuracy", "Learning rate", "Batch Size", "Drop Rate", "Moment",
                    "Optimizer", "Weight Decay", "Dense Size", "N Epochs"]
    results = pd.DataFrame(columns=column_names)
    lrs = [0.001, 0.0005, 0.0001, 0.00001]
    batches = [8]
    opts = ['adam', 'rmsprop', 'sgd']
    mmnts = [0.2, 0.5, 0.8]
    wds = [1.0, 0.5, 0.1, 0.01, 0.001]
    denses = [[1000, 100], [1000, 500], [500, 100], [64, 32]]
    drops = [0.5, 0.3, 0.1]
    n_epochs = [256, 512, 1024]
    data = npy_path + '/images.npy'
    labels = npy_path + '/labels.npy'
    run = 0
    for lr in lrs:
        for batch in batches:
            for drop in drops:
                for mmnt in mmnts:
                    for opt in opts:
                        for wd in wds:
                            for dense in denses:
                                for epoch in n_epochs:
                                    print("----------------------------------------------------------------")
                                    print("Run number: " + str(run))
                                    print("----------------------------------------------------------------")
                                    run += 1
                                    print("Dense Size: {}, Weight Decay: {}, Optimizer: {},"
                                          " Moment: {}, Drop Rate: {}, Batch Size: {}, Learning Rate: {}".
                                          format(dense, wd, opt, mmnt, drop, batch, lr))
                                    cnn = ConvNet(weight_decay=wd, dense_size=dense, drop_rate=drop,
                                                  data_file=data, labels_file=labels)
                                    acc = cnn.train_and_test(lr=lr, n_epochs=epoch, batch_size=batch,
                                                             optim=opt, moment=mmnt)
                                    print("Accuracy: {}".format(acc))
                                    ConvNet.save_results(results=results, acc=acc, path=results_path, n_run=run,
                                                         lr=lr, batch_size=batch, optimizer=opt,
                                                         moment=mmnt, wd=wd, dense_size=dense,
                                                         drop_rate=drop, epch=epoch)


if __name__ == '__main__':
    id = input("Enter device: ")
    try:
        with tf.device('/device:GPU:' + str(id)):
            main()
    except RuntimeError as e:
        print(e)

