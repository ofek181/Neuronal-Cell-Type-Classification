from abc import ABC
import random
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model, Input
from keras.layers import Layer
from keras.layers import Dense, BatchNormalization, Dropout
from keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from helper_functions import calculate_metrics
import matplotlib.pyplot as plt
import seaborn as sns
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
results_path = dir_path + '/results/DANN'
model_path = os.path.join(results_path, 'model')

# Cancel randomness for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
tf.random.set_seed(0)
np.random.seed(0)
random.seed(0)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

n_classes = 2
n_domains = 2

# TODO find hyper-parameters that work well for the domain adaptation task


# Gradient Reversal Layer
class GradientReversal(Layer):
    """
        An implementation of the Gradient Reversal Layer,
        referenced from the "Domain-Adversarial Training of Neural Networks" paper,
        published in the "Journal of Machine Learning Research" 17 (2016) 1-35.
    """
    @tf.custom_gradient
    def grad_reverse(self, x) -> tuple:
        y = tf.identity(x)

        def custom_grad(dy) -> tuple:
            return tf.negative(dy) * self.lamda
        return y, custom_grad

    def __init__(self, lamda: float, **kwargs) -> None:
        super(GradientReversal, self).__init__(**kwargs)
        self.lamda = lamda

    def get_config(self):
        config = super().get_config()
        config.update({"lamda": self.lamda})
        return config

    def call(self, inputs, *args, **kwargs):
        return self.grad_reverse(inputs)


callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_l_pred_loss', patience=30, restore_best_weights=True)]


class DANNClassifier(Model, ABC):
    """
        An implementation of "Domain-Adversarial Training of Neural Networks"
        for the 'allen cell types' database,
        dealing with the domain shift between mouse neuronal cells and human neuronal cells.
    """
    def __init__(self, db: pd.DataFrame, weight_decay: float, dense_size: list,
                 activation_function: list, learning_rate: float, drop_rate: list,
                 batch_size: int, n_epochs: int, optimizer: str = 'adam', lamda: float = 1.0) -> None:
        """
        :param db: database used for training/testing.
        :param weight_decay: l2 regularization value.
        :param dense_size: number of neurons in each dense layer.
        :param activation_function: activation function used in each layer.
        :param learning_rate: learning rate of the model during training.
        :param drop_rate: dropout rate during training.
        :param batch_size: batch size used in model updating.
        :param n_epochs: number of epochs during training.
        :param optimizer: optimizer used (ie adam, sgd or rmsprop).
        :param lamda: lambda value used in the GRL during the training of the domain predictor.
        """
        super(DANNClassifier, self).__init__()
        self.wd = weight_decay
        self.lr = learning_rate
        self.dr = drop_rate
        self.af = activation_function
        self.opt = optimizer
        self._num_nodes = dense_size
        self._batch_size = batch_size
        self.n_epochs = n_epochs
        self.dp_lambda = lamda
        self.history = None
        self.data = self.preprocess_data(db)
        self.model = self._create_model()

    def _create_model(self) -> tf.keras.Model:
        """
        :return: constructed architecture of the model.
        """
        # Feature extractor
        x = Input(shape=(self.data.shape[1]-n_classes,), name='input')
        inputs = x
        for i in range(len(self._num_nodes)):
            x = BatchNormalization()(x)
            x = Dense(self._num_nodes[i], activation=self.af[i],
                      kernel_regularizer=l2(self.wd), bias_regularizer=l2(self.wd))(x)
            x = Dropout(self.dr[i])(x)

        # Label predictor
        label = Dense(n_classes, activation='softmax', name='l_pred')(x)

        # Domain predictor
        flipped_grad = GradientReversal(self.dp_lambda)(x)
        domain = Dense(n_domains, activation='softmax', name='d_pred')(flipped_grad)

        # Define model using Keras' functional API
        model = Model(inputs=inputs, outputs=[label, domain])
        return model

    def train_and_test(self) -> tuple:
        """
        :return: loss and accuracy on the test set after training.
        """
        # Split train and test
        x_train, y_train, x_val, y_val, x_test, y_test = self.split_train_val_test()

        # Assign optimizer
        opt = Adam(learning_rate=self.lr, decay=self.lr/self.n_epochs)
        if self.opt == 'sgd':
            opt = SGD(learning_rate=self.lr, decay=self.lr/self.n_epochs)
        if self.opt == 'rmsprop':
            opt = RMSprop(learning_rate=self.lr, decay=self.lr/self.n_epochs)

        # Compile model
        self.model.compile(loss={'l_pred': 'categorical_crossentropy', 'd_pred': 'categorical_crossentropy'},
                           optimizer=opt, metrics="accuracy")
        # Fit model
        self.history = self.model.fit(x_train, {'l_pred': y_train[:, 0, :], 'd_pred': y_train[:, 1, :]},
                                      validation_data=(x_val, {'l_pred': y_val[:, 0, :], 'd_pred': y_val[:, 1, :]}),
                                      epochs=self.n_epochs, batch_size=self._batch_size, callbacks=callbacks, verbose=0,
                                      shuffle=False, use_multiprocessing=False)

        # Test the model
        y_test = y_test[:, 0, :]  # get true class labels and lose domain labels during testing
        l_acc, l_f1, l_precision, l_recall, l_roc_auc, l_pred, l_true = self.test(x_test, y_test)
        return l_acc, l_f1, l_precision, l_recall, l_roc_auc

    def test(self, x_test: np.ndarray, y_test: np.ndarray) -> tuple:
        """
        :param x_test: testing set
        :param y_test: true testing set labels
        :return: loss and accuracy of the testing set.
        """
        # calculate test loss and accuracy
        predictions = self.model.predict(x_test, verbose=1)
        l_pred = predictions[0]
        l_true = np.argmax(y_test, axis=1)
        l_pred = np.argmax(l_pred, axis=1)

        l_acc, l_f1, l_precision, l_recall, l_roc_auc = calculate_metrics(l_true, l_pred)
        print("Accuracy: " + str(l_acc))
        print("F1 Score: " + str(l_f1))
        # print("Precision: " + str(l_precision))
        # print("Recall: " + str(l_recall))
        # print("ROC AUC: " + str(l_roc_auc))
        return l_acc, l_f1, l_precision, l_recall, l_roc_auc, l_pred, l_true

    def split_train_val_test(self) -> tuple:
        """
        :return: scale the data and divide into train, validation and test.
        """
        scaler = StandardScaler()
        y_label = self.data.pop('dendrite_type')
        y_label = y_label.values.astype(np.float32)
        y_label = to_categorical(y_label, num_classes=n_classes)
        y_domain = self.data.pop('organism')
        y_domain = y_domain.values.astype(np.float32)
        y_domain = to_categorical(y_domain, num_classes=n_domains)
        y = np.zeros((len(y_label), n_classes, n_domains))
        y[:, 0, :] = y_label
        y[:, 1, :] = y_domain
        x = self.data.values.astype(np.float32)
        x = scaler.fit_transform(x)
        x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.85, random_state=0, shuffle=False)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.7,
                                                            random_state=0, shuffle=False)
        return x_train, y_train, x_val, y_val, x_test, y_test

    def get_mouse_human_split_data(self, data_human_test: pd.DataFrame, data_mouse_test: pd.DataFrame) -> tuple:
        # Preprocess data
        scaler = StandardScaler()
        data_human_test = self.preprocess_data(data_human_test)
        data_human_test = data_human_test.drop('organism', axis=1)
        data_mouse_test = self.preprocess_data(data_mouse_test)
        data_mouse_test = data_mouse_test.drop('organism', axis=1)

        y_human = data_human_test.pop('dendrite_type')
        y_human = y_human.values.astype(np.float32)
        y_human = to_categorical(y_human, num_classes=n_classes)
        x_human = data_human_test.values.astype(np.float32)
        x_human = scaler.fit_transform(x_human)

        y_mouse = data_mouse_test.pop('dendrite_type')
        y_mouse = y_mouse.values.astype(np.float32)
        y_mouse = to_categorical(y_mouse, num_classes=n_classes)
        x_mouse = data_mouse_test.values.astype(np.float32)
        x_mouse = scaler.fit_transform(x_mouse)

        return x_human, y_human, x_mouse, y_mouse

    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        :param df: unprocessed dataset.
        :return: processed dataset.
        """
        db = df.dropna(axis=1, how='all')
        db = db.dropna(axis=0)
        irrelevant_columns = ['layer', 'structure_area_abbrev', 'sampling_rate', 'mean_clipped', 'file_name',
                              'mean_threshold_index', 'mean_peak_index', 'mean_trough_index', 'mean_upstroke_index',
                              'mean_downstroke_index', 'mean_fast_trough_index']
        db = db.drop([x for x in irrelevant_columns if x in df.columns], axis=1)
        db['dendrite_type'] = pd.Categorical(db['dendrite_type'])
        db['dendrite_type'] = db['dendrite_type'].cat.codes
        db['organism'] = pd.Categorical(db['organism'])
        db['organism'] = db['organism'].cat.codes
        return db

    @staticmethod
    def plot_matrix(l_pred: np.ndarray, l_true: np.ndarray, title: str) -> None:
        plt.figure()
        matrix = confusion_matrix(l_pred, l_true)
        label_names = ['aspiny', 'spiny']
        s = sns.heatmap(matrix / np.sum(matrix), annot=True, fmt='.2%',
                        cmap='Blues', xticklabels=label_names, yticklabels=label_names)
        s.set(xlabel='Predicted label', ylabel='True label')
        plt.title(title)
        name = title + '.png'
        plt.savefig(fname=os.path.join(results_path, name))

    @staticmethod
    def plot_history(history) -> None:
        """
        :param history: training history data
        :return: plots the training process.
        """
        plt.figure()
        plt.plot(history.history['l_pred_accuracy'], label='train accuracy')
        plt.plot(history.history['val_l_pred_accuracy'], label='val accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.savefig(fname=os.path.join(results_path, 'History.png'))


def get_data() -> tuple:
    """
    :return: tuple of merged mouse/human data, human testing data and mouse testing data.
    """
    # get data from directories
    dataframe_path_mouse = dir_path + '/data/dataframe/mouse'
    dataframe_path_human = dir_path + '/data/dataframe/human'
    dataframe_name = 'extracted_mean_ephys_data.csv'
    data_mouse = pd.read_csv(dataframe_path_mouse + '/' + dataframe_name)
    data_mouse['organism'] = 0
    data_mouse, data_mouse_test = train_test_split(data_mouse, test_size=0.2, random_state=0, shuffle=False)
    data_human = pd.read_csv(dataframe_path_human + '/' + dataframe_name)
    data_human['organism'] = 1
    data_human, data_human_test = train_test_split(data_human, test_size=0.2, random_state=0, shuffle=False)
    data = data_mouse.append(data_human, ignore_index=True)
    return data, data_human_test, data_mouse_test


def grid_search() -> bool:
    """
    :return: grid search over different hyperparameter permutations.
    """
    data, data_human_test, data_mouse_test = get_data()
    column_names = ["Network Architecture", "Activation Function", "Drop Rate", "Optimizer",
                    "N Epochs", "Weight Decay", "Learning Rate", "Batch Size", "Lambda",
                    "Total Accuracy", "Human Accuracy", "Human F1", "Mouse Accuracy", "Mouse F1"]
    results = pd.DataFrame(columns=column_names)
    # Hyperparameter grid search
    wds = [0.0001, 0.001, 0.01]
    dense_sizes = [[512, 256, 128, 64, 32], [32, 64, 64, 32], [256, 128, 64, 32, 16], [256, 128, 64]]
    afs = [['selu', 'selu', 'selu', 'selu', 'selu'], ['swish', 'swish', 'swish', 'swish', 'swish']]
    lrs = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    drops = [[0.4, 0.4, 0.4, 0.4, 0.4], [0.2, 0.2, 0.2, 0.2, 0.2]]
    batches = [64]
    epochs = [1024]
    optimizers = ['adam', 'sgd', 'rmsprop']
    lambdas = [0.3, 0.35, 0.42, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1, 1.2, 1.5]

    dummy = DANNClassifier(db=data, weight_decay=0, dense_size=[], activation_function=[], learning_rate=0,
                           drop_rate=[0], batch_size=0, n_epochs=0, optimizer='adam', lamda=0)
    x_human, y_human, x_mouse, y_mouse = dummy.get_mouse_human_split_data(data_human_test, data_mouse_test)

    n_run = 0
    for ds in dense_sizes:
        for af in afs:
            for drop in drops:
                for optimizer in optimizers:
                    for epoch in epochs:
                        for wd in wds:
                            for lr in lrs:
                                for batch in batches:
                                    for lamda in lambdas:
                                        print("----------------------------------------------------------------")
                                        print("Run number: " + str(n_run))
                                        print("----------------------------------------------------------------")
                                        print("Network architecture: {0}".format(ds))
                                        print("Activations: {0}, Drop rates: {1}".format(af, drop))
                                        print("Optimizer: {0}, N_epochs: {1}".format(optimizer, epoch))
                                        print("Weight decay: {0}, Learning rate: {1}".format(wd, lr))
                                        print("Batch size: {0}, Lambda: {1}".format(batch, lamda))

                                        DANN = DANNClassifier(db=data,
                                                              weight_decay=wd,
                                                              dense_size=ds,
                                                              activation_function=af,
                                                              learning_rate=lr,
                                                              drop_rate=drop,
                                                              batch_size=batch,
                                                              n_epochs=epoch,
                                                              optimizer=optimizer,
                                                              lamda=lamda)
                                        l_acc, l_f1, l_precision, l_recall, l_roc_auc = DANN.train_and_test()

                                        # Test network on test human data
                                        print('Human Test:')
                                        acc_h, f1_h, precision_h, recall_h, roc_auc_h, pred_h, true_h = DANN.test(
                                            x_human, y_human)

                                        # Test network on test mouse data
                                        print('Mouse Test:')
                                        acc_m, f1_m, precision_m, recall_m, roc_auc_m, pred_m, true_m = DANN.test(
                                            x_mouse, y_mouse)

                                        results.loc[n_run] = [ds, af, drop, optimizer,
                                                              epoch, wd, lr, batch, lamda,
                                                              l_acc, acc_h, f1_h, acc_m, f1_m]
                                        n_run += 1
                                        results.to_csv(os.path.join(results_path, 'DANN_results.csv'), index=True)

                                        if acc_h > 0.9 and acc_m > 0.86:
                                            print("hyper parameters found!")
                                            print("Results are:")
                                            print("=============================================================")
                                            print("Network architecture: {0}".format(ds))
                                            print("Activations: {0}, Drop rates: {1}".format(af, drop))
                                            print("Optimizer: {0}, N_epochs: {1}".format(optimizer, epoch))
                                            print("Weight decay: {0}, Learning rate: {1}".format(wd, lr))
                                            print("Batch size: {0}, Lambda: {1}".format(batch, lamda))
                                            print("=============================================================")
                                            DANN.model.save(filepath=model_path)
                                            print('Model Saved!')
                                            DANN.plot_history(DANN.history)
                                            DANN.plot_matrix(pred_m, true_m, 'Mouse dendrite type classification')
                                            DANN.plot_matrix(pred_h, true_h, 'Human dendrite type classification')
                                            return True
    return False


def run_best_model() -> None:
    """
        run best model based on the one saved from the hyperparameter grid search.
    """
    data, data_human_test, data_mouse_test = get_data()
    dummy = DANNClassifier(db=data, weight_decay=0, dense_size=[], activation_function=[], learning_rate=0,
                           drop_rate=[0], batch_size=0, n_epochs=0, optimizer='adam', lamda=0)
    DANN = keras.models.load_model(filepath=model_path, custom_objects={"GradientReversal": GradientReversal})
    x_human, y_human, x_mouse, y_mouse = dummy.get_mouse_human_split_data(data_human_test, data_mouse_test)
    labels = [y_human, y_mouse]
    tests = ['Human test:', 'Mouse test:']
    for idx, data in enumerate([x_human, x_mouse]):
        y_pred = DANN.predict(data, verbose=1)[0]
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(labels[idx], axis=1)
        l_acc, l_f1, l_precision, l_recall, l_roc_auc = calculate_metrics(y_true, y_pred)
        print(tests[idx])
        print("Accuracy: " + str(l_acc))
        print("F1 Score: " + str(l_f1))
        print("Precision: " + str(l_precision))
        print("Recall: " + str(l_recall))
        print("ROC AUC: " + str(l_roc_auc))
    DANN.summary()


def main():
    run_best_model()


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
