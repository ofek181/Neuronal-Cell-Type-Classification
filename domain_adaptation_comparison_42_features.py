import random
import keras
import optuna
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from helper_functions import calculate_metrics
from gpu_check import get_device
import warnings
warnings.filterwarnings('ignore')

global best_model, model

# get directories
filepath = os.path.dirname(os.path.realpath(__file__))
results_path = filepath + '/results/dann'
model_path = results_path + '/nn_model_42_features'
mouse_data = pd.read_csv(filepath + '/data/mouse/ephys_data.csv')
human_data = pd.read_csv(filepath + '/data/human/ephys_data.csv')
mouse_data['organism'] = 0
human_data['organism'] = 1

# cancel randomness for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
tf.random.set_seed(0)
np.random.seed(0)
random.seed(0)

# configurations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
n_classes = 2

plt.style.use(filepath + '/plot_style.txt')


callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]


class NeuralNetwork(Model):
    def __init__(self, db: pd.DataFrame, weight_decay: float, dense_size: list,
                 activation_function: list, learning_rate: float, drop_rate: list,
                 batch_size: int, n_epochs: int, optimizer: str = 'adam') -> None:
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
        super(NeuralNetwork, self).__init__()
        self.wd = weight_decay
        self.lr = learning_rate
        self.dr = drop_rate
        self.af = activation_function
        self.opt = optimizer
        self._num_nodes = dense_size
        self._batch_size = batch_size
        self.n_epochs = n_epochs
        self.history = None
        self.data = self.preprocess_data(db)
        self.model = self._create_model()

    def _create_model(self) -> tf.keras.Model:
        """
        :return: constructed architecture of the model.
        """
        # Feature extractor
        x = Input(shape=(self.data.shape[1] - 1,), name='input')
        inputs = x
        for i in range(len(self._num_nodes)):
            x = BatchNormalization()(x)
            x = Dense(self._num_nodes[i], activation=self.af[i],
                      kernel_regularizer=l2(self.wd), bias_regularizer=l2(self.wd))(x)
            x = Dropout(self.dr[i])(x)

        label = Dense(n_classes, activation='softmax')(x)

        # Define model using Keras' functional API
        nn_model = Model(inputs=inputs, outputs=label)
        return nn_model

    def train_and_test(self) -> tuple:
        """
        :return: loss and accuracy on the test set after training.
        """
        # Split train and test
        x_train, y_train, x_val, y_val, x_test, y_test = self.split_train_val_test()

        # Assign optimizer
        opt = Adam(learning_rate=self.lr)
        if self.opt == 'sgd':
            opt = SGD(learning_rate=self.lr)
        if self.opt == 'rmsprop':
            opt = RMSprop(learning_rate=self.lr)

        # Compile model
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics="accuracy")
        # Fit model
        self.history = self.model.fit(x_train, y_train, validation_data=(x_val, y_val),
                                      epochs=self.n_epochs, batch_size=self._batch_size, callbacks=callbacks, verbose=0,
                                      shuffle=False, use_multiprocessing=False)
        # Test the model
        acc, f1, precision, recall, roc_auc, pred, true = self.test(x_test, y_test)
        return acc, f1, precision, recall, roc_auc

    def test(self, x_test: np.ndarray, y_test: np.ndarray) -> tuple:
        """
        :param x_test: testing set
        :param y_test: true testing set labels
        :return: loss and accuracy of the testing set.
        """
        # calculate test loss and accuracy
        true = np.argmax(y_test, axis=1)
        pred = np.argmax(self.model.predict(x_test, verbose=0), axis=1)

        acc, f1, precision, recall, roc_auc = calculate_metrics(true, pred)
        return acc, f1, precision, recall, roc_auc, pred, true

    def split_train_val_test(self) -> tuple:
        """
        :return: scale the data and divide into train, validation and test.
        """
        scaler = StandardScaler()
        y = self.data.pop('dendrite_type')
        y = y.values.astype(np.float32)
        y = to_categorical(y, num_classes=n_classes)
        x = self.data.values.astype(np.float32)
        x = scaler.fit_transform(x)
        x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.85, random_state=42, shuffle=False)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.7,
                                                            random_state=0, shuffle=False)
        return x_train, y_train, x_val, y_val, x_test, y_test

    def get_mouse_human_split_data(self, data_human_test: pd.DataFrame, data_mouse_test: pd.DataFrame) -> tuple:
        """
        :param data_human_test: testing data for human cells.
        :param data_mouse_test: testing data for mouse cells.
        :return: processed data for human and mouse.
        """
        # Preprocess data
        scaler = StandardScaler()
        data_human_test = self.preprocess_data(data_human_test)
        data_mouse_test = self.preprocess_data(data_mouse_test)

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
        irrelevant_columns = ['transgenic_line', 'neurotransmitter',
                              'reporter_status', 'layer', 'file_name']
        db = db.drop([x for x in irrelevant_columns if x in df.columns], axis=1, errors='ignore')
        db['dendrite_type'] = pd.Categorical(db['dendrite_type'])
        db['dendrite_type'] = db['dendrite_type'].cat.codes
        db['organism'] = pd.Categorical(db['organism'])
        db['organism'] = db['organism'].cat.codes
        return db

    @staticmethod
    def plot_matrix(l_pred: np.ndarray, l_true: np.ndarray, title: str) -> None:
        """
        :param l_pred: predicted labels.
        :param l_true: true labels.
        :param title: title of the plot.
        :return: confusion matrix plot for the classification task.
        """
        plt.figure()
        matrix = confusion_matrix(l_pred, l_true)
        label_names = ['aspiny', 'spiny']
        s = sns.heatmap(matrix / np.sum(matrix), annot=True, fmt='.2%',
                        cmap='Blues', xticklabels=label_names, yticklabels=label_names)
        s.set(xlabel='Predicted label', ylabel='True label')
        plt.title(title)
        name = title + '.png'
        plt.savefig(fname=os.path.join(results_path, name))

    def plot_history(self) -> None:
        """
        :param history: training history data
        :return: plots the training process.
        """
        plt.figure()
        plt.plot(self.history.history['accuracy'], label='train accuracy')
        plt.plot(self.history.history['val_accuracy'], label='val accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.savefig(fname=os.path.join(results_path, 'nn_History_42_features.png'))


def objective(trial) -> float:
    """
    Wrap model training with an objective function and return accuracy.
    Suggest hyperparameters using a trial object.
    """
    global model
    wds = trial.suggest_loguniform("weight_decay", 0.0001, 1)
    hidden_architecture = trial.suggest_categorical("hidden_architecture", ([64, 32, 16],
                                                                            [128, 64, 32],
                                                                            [100, 50, 25],
                                                                            [64, 32, 16, 8],
                                                                            [128, 64, 32, 16],
                                                                            [256, 128, 64, 32, 16]))
    afs = trial.suggest_categorical("activation", (['relu', 'relu', 'relu', 'relu', 'relu'],
                                                   ['selu', 'selu', 'selu', 'selu', 'selu'],
                                                   ['swish', 'swish', 'swish', 'swish', 'swish']))
    lrs = trial.suggest_loguniform('learning_rate', 0.001, 0.2)
    drops = trial.suggest_categorical("drop_rate", ([0.5, 0.5, 0.5, 0.5, 0.5],
                                                    [0.3, 0.3, 0.3, 0.3, 0.3],
                                                    [0.1, 0.1, 0.1, 0.1, 0.1],
                                                    [0.0, 0.0, 0.0, 0.0, 0.0]))
    batches = trial.suggest_categorical("batch_size", (32, 64))
    epochs = trial.suggest_categorical("epochs", (500, 1000, 1500, 2000))
    optimizers = trial.suggest_categorical("optimizer", ('adam', 'sgd', 'rmsprop'))

    model = NeuralNetwork(db=data,
                          weight_decay=wds,
                          dense_size=hidden_architecture,
                          activation_function=afs,
                          learning_rate=lrs,
                          drop_rate=drops,
                          batch_size=batches,
                          n_epochs=epochs,
                          optimizer=optimizers)

    accuracy, f1, precision, recall, roc_auc = model.train_and_test()

    # Test network on test human data
    print('Human Test:')
    acc_h, f1_h, precision_h, recall_h, roc_auc_h, pred_h, true_h = model.test(x_human, y_human)

    # Test network on test mouse data
    print('Mouse Test:')
    acc_m, f1_m, precision_m, recall_m, roc_auc_m, pred_m, true_m = model.test(x_mouse, y_mouse)

    print("Mouse accuracy: " + str(acc_m))
    print("Human accuracy: " + str(acc_h))
    return (acc_m + acc_h) / 2


def callback(study, trial) -> None:
    """
    Save best model
    """
    global best_model, model
    if study.best_trial == trial:
        best_model = model
        model.plot_history()


def optimize(n_trials: int, n_jobs: int = 1) -> None:
    """
    Create a study object and execute the optimization.
    """
    global best_model

    study = optuna.create_study(study_name='nn_domain_adaptation', direction='maximize')
    study.optimize(func=objective, n_trials=n_trials, n_jobs=n_jobs, callbacks=[callback])

    weight_decay = study.best_params['weight_decay']
    hidden_architecture = study.best_params['hidden_architecture']
    activation = study.best_params['activation']
    learning_rate = study.best_params['learning_rate']
    drop_rate = study.best_params['drop_rate']
    batch_size = study.best_params['batch_size']
    epochs = study.best_params['epochs']
    optimizer = study.best_params['optimizer']

    acc_h, f1_h, precision_h, recall_h, roc_auc_h, pred_h, true_h = best_model.test(x_human, y_human)
    acc_m, f1_m, precision_m, recall_m, roc_auc_m, pred_m, true_m = best_model.test(x_mouse, y_mouse)

    print("Trial Finished*************")
    print("Best model's accuracy for human data: {}".format(acc_h))
    print("Best model's accuracy for mouse data: {}".format(acc_m))
    print("Best model's weight decay: {}".format(weight_decay))
    print("Best model's hidden architecture: {}".format(hidden_architecture))
    print("Best model's activation function: {}".format(activation))
    print("Best model's learning rate: {}".format(learning_rate))
    print("Best model's drop rate: {}".format(drop_rate))
    print("Best model's batch size: {}".format(batch_size))
    print("Best model's n_epochs: {}".format(epochs))
    print("Best model's optimizer: {}".format(optimizer))

    best_model.model.save(filepath=model_path)


def get_data() -> tuple:
    """
    :return: tuple of merged mouse/human data, human testing data and mouse testing data.
    """
    data_mouse, data_mouse_test = train_test_split(mouse_data, test_size=0.2, random_state=0, shuffle=False)
    data_human, data_human_test = train_test_split(human_data, test_size=0.2, random_state=0, shuffle=False)
    data = data_mouse.append(data_human, ignore_index=True)
    return data, data_human_test, data_mouse_test


def run_best_model() -> None:
    """
        run best model based on the one saved from the hyperparameter grid search.
    """
    model = keras.models.load_model(filepath=model_path)
    labels = [y_human, y_mouse]
    tests = ['NN Human test 42 features', 'NN Mouse test 42 features']
    for idx, data in enumerate([x_human, x_mouse]):
        y_pred = np.argmax(model.predict(data, verbose=1), axis=1)
        y_true = np.argmax(labels[idx], axis=1)
        acc, f1, precision, recall, roc_auc = calculate_metrics(y_true, y_pred)
        print(tests[idx])
        print("Accuracy: " + str(acc))
        print("F1 Score: " + str(f1))
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("ROC AUC: " + str(roc_auc))
        dummy.plot_matrix(y_pred, y_true, tests[idx])
    model.summary()


def main():
    optimize(n_trials=150)
    run_best_model()
    plt.show()


if __name__ == '__main__':
    device = get_device()
    with tf.device(device):
        data, data_human_test, data_mouse_test = get_data()
        dummy = NeuralNetwork(db=data, weight_decay=0, dense_size=[], activation_function=[], learning_rate=0,
                               drop_rate=[0], batch_size=0, n_epochs=0, optimizer='adam')
        x_human, y_human, x_mouse, y_mouse = dummy.get_mouse_human_split_data(data_human_test, data_mouse_test)
        main()
