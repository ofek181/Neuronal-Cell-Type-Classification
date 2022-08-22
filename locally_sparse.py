import os
import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from lspin import Model
from lspin import DataSet_meta
from gpu_check import get_device
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('INFO')

dir_path = os.path.abspath('')
inhibitory = dir_path + '/data/dataframe/mouse/inhibitory_transgenic_data.csv'
excitatory = dir_path + '/data/dataframe/mouse/excitatory_transgenic_data.csv'
data_inhibitory = pd.read_csv(inhibitory)
data_excitatory = pd.read_csv(excitatory)
results_path = dir_path + '/results/lspin'


class LocallySparse:
    def __init__(self, data: pd.DataFrame, n_classes: int) -> None:
        """
        :param data: raw dataframe of neuronal recordings.
        :param n_classes: number of transgenic lines in the dataframe.
        """
        self.data = data
        self.n_classes = n_classes
        self.model, self.best_model = None, None
        self.model_params, self.training_params = {}, {}
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = self._preprocess_data()

    def _preprocess_data(self) -> tuple:
        """
        :param df: raw dataframe to be processed.
        :param n_classes: number of classes in the dataframe.
        :return: split into train, validation and test.
        """
        db = self.data.dropna(axis=1, how='all')
        db = db.dropna(axis=0)
        irrelevant_columns = ['dendrite_type', 'layer', 'mean_clipped', 'file_name', 'mean_threshold_index',
                              'mean_peak_index', 'mean_trough_index', 'mean_upstroke_index', 'mean_downstroke_index',
                              'mean_fast_trough_index']
        db = db.drop([x for x in irrelevant_columns if x in db.columns], axis=1)
        db['transgenic_line'] = pd.Categorical(db['transgenic_line'])
        db['transgenic_line'] = db['transgenic_line'].cat.codes
        scaler = StandardScaler()
        y = db.pop('transgenic_line')
        y = y.values.astype(np.float32)
        y = to_categorical(y, num_classes=self.n_classes)
        x = db.values.astype(np.float32)
        x = scaler.fit_transform(x)
        x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.9, random_state=42)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.65, random_state=42)
        return x_train, y_train, x_val, y_val, x_test, y_test

    def _create_metadata(self) -> DataSet_meta:
        """
        :return: metadata supported by the locally sparse model.
        """
        dataset = DataSet_meta(**{'_data': self.x_train, '_labels': self.y_train, '_meta': self.y_train,
                                  '_valid_data': self.x_val, '_valid_labels': self.y_val, '_valid_meta': self.y_val,
                                  '_test_data': self.x_test, '_test_labels': self.y_test, '_test_meta': self.y_test})
        return dataset

    def create_model(self, display_step: int, feature_selection: bool):
        """
        :param display_step: integer, number of epochs to output info.
        :param feature_selection: bool, if using the gating net.
        :return: dict of model parameters.
        """
        self.model_params = {"input_node": self.x_train.shape[1],
                             "output_node": self.n_classes,
                             "feature_selection": feature_selection,
                             "display_step": display_step}
        self.training_params = {'batch_size': self.x_train.shape[0]}

    def __objective(self, trial) -> float:
        """
        :param trial: a process of evaluating an objective function using optuna.
        :return: accuracy for the trial.
        """
        self.model_params['hidden_layers_node'] = trial.suggest_categorical("hidden_layers_node",
                                                                            [[100, 100, 100, 100],
                                                                             [256, 256, 256],
                                                                             [512, 256, 128],
                                                                             [100, 100, 100],
                                                                             [64, 32, 16],
                                                                             [32, 16]])
        self.model_params['gating_net_hidden_layers_node'] = trial.suggest_categorical("gating_net_hidden_layers_node",
                                                                                       [[100], [50], [10],
                                                                                        [100, 100], [100, 100, 100],
                                                                                        [200, 200], [50, 50, 50]])
        self.model_params['activation_pred'] = trial.suggest_categorical("activation_pred",
                                                                         ['relu', 'l_relu', 'sigmoid', 'tanh'])
        self.model_params['activation_gating'] = trial.suggest_categorical("activation_gating",
                                                                           ['relu', 'l_relu', 'sigmoid', 'tanh'])
        self.model_params['lam'] = trial.suggest_loguniform('lam', 0.0001, 1)
        self.training_params['lr'] = trial.suggest_loguniform('learning_rate', 0.001, 0.1)
        self.training_params['num_epoch'] = trial.suggest_categorical('num_epoch', [500, 1000, 5000, 10000])

        self.model = Model(**self.model_params)
        _, _, _, _ = self.model.train(dataset=self._create_metadata(), **self.training_params)

        y_pred = self.model.test(self.x_val)
        y_true = np.argmax(self.y_val, axis=1)
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy

    def __callback(self, study, trial) -> None:
        """
        :param study: study of the optuna framework.
        :param trial: specific trial of the optuna optimization.
        """
        if study.best_trial == trial:
            self.best_model = self.model

    def optimize(self, n_trials: int, n_jobs: int = 1) -> float:
        """
        :param n_trials: number of optimization trials.
        :param n_jobs: maximum number of concurrently running workers.
        :return: best m
        """
        study = optuna.create_study(study_name='lspin', direction='maximize')
        study.optimize(func=self.__objective, n_trials=n_trials, n_jobs=n_jobs, callbacks=[self.__callback])
        best_pred_arch = study.best_params['hidden_layers_node']
        best_gate_arch = study.best_params['gating_net_hidden_layers_node']
        best_activ_pred = study.best_params['activation_pred']
        best_gate_pred = study.best_params['activation_gating']
        best_lam = study.best_params['lam']
        best_lr = study.best_params['learning_rate']
        best_epoch = study.best_params['num_epoch']
        y_pred = self.best_model.test(self.x_test)
        accuracy = accuracy_score(np.argmax(self.y_test, axis=1), y_pred)
        print("Trial Finished*************")
        print("Best model's prediction architecture: {}".format(best_pred_arch))
        print("Best model's gating architecture: {}".format(best_gate_arch))
        print("Best model's prediction activation function: {}".format(best_activ_pred))
        print("Best model's gating activation function: {}".format(best_gate_pred))
        print("Best model's lambda: {}".format(best_lam))
        print("Best model's learning rate: {}".format(best_lr))
        print("Best model's num of epochs: {}".format(best_epoch))
        print("Test accuracy : {}".format(accuracy))
        return accuracy


def main():
    # column_names = ["Accuracy", "Prediction Network Architecture", "Gating Network Architecture",
    #                 "Prediction Activation Function", "Gating Activation Function",
    #                 "Lambda", "Learning Rate", "Epoch"]
    # results = pd.DataFrame(columns=column_names)
    # title = 'inhibitory_classification.csv'
    clf = LocallySparse(data=data_inhibitory, n_classes=5)
    clf.create_model(display_step=1000, feature_selection=True)
    acc = clf.optimize(n_trials=500)

    # TODO find good hyper-parameters
    # TODO try removing the sparsely spiny class?
    # TODO filter out spiny-aspiny classes into corresponding labels



if __name__ == '__main__':
    device = get_device()
    with tf.device(device):
        main()


