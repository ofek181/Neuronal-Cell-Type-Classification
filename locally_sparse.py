import os
import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from lspin import Model
from lspin import DataSet_meta
from gpu_check import get_device

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('INFO')

dir_path = os.path.abspath('')
inhibitory = dir_path + '/data/dataframe/mouse/inhibitory_transgenic_data.csv'
excitatory = dir_path + '/data/dataframe/mouse/excitatory_transgenic_data.csv'
data_inhibitory = pd.read_csv(inhibitory)
data_excitatory = pd.read_csv(excitatory)
results_path = dir_path + '/results/lspin'


class LocallySparse:
    def __init__(self, data: pd.DataFrame, n_classes: int):
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
        class_names = dict(enumerate(db['transgenic_line'].cat.categories))
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

    def _create_metadata(self):
        """
        :return: metadata supported by the locally sparse model.
        """
        dataset = DataSet_meta(**{'_data': self.x_train, '_labels': self.y_train, '_meta': self.y_train,
                                  '_valid_data': self.x_val, '_valid_labels': self.y_val,
                                  '_valid_meta': self.y_val, '_test_data': self.x_test,
                                  '_test_labels': self.y_test, '_test_meta': self.y_test})
        return dataset

    def create_model(self, hidden_architecture: list, gating_hidden_architecture: list,
                     display_step: int, activation_pred: str, activation_gating: str,
                     feature_selection: bool):
        """
        :param hidden_architecture: list, number of nodes for each hidden layer for the prediction net.
        :param gating_hidden_architecture: list, number of nodes for each hidden layer of the gating net.
        :param display_step: integer, number of epochs to output info.
        :param activation_pred: string, activation function of the prediction net: 'relu', 'l_relu', 'sigmoid', 'tanh'.
        :param activation_gating: string, activation function of the gating net: 'relu', 'l_relu', 'sigmoid', 'tanh'.
        :param feature_selection: bool, if using the gating net.
        :return: dict of model parameters.
        """
        self.model_params = {"input_node": self.x_train.shape[1],
                             "hidden_layers_node": hidden_architecture,
                             "output_node": self.n_classes,
                             "feature_selection": feature_selection,
                             "gating_net_hidden_layers_node": gating_hidden_architecture,
                             "display_step": display_step,
                             'activation_pred': activation_pred,
                             'activation_gating': activation_gating}
        self.training_params = {'batch_size': self.x_train.shape[0]}

    def __objective(self, trial):
        """
        :param trial: a process of evaluating an objective function using optuna.
        :param model_params: dict of model parameters.
        :param training_params: dict of training parameters.
        :return: accuracy for the trial.
        """
        self.model_params['lam'] = trial.suggest_loguniform('lam', 0.001, 0.1)
        self.training_params['lr'] = trial.suggest_loguniform('learning_rate', 0.001, 0.1)
        self.training_params['num_epoch'] = trial.suggest_categorical('num_epoch', [500, 1000, 2000])

        # specify the model with these parameters and train the model
        self.model = Model(**self.model_params)
        _, _, _, _ = self.model.train(dataset=self._create_metadata(), **self.training_params)

        print("In trial:---------------------")
        y_pred = self.model.test(self.x_val)
        true = np.argmax(self.y_val, axis=1)
        accuracy = accuracy_score(true, y_pred)
        return accuracy

    def __callback(self, study, trial):
        if study.best_trial == trial:
            self.best_model = self.model

    def optimize(self, n_trials: int, n_jobs: int = 1):
        """
        :param n_trials: number of optimization trials.
        :param n_jobs: maximum number of concurrently running workers.
        :return: optimize the model via Optuna and return the best accuracy and parameters.
        """
        study = optuna.create_study(study_name='lspin', direction='maximize')
        study.optimize(func=self.__objective, n_trials=n_trials, n_jobs=n_jobs, callbacks=[self.__callback])
        best_lr = study.best_params['learning_rate']
        best_epoch = study.best_params['num_epoch']
        best_lam = study.best_params['lam']
        y_pred = self.best_model.test(self.x_test)
        print("Trial Finished*************")
        print("Best model's lambda: {}".format(best_lam))
        print("Best model's learning rate: {}".format(best_lr))
        print("Best model's num of epochs: {}".format(best_epoch))
        print("Test accuracy : {}".format(accuracy_score(np.argmax(self.y_test, axis=1), y_pred)))
        return accuracy_score(np.argmax(self.y_test, axis=1), y_pred), best_lam, best_lr, best_epoch


def main():
    column_names = ["Accuracy", "Lambda", "Learning Rate", "Epoch"]
    results = pd.DataFrame(columns=column_names)
    title = 'inhibitory_classification.csv'
    clf = LocallySparse(data=data_inhibitory, n_classes=5)
    hidden_architecture = [[200, 100, 50, 20], [100, 50, 20, 10], [100, 50]]
    gating_hidden_architecture = [[100], [50], [10]]
    activation_pred = ['l_relu', 'relu']
    activation_gating = ['relu', 'tanh']
    n_run = 0
    for a in hidden_architecture:
        for b in gating_hidden_architecture:
            for c in activation_pred:
                for d in activation_gating:
                    clf.create_model(hidden_architecture=a, gating_hidden_architecture=b,
                                     display_step=1000, activation_pred=c, activation_gating=d,
                                     feature_selection=True)
                    acc, lam, lr, epoch = clf.optimize(n_trials=100)
                    if acc > 0.7:
                        results.loc[n_run] = [acc, lam, lr, epoch]
                        results.to_csv(os.path.join(results_path, title), index=True)
                        n_run += 1


if __name__ == '__main__':
    device = get_device()
    with tf.device(device):
        main()


