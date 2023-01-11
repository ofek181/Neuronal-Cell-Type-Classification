import os
import optuna
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

import lspin_model
from lspin_model import Model
from lspin_model import DataSet_meta
from gpu_check import get_device
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('INFO')
plt.rcParams.update({'font.size': 14})

# get directories
filepath = os.path.dirname(os.path.realpath(__file__))
transgenic_data = pd.read_csv(filepath + '/data/mouse/ephys_data.csv')
results_path = filepath + '/results/lspin'


class LocallySparse:
    """
    uses the LSPIN model to classify transgenic targeting in mouse data.
    """
    def __init__(self, data: pd.DataFrame, n_classes: int) -> None:
        """
        :param data: raw dataframe of neuronal recordings.
        :param n_classes: number of transgenic lines in the dataframe.
        """
        self.data = data
        self.n_classes = n_classes
        self.model, self.best_model = None, None
        self.model_params, self.training_params, self.class_names = {}, {}, {}
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = self._preprocess_data()

    def _preprocess_data(self) -> tuple:
        """
        :param df: raw dataframe to be processed.
        :param n_classes: number of classes in the dataframe.
        :return: split into train, validation and test.
        """
        db = self.data.dropna(axis=1, how='all')
        db = db.dropna(axis=0)
        irrelevant_columns = ['dendrite_type', 'neurotransmitter', 'reporter_status', 'layer', 'file_name']
        db = db.drop([x for x in irrelevant_columns if x in db.columns], axis=1, errors='ignore')
        db['transgenic_line'] = pd.Categorical(db['transgenic_line'])
        self.class_names = dict(enumerate(db['transgenic_line'].cat.categories))
        db['transgenic_line'] = db['transgenic_line'].cat.codes
        scaler = StandardScaler()
        y = db.pop('transgenic_line')
        y = y.values.astype(np.float32)
        y = to_categorical(y, num_classes=self.n_classes)
        x = db.values.astype(np.float32)
        x = scaler.fit_transform(x)
        x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=42)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.8, random_state=42)
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
                             "activation_gating": 'tanh',
                             "display_step": display_step}
        self.training_params = {'batch_size': self.x_train.shape[0]}

    def __objective(self, trial) -> float:
        """
        :param trial: a process of evaluating an objective function using optuna.
        :return: accuracy for the trial.
        """
        self.model_params['hidden_layers_node'] = trial.suggest_categorical("hidden_layers_node",
                                                                            [[100, 50, 25, 10],
                                                                             [50, 30, 20, 10],
                                                                             [50, 25, 10], [32, 16, 8],
                                                                             [50, 25], [40, 20]])
        self.model_params['gating_net_hidden_layers_node'] = trial.suggest_categorical("gating_net_hidden_layers_node",
                                                                                       [[100, 100],
                                                                                        [100, 100, 100],
                                                                                        [50, 50, 50],
                                                                                        [100, 100, 100, 100]])
        self.model_params['activation_pred'] = trial.suggest_categorical("activation_pred",
                                                                         ['relu', 'l_relu', 'sigmoid', 'tanh'])
        self.model_params['lam'] = trial.suggest_loguniform('lam', 0.01, 1)
        self.training_params['lr'] = trial.suggest_loguniform('learning_rate', 0.001, 0.1)
        self.training_params['num_epoch'] = trial.suggest_categorical('num_epoch', [500, 1000, 1500])

        self.model = Model(**self.model_params)
        _, _, _, _ = self.model.train(dataset=self._create_metadata(), **self.training_params)

        y_pred = self.model.test(self.x_test)
        y_true = np.argmax(self.y_test, axis=1)
        accuracy = accuracy_score(y_true, y_pred)
        matrix = confusion_matrix(y_true, y_pred)
        print(matrix)
        return accuracy

    def __callback(self, study, trial) -> None:
        """
        :param study: study of the optuna framework.
        :param trial: specific trial of the optuna optimization.
        """
        if study.best_trial == trial:
            self.best_model = self.model

    def optimize(self, n_trials: int, n_jobs: int = 1) -> None:
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
        best_lam = study.best_params['lam']
        best_lr = study.best_params['learning_rate']
        best_epoch = study.best_params['num_epoch']
        y_pred = self.best_model.test(self.x_test)
        accuracy = accuracy_score(np.argmax(self.y_test, axis=1), y_pred)
        f1 = f1_score(np.argmax(self.y_test, axis=1), y_pred, average='weighted')
        precision = precision_score(np.argmax(self.y_test, axis=1), y_pred, average='weighted')
        recall = recall_score(np.argmax(self.y_test, axis=1), y_pred, average='weighted')
        print("Trial Finished*************")
        print("Best model's prediction architecture: {}".format(best_pred_arch))
        print("Best model's gating architecture: {}".format(best_gate_arch))
        print("Best model's prediction activation function: {}".format(best_activ_pred))
        print("Best model's lambda: {}".format(best_lam))
        print("Best model's learning rate: {}".format(best_lr))
        print("Best model's num of epochs: {}".format(best_epoch))
        print("Test accuracy : {}".format(accuracy))
        print("Test F1 : {}".format(f1))
        print("Test precision : {}".format(precision))
        print("Test recall : {}".format(recall))

        with open(os.path.join(results_path, 'hyperparameters.txt'), "w") as file:
            file.write(
                "prediction architecture: {}\n"
                "gating architecture: {}\n"
                "prediction activation function: {}\n"
                "lambda: {}\n"
                "learning rate: {}\n"
                "number of epochs: {}\n"
                "Accuracy: {}\n"
                "F1: {}\n"
                "precision: {}\n"
                "recall: {}\n".format(best_pred_arch, best_gate_arch, best_activ_pred, best_lam,
                                      best_lr, best_epoch, accuracy, f1, precision, recall))

    def get_results(self) -> None:
        """
        plot the gate feature selection for each label in the data and confusion matrix.
        """
        # plot the gate feature selection matrix
        gate_matrix = []
        test_labels = np.argmax(self.y_test, axis=1)
        for i in range(self.n_classes):
            label_data = np.empty((0, self.x_test.shape[1]))
            for j in range(self.x_test.shape[0]):
                if test_labels[j] == i:
                    label_data = np.vstack([label_data, self.x_test[j, :]])

            gate_matrix.append(self.best_model.get_prob_alpha(label_data))
            plt.figure()
            sns.heatmap(gate_matrix[i], vmin=0, vmax=1)
            plt.title("Label: {}".format(self.class_names[i]))
            plt.draw()
            plt.savefig(results_path + "/gate_matrix_" + str(self.class_names[i]) + ".png")

        # plot the confusion matrix
        y_pred = self.best_model.test(self.x_test)
        y_true = np.argmax(self.y_test, axis=1)

        def reverse_labels(tup: tuple) -> list:
            """
            :param tup: tuple of label.
            :return: names of labels.
            """
            return [self.class_names[x] for x in tup]

        y_true_labeled, y_pred_labeled = reverse_labels(tuple(y_true)), reverse_labels(tuple(y_pred))
        matrix = confusion_matrix(y_true_labeled, y_pred_labeled)
        df_cm = pd.DataFrame(matrix, columns=np.unique(y_true_labeled), index=np.unique(y_true_labeled))
        # df_cm.index.name = 'Actual'
        # df_cm.columns.name = 'Predicted'
        plt.figure()
        cmap = sns.cubehelix_palette(light=0.9, as_cmap=True)
        cm_normalized = df_cm.div(df_cm.sum(axis=0), axis=1)
        sns.heatmap(cm_normalized, cbar=False, annot=True, cmap=cmap, square=True, fmt='.1%', annot_kws={'size': 10})
        plt.title('LSPIN Classification')
        plt.tight_layout()
        plt.draw()
        plt.savefig(results_path + "/lspin_results.png")
        plt.show()

    def save_model(self):
        """
        save the best model to path.
        """
        self.best_model.save(model_dir=results_path+'/model')


def load_model():
    tf.compat.v1.reset_default_graph()
    v1 = tf.compat.v1.Variable(tf.compat.v1.constant(0.1, shape=[2]), name="v1")
    v2 = tf.compat.v1.Variable(tf.compat.v1.constant(0.2, shape=[2]), name="v2")
    saver = tf.compat.v1.train.import_meta_graph(results_path + '/model/model.ckpt.meta')
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, results_path + '/model/model.ckpt')
        print("Model restored")


def main():
    clf = LocallySparse(data=transgenic_data, n_classes=5)
    clf.create_model(display_step=2000, feature_selection=True)
    clf.optimize(n_trials=2800)
    clf.get_results()
    clf.save_model()
    # load_model()


if __name__ == '__main__':
    device = get_device()
    with tf.device(device):
        main()
