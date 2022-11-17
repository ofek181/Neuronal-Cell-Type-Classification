import os
import random
import shap
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop
from keras.utils import to_categorical
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from classifier import Model
from helper_functions import calculate_metrics
from gpu_check import get_device

# get directories
dir_path = os.path.dirname(os.path.realpath(__file__))
results_path = dir_path + '/results/neural_net'
model_path = results_path + '/results/neural_net/model'
data_mouse = pd.read_csv(dir_path + '/data/mouse/ephys_data.csv')
data_human = pd.read_csv(dir_path + '/data/human/ephys_data.csv')
results_mouse = dir_path + 'results/neural_net/mouse/dendrite_type'
results_human = dir_path + 'results/neural_net/human'

# Cancel randomness for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]


class DNNClassifier(Model):
    def __init__(self, db: pd.DataFrame, n_layers: int, weight_decay: float, dense_size: list,
                 activation_function: list, learning_rate: float, drop_rate: list,
                 batch_size: int,  n_epochs: int, optimizer: str = 'adam', n_classes: int = 2) -> None:
        """
        :param db: dataframe for training and testing.
        :param n_layers: number of layer in the model.
        :param weight_decay: l2 regularization values.
        :param dense_size: size of the dense layers.
        :param activation_function: activation function in each dense layer.
        :param learning_rate: learning rate during training.
        :param drop_rate: dropout rate.
        :param batch_size: batch size during training and testing.
        :param n_epochs: number of epochs during training.
        :param optimizer: optimizer used (adam, sgd or rmsprop).
        :param n_classes: number of classes for the task.
        """
        self.wd = weight_decay
        self.lr = learning_rate
        self.dr = drop_rate
        self.af = activation_function
        self.opt = optimizer
        self.n_classes = n_classes
        db = self.preprocess_data(db)
        super(DNNClassifier, self).__init__(data=db, num_layers=n_layers, num_neurons=dense_size,
                                            batch_size=batch_size, n_epochs=n_epochs)

    def _create_model(self) -> Sequential:
        """
        :return: a sequential keras model.
        """
        model = Sequential()
        for i in range(self._num_layers):
            model.add(BatchNormalization())
            model.add(Dense(self._num_nodes[i], activation=self.af[i],
                            kernel_regularizer=l2(self.wd), bias_regularizer=l2(self.wd)))
            model.add(Dropout(self.dr[i]))
        model.add(Dense(self.n_classes, activation='softmax'))
        return model

    def train_and_test(self) -> tuple:
        """
        :return: trains and tests a neural network.
        """
        # Split into train, val and test
        x_train, y_train, x_val, y_val, x_test, y_test = self.split_train_val_test(self.data)

        # Assign optimizer
        opt = Adam(learning_rate=self.lr, decay=self.lr/self.n_epochs)
        if self.opt == 'sgd':
            opt = SGD(learning_rate=self.lr, decay=self.lr/self.n_epochs)
        if self.opt == 'rmsprop':
            opt = RMSprop(learning_rate=self.lr, decay=self.lr/self.n_epochs)

        # Compile model
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics="accuracy")
        # Fit model
        history = self.model.fit(x_train, y_train, epochs=self.n_epochs, batch_size=self._batch_size,
                                 validation_data=(x_val, y_val), verbose=0, callbacks=callbacks)
        # Plot history
        self.plot_history(history)

        # Test model
        accuracy, f1, precision, recall, roc_auc = self.test(x_test, y_test)
        return accuracy, f1, precision, recall, roc_auc

    def test(self, x_test, y_test) -> tuple:
        """
        :param x_test: testing data.
        :param y_test: true labels of the testing data.
        :return: loss and accuracy of the model on the testing data.
        """
        # Calculate test loss and accuracy
        predictions = self.model.predict(x_test, verbose=0)
        y_pred, y_test = np.argmax(predictions, axis=1), np.argmax(y_test, axis=1)
        accuracy, f1, precision, recall, roc_auc = calculate_metrics(y_test, y_pred)

        print('==============================================')
        print("Accuracy: " + str(accuracy))
        print("F1 Score: " + str(f1))
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("ROC AUC: " + str(roc_auc))

        # Plot confusion matrix
        matrix = confusion_matrix(y_test, y_pred)
        plt.figure()
        label_names = ['aspiny', 'spiny']
        s = sns.heatmap(matrix / np.sum(matrix), annot=True, fmt='.2%',
                        cmap='Blues', xticklabels=label_names, yticklabels=label_names)
        s.set(xlabel='Predicted label', ylabel='True label')
        plt.draw()
        return accuracy, f1, precision, recall, roc_auc

    @staticmethod
    def preprocess_data(df) -> pd.DataFrame:
        """
        :param df: raw dataframe.
        :return: processed dataframe.
        """
        db = df.dropna(axis=1, how='all')
        db = db.dropna(axis=0)
        irrelevant_columns = ['transgenic_line', 'neurotransmitter', 'reporter_status', 'layer', 'file_name']
        db = db.drop([x for x in irrelevant_columns if x in df.columns], axis=1, errors='ignore')
        db['dendrite_type'] = pd.Categorical(db['dendrite_type'])
        db['dendrite_type'] = db['dendrite_type'].cat.codes
        return db

    def split_train_val_test(self, data: pd.DataFrame) -> tuple:
        """
        :param data: processed dataset.
        :return: data split into train, val and test.
        """
        scaler = StandardScaler()
        y = data.pop('dendrite_type')
        y = y.values.astype(np.float32)
        y = to_categorical(y, num_classes=self.n_classes)
        x = data.values.astype(np.float32)
        x = scaler.fit_transform(x)
        x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=42)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.8, random_state=42)
        return x_train, y_train, x_val, y_val, x_test, y_test

    @staticmethod
    def plot_history(history) -> None:
        """
        :param history: history of the training process.
        :return: plots the training process over the number of epochs.
        """
        plt.figure()
        plt.plot(history.history['accuracy'], label='train_accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.draw()

    @staticmethod
    def save_results(results: pd.DataFrame, path: str, name: str) -> None:
        pass


class NeuralShap:
    """
    Explainer uses Shapley Values in order to explain feature importance in a model.
    """
    def __init__(self, title: str = None, data: pd.DataFrame = None, model: str = None) -> None:
        """
        :param title: name of the model.
        :param data: type of data to be processed.
        """
        self.model = tf.keras.models.load_model(filepath=model)
        self.title = title
        self.data = data

    def explain_model(self) -> None:
        """
        :return: shap.summary_plot for a multi-output model.
        """
        # use a dummy classifier to preprocess the data
        dummy = DNNClassifier(db=self.data, n_layers=0, weight_decay=0, dense_size=[], activation_function=[],
                              learning_rate=0, drop_rate=[], batch_size=0, n_epochs=0, optimizer='adam')
        data = dummy.preprocess_data(self.data)

        # get features' names
        features = list(data.columns)
        features.remove('dendrite_type')

        # split into train, val and test
        x_train, y_train, x_val, y_val, x_test, y_test = dummy.split_train_val_test(data)

        # use the kernel explainer to get the Shapley values
        kernel_explainer = shap.KernelExplainer(self.model, x_test)
        shapley_values = kernel_explainer.shap_values(x_test)

        # draw summary plot
        self._draw_summary_plot(shapley_values, x_test, features)

    def _draw_summary_plot(self, shapley_values: list, data: np.ndarray,
                           features: list, size: tuple = (10, 10)) -> None:
        """
        :param shapley_values: shap values obtained from the explainer.
        :param data: data that was tested.
        :param features: features' names
        :param size: size of the plot
        """
        plt.figure()
        shap.summary_plot(shap_values=shapley_values, features=data, feature_names=features, plot_size=size,
                          show=False, color=plt.get_cmap("Pastel1"), class_names=["aspiny", "spiny"])
        plt.title(self.title)
        plt.tight_layout()
        plt.draw()


def grid_search(data: pd.DataFrame) -> DNNClassifier:
    layers = [3, 5]
    l2s = [0.01, 0.0001]
    denses = [[64, 64, 64, 64, 64], [64, 32, 16, 8, 4], [64, 64, 32, 32, 16, 16]]
    activations = [['relu', 'relu', 'relu', 'relu', 'relu']]
    lrs = [0.01, 0.001]
    drops = [[0.3, 0.3, 0.3, 0.3, 0.3]]
    bss = [32]
    epochs = [100]
    optims = 'adam'
    for layer in layers:
        for l2 in l2s:
            for dense in denses:
                for activation in activations:
                    for lr in lrs:
                        for drop in drops:
                            for bs in bss:
                                for epoch in epochs:
                                    for optim in optims:
                                        clf = DNNClassifier(db=data, n_layers=layer, weight_decay=l2,
                                                            dense_size=dense, activation_function=activation,
                                                            learning_rate=lr, drop_rate=drop, batch_size=bs,
                                                            n_epochs=epoch, optimizer=optim)
                                        accuracy, f1, precision, recall, roc_auc = clf.train_and_test()
                                        if accuracy > 0.9:
                                            return clf


def train(data: pd.DataFrame) -> DNNClassifier:
    """
    :param data: data to be trained on
    :return: a trained DNNClassifier model
    """
    clf = DNNClassifier(db=data, n_layers=3, weight_decay=0.001, dense_size=[64, 64, 32, 32, 16, 16],
                        activation_function=['swish', 'swish', 'swish', 'swish', 'swish', 'swish'], learning_rate=0.01,
                        drop_rate=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], batch_size=16, n_epochs=1024, optimizer='adam')
    clf.train_and_test()
    return clf


def main():
    # dnnclf = grid_search(data_mouse)
    # dnnclf.model.save(filepath=results_mouse + '/model')
    # Explain model using SHAP
    # model_path = results_mouse+'/model'
    # xai = NeuralShap(title='Mouse Feature Importance', data=data_mouse, model=model_path)
    # xai.explain_model()
    # plt.show()
    dnnclf = grid_search(data_human)
    dnnclf.model.save(filepath=results_human + '/model')
    model_path = results_human+'/model'
    xai = NeuralShap(title='Human Feature Importance', data=data_human, model=model_path)
    xai.explain_model()
    plt.show()

    # print("==============================================")
    # print("Train on mouse data:")
    # dnnclf = train(data_mouse)
    # print("==============================================")
    # print("Human test on mouse network:")
    # human_test = dnnclf.preprocess_data(data_human)
    # scaler = StandardScaler()
    # y = human_test.pop('dendrite_type')
    # y = y.values.astype(np.float32)
    # y = to_categorical(y, num_classes=2)
    # x = human_test.values.astype(np.float32)
    # x = scaler.fit_transform(x)
    # accuracy_h, f1_h, precision_h, recall_h, roc_auc_h = dnnclf.test(x, y)
    # dnnclf.model.save(filepath=results_mouse + '/model')

    # print("==============================================")
    # print("Training on human data:")
    # dnnclf = train(data_human)
    # print("==============================================")
    # print("Mouse test on human network:")
    # mouse_test = dnnclf.preprocess_data(data_mouse)
    # scaler = StandardScaler()
    # y = mouse_test.pop('dendrite_type')
    # y = y.values.astype(np.float32)
    # y = to_categorical(y, num_classes=2)
    # x = mouse_test.values.astype(np.float32)
    # x = scaler.fit_transform(x)
    # accuracy_m, f1_m, precision_m, recall_m, roc_auc_m = dnnclf.test(x, y)
    # dnnclf.model.save(filepath=results_human + '/model')

    # Explain model using SHAP
    # models = ['Human Feature Importance', 'Mouse Feature Importance']
    # data = [data_human, data_mouse]
    # model_paths = [results_human+'/model', results_mouse+'/model']
    # for idx, val in enumerate(models):
    #     xai = NeuralShap(title=val, data=data[idx], model=model_paths[idx])
    #     xai.explain_model()
    #
    # plt.show()


if __name__ == '__main__':
    device = get_device()
    with tf.device(device):
        main()
