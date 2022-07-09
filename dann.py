import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model, Input
from keras.layers import Layer
from keras.layers import Dense, BatchNormalization, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam, SGD, RMSprop
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

n_classes = 2
n_domains = 2

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if tf.test.gpu_device_name():
    print('GPU found')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
else:
    print("No GPU found")

# TODO plot confusion matrix
# TODO make the code pretty and add docstring
# TODO find hyper-parameters that work well for the domain adaptation task


# Gradient Reversal Layer
class GradientReversal(Layer):
    @tf.custom_gradient
    def grad_reverse(self, x):
        y = tf.identity(x)

        def custom_grad(dy):
            return tf.negative(dy) * self.lamda
        return y, custom_grad

    def __init__(self, lamda: float, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.lamda = lamda

    def call(self, inputs, *args, **kwargs):
        return self.grad_reverse(inputs)


class DANNClassifier(Model):
    def __init__(self, db: pd.DataFrame, n_layers: int, weight_decay: float, dense_size: list,
                 activation_function: list, learning_rate: float, drop_rate: list,
                 batch_size: int,  n_epochs: int, optimizer: str = 'adam', dp_lambda: float = 1.0) -> None:
        super(DANNClassifier, self).__init__()
        self.wd = weight_decay
        self.lr = learning_rate
        self.dr = drop_rate
        self.af = activation_function
        self.opt = optimizer
        self._num_layers = n_layers
        self._num_nodes = dense_size
        self._batch_size = batch_size
        self.n_epochs = n_epochs
        self.dp_lambda = dp_lambda
        self.data = self.preprocess_data(db)
        self.model = self._create_model()

    def _create_model(self):
        # Feature extractor
        n_label_columns = 2
        inputs = Input(shape=(self.data.shape[1]-n_label_columns,), name='input')
        x = BatchNormalization()(inputs)
        x = Dense(self._num_nodes[0], activation=self.af[0],
                  kernel_regularizer=l2(self.wd), bias_regularizer=l2(self.wd))(x)
        x = Dropout(self.dr[0])(x)
        for i in range(1, self._num_layers):
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
        # Split for train and test
        # x_train, y_train_domain, y_train_label, x_val, y_val_domain,\
        # y_val_label, x_test, y_test_domain, y_test_label = self.split_train_val_test(self.data)
        x_train, y_train, x_val, y_val, x_test, y_test = self.split_train_val_test()

        # Get optimizer
        opt = Adam(learning_rate=self.lr, decay=self.lr/self.n_epochs)
        if self.opt == 'sgd':
            opt = SGD(learning_rate=self.lr, decay=self.lr/self.n_epochs)
        if self.opt == 'rmsprop':
            opt = RMSprop(learning_rate=self.lr, decay=self.lr/self.n_epochs)

        # Compile model
        self.model.compile(loss={'l_pred': 'categorical_crossentropy','d_pred': 'categorical_crossentropy'},
                           optimizer=opt, metrics="accuracy")
        # fit model
        history = self.model.fit(x_train, {'l_pred': y_train[:, 0, :], 'd_pred': y_train[:, 1, :]},
                                 validation_data=(x_val, {'l_pred': y_val[:, 0, :], 'd_pred': y_val[:, 1, :]}),
                                 epochs=self.n_epochs, batch_size=self._batch_size, verbose=0)
        # plot history
        self.plot_history(history)

        # test the model
        loss, acc = self.test(x_test, y_test)

        # print summary
        # print('--------------------------------------------------------------')
        # print("DNN Summary: ")
        # self.model.summary()
        return loss, acc

    def test(self, x_test, y_test) -> tuple:
        # calculate test loss and accuracy
        cce = tf.keras.losses.CategoricalCrossentropy()
        predictions = self.model.predict(x_test, verbose=0)
        l_pred = predictions[0]
        d_pred = predictions[1]
        l_true = np.argmax(y_test[:, 0, :], axis=1)
        d_true = np.argmax(y_test[:, 1, :], axis=1)
        loss = cce(y_test[:, 0, :], l_pred).numpy()
        l_pred = np.argmax(l_pred, axis=1)
        d_pred = np.argmax(d_pred, axis=1)
        l_acc = accuracy_score(l_true, l_pred)
        d_acc = accuracy_score(d_true, d_pred)
        # print('====================================================')
        print("Label Accuracy: " + str(l_acc))
        # print("Domain Accuracy: " + str(d_acc))
        # # plot confusion matrix
        # y_pred = self.model.predict(x_test)
        # matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
        # plt.figure()
        # label_names = ['aspiny', 'spiny']
        # s = sns.heatmap(matrix / np.sum(matrix), annot=True, fmt='.2%',
        #             cmap='Blues', xticklabels=label_names, yticklabels=label_names)
        # s.set(xlabel='Predicted label', ylabel='True label')
        # plt.draw()
        return loss, l_acc

    def test_label_predictions(self, x_test, y_test) -> tuple:
        # calculate test loss and accuracy
        cce = tf.keras.losses.CategoricalCrossentropy()
        l_pred = self.model.predict(x_test, verbose=0)[0]
        l_true = np.argmax(y_test, axis=1)
        loss = cce(y_test, l_pred).numpy()
        l_pred = np.argmax(l_pred, axis=1)
        l_acc = accuracy_score(l_true, l_pred)
        # print('====================================================')
        print("Accuracy: " + str(l_acc))
        # # plot confusion matrix
        # y_pred = self.model.predict(x_test)
        # matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
        # plt.figure()
        # label_names = ['aspiny', 'spiny']
        # s = sns.heatmap(matrix / np.sum(matrix), annot=True, fmt='.2%',
        #             cmap='Blues', xticklabels=label_names, yticklabels=label_names)
        # s.set(xlabel='Predicted label', ylabel='True label')
        # plt.draw()
        return loss, l_acc

    @staticmethod
    def _create_label_predictor() -> list:
        label_predictor = [Dense(units=6, activation='relu'),
                           Dense(units=n_classes, activation='softmax')]
        return label_predictor

    @staticmethod
    def _create_domain_predictor() -> list:
        domain_predictor = [Dense(units=6, activation='relu'),
                            Dense(units=n_domains, activation='softmax')]
        return domain_predictor

    @staticmethod
    def preprocess_data(df):
        db = df.dropna(axis=1, how='all')
        db = db.dropna(axis=0)
        irrelevant_columns = ['layer', 'structure_area_abbrev', 'sampling_rate', 'mean_clipped', 'file_name']
        db = db.drop([x for x in irrelevant_columns if x in df.columns], axis=1)
        db['dendrite_type'] = pd.Categorical(db['dendrite_type'])
        db['dendrite_type'] = db['dendrite_type'].cat.codes
        db['organism'] = pd.Categorical(db['organism'])
        db['organism'] = db['organism'].cat.codes
        return db

    def split_train_val_test(self) -> tuple:
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

        x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.85, random_state=42)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.7, random_state=42)
        return x_train, y_train, x_val, y_val, x_test, y_test

    @staticmethod
    def plot_history(history) -> None:
        plt.figure()
        plt.plot(history.history['l_pred_accuracy'], label='label train accuracy')
        plt.plot(history.history['val_l_pred_accuracy'], label='label val accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.draw()


def get_data() -> tuple:
    # get data from directories
    dataframe_path_mouse = dir_path + '/data/dataframe/mouse'
    dataframe_path_human = dir_path + '/data/dataframe/human'
    dataframe_name = 'extracted_mean_ephys_data.csv'
    data_mouse = pd.read_csv(dataframe_path_mouse + '/' + dataframe_name)
    data_mouse['organism'] = 0
    data_mouse, data_mouse_test = train_test_split(data_mouse, test_size=0.2)
    data_human = pd.read_csv(dataframe_path_human + '/' + dataframe_name)
    data_human['organism'] = 1
    data_human, data_human_test = train_test_split(data_human, test_size=0.1)
    data = data_mouse.append(data_human, ignore_index=True)
    return data, data_human_test, data_mouse_test


def main(args):
    data, data_human_test, data_mouse_test = get_data()
    results_path = dir_path + '/results/DANN'
    column_names = ["N Layers", "Dense Size", "Activation Function", "Drop Rate", "Optimizer",
               "N Epochs", "Weight Decay", "Learning Rate", "Batch Size", "Lambda",
               "Total Accuracy", "Human Accuracy", "Mouse Accuracy"]
    results = pd.DataFrame(columns=column_names)
    # Hyperparameter grid search
    layers = [6, 5, 4, 3, 2, 1]
    wds = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    dense_sizes = [[27, 128, 64, 32, 16, 8], [64, 256, 128, 64, 32, 32], [27, 32, 64, 128, 64, 32],
                   [256, 512, 1024, 128, 64, 32], [27, 32, 32, 16, 16, 8], [27, 20, 16, 10, 8, 4]]
    afs = [['swish', 'swish', 'swish', 'swish', 'swish', 'swish'], ['relu', 'relu', 'relu', 'relu', 'relu', 'relu'],
           ['swish', 'swish', 'swish', 'relu', 'relu', 'relu'], ['relu', 'relu', 'relu', 'swish', 'swish', 'swish']]
    lrs = [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
    drops = [[0.4, 0.4, 0.4, 0.4, 0.4, 0.4], [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
             [0.1, 0.1, 0.1, 0.2, 0.2, 0.2], [0.5, 0.4, 0.3, 0.3, 0.2, 0.1],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    batches = [16, 32, 64]
    epochs = [100, 250, 500, 1000, 2000]
    optimizers = ['adam', 'sgd', 'rmsprop']
    lambdas = [0.3, 0.35, 0.4, 0.45, 0.48, 0.5, 0.52, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
               1, 1.1, 1.2, 1.3, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

    DANN = DANNClassifier(db=data,
                          n_layers=1,
                          weight_decay=0.1,
                          dense_size=[100],
                          activation_function=['relu'],
                          learning_rate=0.1,
                          drop_rate=[0.1],
                          batch_size=8,
                          n_epochs=1,
                          optimizer='adam',
                          dp_lambda=1)
    # Preprocess data
    scaler = StandardScaler()
    data_human_test = DANN.preprocess_data(data_human_test)
    data_human_test = data_human_test.drop('organism', axis=1)
    data_mouse_test = DANN.preprocess_data(data_mouse_test)
    data_mouse_test = data_mouse_test.drop('organism', axis=1)

    y_label_human = data_human_test.pop('dendrite_type')
    y_label_human = y_label_human.values.astype(np.float32)
    y_label_human = to_categorical(y_label_human, num_classes=n_classes)
    x_human = data_human_test.values.astype(np.float32)
    x_human = scaler.fit_transform(x_human)

    y_label_mouse = data_mouse_test.pop('dendrite_type')
    y_label_mouse = y_label_mouse.values.astype(np.float32)
    y_label_mouse = to_categorical(y_label_mouse, num_classes=n_classes)
    x_mouse = data_mouse_test.values.astype(np.float32)
    x_mouse = scaler.fit_transform(x_mouse)

    n_run = 0

    for layer in layers:
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
                                            print("Num layers: {0}, Network architecture: {1}".format(layer, ds))
                                            print("Activations: {0}, Drop rates: {1}".format(af, drop))
                                            print("Optimizer: {0}, N_epochs: {1}".format(optimizer, epoch))
                                            print("Weight decay: {0}, Learning rate: {1}".format(wd, lr))
                                            print("Batch size: {0}, Lambda: {1}".format(batch, lamda))

                                            DANN = DANNClassifier(db=data,
                                                                  n_layers=layer,
                                                                  weight_decay=wd,
                                                                  dense_size=ds,
                                                                  activation_function=af,
                                                                  learning_rate=lr,
                                                                  drop_rate=drop,
                                                                  batch_size=batch,
                                                                  n_epochs=epoch,
                                                                  optimizer=optimizer,
                                                                  dp_lambda=lamda)
                                            loss, acc = DANN.train_and_test()

                                            # Test network on test human data
                                            print('Human Test:')
                                            human_loss, human_acc = DANN.test_label_predictions(x_human, y_label_human)

                                            # Test network on test mouse data
                                            print('Mouse Test:')
                                            mouse_loss, mouse_acc = DANN.test_label_predictions(x_mouse, y_label_mouse)

                                            results.loc[n_run] = [layer, ds, af, drop, optimizer,
                                                                  epoch, wd, lr, batch, lamda,
                                                                  acc, human_acc, mouse_acc]
                                            n_run += 1
                                            results.to_csv(os.path.join(results_path, 'DANN_results.csv'), index=True)

                                            if human_acc > 0.94 and mouse_acc > 0.94:
                                                print("hyper parameters found!")
                                                print("Results are:")
                                                print("=============================================================")
                                                print("Num layers: {0}, Network architecture: {1}".format(layer, ds))
                                                print("Activations: {0}, Drop rates: {1}".format(af, drop))
                                                print("Optimizer: {0}, N_epochs: {1}".format(optimizer, epoch))
                                                print("Weight decay: {0}, Learning rate: {1}".format(wd, lr))
                                                print("Batch size: {0}, Lambda: {1}".format(batch, lamda))
                                                print("=============================================================")
                                                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--cuda',
                        help='cuda device index',
                        type=int,
                        default=3)
    args = parser.parse_args()
    main(args)
