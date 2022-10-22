import copy
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_tree, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from helper_functions import calculate_metrics_multiclass

# get directories
dir_path = os.path.dirname(os.path.realpath(__file__))
data = pd.read_csv(dir_path + '/data/mouse/ephys_data.csv')
model_name = os.path.abspath(dir_path + '/results/xgboost/xgboost_model.json')


class XGB:
    """
    XGBoost classifier
    """
    def __init__(self, db: pd.DataFrame, n_estimators: int = 50, max_depth: int = 8,
                 max_leaves: int = 0, learning_rate: float = 0.25) -> None:
        """
        :param db: data used for the classification.
        :param n_estimators: number of gradient boosted trees. Equivalent to number of boosting rounds.
        :param max_depth: maximum tree depth for base learners.
        :param max_leaves: maximum number of leaves; 0 indicates no limit.
        :param learning_rate: boosting learning rate (xgb’s “eta”).
        """
        db = self.process_data(db)
        self.y, self.x, self.class_names = db.pop('transgenic_line'), db, {}
        self.feature_names = db.columns.tolist()
        self.encode()
        self.scale()
        self.x_train, self.y_train, self.x_test, self.y_test = self.split_train_test()
        self.model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, max_leaves=max_leaves,
                                   learning_rate=learning_rate, verbosity=1,
                                   random_state=1)

    def train(self) -> None:
        """
        fit the XGBoost classifier to the training data.
        """
        self.model.fit(self.x_train, self.y_train)

    def test(self) -> None:
        """
        test the classifier over the test sample and obtain metrics.
        """
        self.model.get_booster().feature_names = self.feature_names
        print("XGBoost")
        print("Learning Rate: " + str(self.model.learning_rate) + ", Max Depth: " + str(self.model.max_depth) +
              ", N Estimators: " + str(self.model.n_estimators), ", Max Leaves: " + str(self.model.max_leaves))
        y_pred = self.model.predict(self.x_test)
        y_pred_proba = self.model.predict_proba(self.x_test)
        accuracy, f1, precision, recall, roc_auc = calculate_metrics_multiclass(self.y_test, y_pred, y_pred_proba)
        print('--------------------------------------------------------------')
        print("Accuracy: " + str(accuracy))
        print("F1 Score: " + str(f1))
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("ROC AUC: " + str(roc_auc))

    def scale(self) -> None:
        """
        scale the data using StandardScaler.
        """
        scaler = StandardScaler()
        self.x = scaler.fit_transform(self.x)

    def encode(self) -> None:
        """
        encode the labels using LabelEncoder.
        """
        encoder = LabelEncoder()
        self.y = pd.Categorical(self.y)
        self.class_names = dict(enumerate(self.y.categories))
        self.y = encoder.fit_transform(self.y)

    def split_train_test(self) -> tuple:
        """
        split the data into train and test.
        """
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, train_size=0.75, random_state=1)
        return x_train, y_train, x_test, y_test

    def visualize_tree(self) -> None:
        """
        visualize one of the boosting trees.
        """
        fig, ax = plt.subplots(figsize=(30, 30))
        plot_tree(self.model, ax=ax)
        plt.draw()

    def visualize_feature_importance(self) -> None:
        """
        Visualize the feature importance for the XGBoost classifier.
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        plot_importance(booster=self.model, ax=ax, title='XGB feature importance', max_num_features=5)
        plt.draw()

    def save_model(self) -> None:
        """
        save model as json file.
        """
        self.model.save_model(model_name)

    def load_model(self) -> None:
        """
        load model from a json file.
        """
        self.model.load_model(model_name)

    def plot_confusion_matrix(self) -> None:
        """
        plot the confusion matrix.
        """
        y_pred = self.model.predict(self.x_test)

        def reverse_labels(tup: tuple) -> list:
            """
            :param tup: tuple of label.
            :return: names of labels.
            """
            return [self.class_names[x] for x in tup]

        y_true_labeled, y_pred_labeled = reverse_labels(tuple(self.y_test)), reverse_labels(tuple(y_pred))
        matrix = confusion_matrix(y_true_labeled, y_pred_labeled)
        df_cm = pd.DataFrame(matrix, columns=np.unique(y_true_labeled), index=np.unique(y_true_labeled))
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        plt.figure()
        cmap = sns.cubehelix_palette(light=0.9, as_cmap=True)
        cm_normalized = df_cm.div(df_cm.sum(axis=0), axis=1)
        sns.heatmap(cm_normalized, cbar=False, annot=True, cmap=cmap, square=True, fmt='.1%', annot_kws={'size': 10})
        plt.title('XGBoost Neuron Classification')
        plt.tight_layout()
        plt.draw()

    @staticmethod
    def process_data(db: pd.DataFrame) -> pd.DataFrame:
        """
        :param db: data to be processed.
        preprocess the data.
        """
        db = db.dropna(axis=1, how='all')
        db = db.dropna(axis=0)
        irrelevant_columns = ['dendrite_type', 'neurotransmitter', 'reporter_status', 'layer', 'file_name']
        db = db.drop([x for x in irrelevant_columns if x in db.columns], axis=1, errors='ignore')
        return db


def grid_search():
    n_estimators = [5, 25, 50, 75]
    max_depth = [5, 10, 15, 20, 30]
    max_leaves = [0, 5, 10, 20, 30]
    lr = [0.1, 0.25, 0.5]
    n_iter = len(n_estimators) * len(max_depth) * len(max_leaves) * len(lr)
    iter: int = 0
    best = XGB(data)
    best.train()
    for est in n_estimators:
        for depth in max_depth:
            for leaves in max_leaves:
                for eta in lr:
                    clf = XGB(data, est, depth, leaves, eta)
                    clf.train()
                    print(clf.model.score(clf.x_test, clf.y_test))
                    if clf.model.score(clf.x_test, clf.y_test) >= best.model.score(clf.x_test, clf.y_test):
                        best = copy.copy(clf)
                    print("Iteration number: " + str(iter) + " Out of: " + str(n_iter) +
                          " ,best score: " + str(best.model.score(clf.x_test, clf.y_test)))
                    iter += 1
    best.save_model()


def main():
    grid_search()
    clf = XGB(data)
    clf.load_model()
    clf.test()
    clf.visualize_tree()
    clf.visualize_feature_importance()
    clf.plot_confusion_matrix()
    plt.show()


if __name__ == '__main__':
    main()

