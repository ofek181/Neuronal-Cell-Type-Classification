import copy
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_tree, plot_importance, DMatrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from helper_functions import calculate_metrics_multiclass

# get directories
dir_path = os.path.dirname(os.path.realpath(__file__))
data = pd.read_csv(dir_path + '/data/mouse/ephys_data.csv')
model_name = os.path.abspath(dir_path + '/results/xgboost/xgboost_model')


# TODO add docstring


class XGB:
    def __init__(self, db: pd.DataFrame, n_estimators: int = 50, max_depth: int = 8,
                 max_leaves: int = 0, learning_rate: float = 0.25) -> None:
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
        self.model.fit(self.x_train, self.y_train)
        self.model.get_booster().feature_names = self.feature_names

    def test(self) -> None:
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
        scaler = StandardScaler()
        self.x = scaler.fit_transform(self.x)

    def encode(self) -> None:
        encoder = LabelEncoder()
        self.y = pd.Categorical(self.y)
        self.class_names = dict(enumerate(self.y.categories))
        self.y = encoder.fit_transform(self.y)

    def split_train_test(self) -> tuple:
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, train_size=0.75, random_state=1)
        return x_train, y_train, x_test, y_test

    def visualize_tree(self) -> None:
        fig, ax = plt.subplots(figsize=(30, 30))
        plot_tree(self.model, ax=ax)
        plt.draw()

    def visualize_feature_importance(self) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        plot_importance(booster=self.model, ax=ax, title='XGB feature importance', max_num_features=5)
        plt.draw()

    def save_model(self) -> None:
        self.model.save_model(model_name)

    def load_model(self) -> None:
        self.model.load_model(model_name)

    @staticmethod
    def process_data(db: pd.DataFrame) -> pd.DataFrame:
        db = db.dropna(axis=1, how='all')
        db = db.dropna(axis=0)
        irrelevant_columns = ['dendrite_type', 'neurotransmitter', 'reporter_status', 'layer', 'file_name']
        db = db.drop([x for x in irrelevant_columns if x in db.columns], axis=1, errors='ignore')
        return db


def grid_search():
    n_estimators = [10, 50, 100]
    max_depth = [3, 5, 10, 15, 30]
    max_leaves = [0, 8, 20, 40]
    lr = [0.1, 0.2, 0.5]
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
    plt.show()


if __name__ == '__main__':
    main()

