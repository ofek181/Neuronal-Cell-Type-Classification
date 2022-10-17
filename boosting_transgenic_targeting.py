import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from helper_functions import calculate_metrics_multiclass

# get directories
dir_path = os.path.dirname(os.path.realpath(__file__))
data = pd.read_csv(dir_path + '/data/mouse/ephys_data.csv')
results_mouse = dir_path + '/results/mlp/mouse/multilabel_classification'

# Cancel randomness for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1)
random.seed(1)

# TODO add docstring


class RFClassifier:
    def __init__(self, db: pd.DataFrame, learning_rate: float = 0.1, n_estimators: int = 100,
                 max_depth: int = 3, max_features: int = None) -> None:
        self.model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators,
                                                max_depth=max_depth, max_features=max_features)
        db = self.process_data(db)
        self.y, self.x, self.class_names = db.pop('transgenic_line'), db, {}
        self.encode()
        self.scale()
        self.x_train, self.y_train, self.x_test, self.y_test = self.split_train_test()

    def train(self) -> None:
        self.model.fit(self.x_train, self.y_train)

    def test(self) -> None:
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
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, train_size=0.75, random_state=42)
        return x_train, y_train, x_test, y_test

    @staticmethod
    def process_data(db: pd.DataFrame) -> pd.DataFrame:
        db = db.dropna(axis=1, how='all')
        db = db.dropna(axis=0)
        irrelevant_columns = ['dendrite_type', 'neurotransmitter', 'reporter_status', 'layer', 'file_name']
        db = db.drop([x for x in irrelevant_columns if x in db.columns], axis=1, errors='ignore')
        return db

    @staticmethod
    def plot_history(history) -> None:
        pass

    @staticmethod
    def save_results(results: pd.DataFrame, path: str, name: str) -> None:
        pass


def main():
    clf = RFClassifier(data)
    clf.train()
    clf.test()


if __name__ == '__main__':
    main()

