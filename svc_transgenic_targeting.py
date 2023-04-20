import pickle
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from helper_functions import calculate_metrics_multiclass

# get directories
dir_path = os.path.dirname(os.path.realpath(__file__))
data = pd.read_csv(dir_path + '/data/mouse/ephys_data.csv')
model_name = os.path.abspath(dir_path + '/results/svc/svc.pkl')

# style format
plt.style.use(dir_path + '/plot_style.txt')


class SVClassifier:
    """
    Logistic Regression Classifier
    """
    def __init__(self, db: pd.DataFrame) -> None:
        """
        :param db: data used for the classification.
        """
        db = self.process_data(db)
        self.y, self.x, self.class_names = db.pop('transgenic_line'), db, {}
        self.feature_names = db.columns.tolist()
        self.encode()
        self.scale()
        self.x_train, self.y_train, self.x_test, self.y_test = self.split_train_test()
        self.model = SVC(random_state=1, probability=True)

    def train(self) -> None:
        """
        fit the SVC classifier to the training data.
        """
        self.model.fit(self.x_train, self.y_train)

    def test(self) -> None:
        """
        test the classifier over the test sample and obtain metrics.
        """
        print("SVC")
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
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, train_size=0.2, random_state=1)
        return x_train, y_train, x_test, y_test

    def save_model(self) -> None:
        """
        save model as pickle file.
        """
        pickle.dump(self.model, open(model_name, 'wb'))

    def load_model(self) -> None:
        """
        load model from a json file.
        """
        self.model = pickle.load(open(model_name, 'rb'))

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
        plt.title('SVC Neuron Classification')
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


def main():
    clf = SVClassifier(data)
    clf.train()
    print(clf.model.score(clf.x_test, clf.y_test))
    clf.save_model()
    clf.load_model()
    clf.test()
    clf.plot_confusion_matrix()
    plt.show()


if __name__ == '__main__':
    main()

