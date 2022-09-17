import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from classifier import Model
from helper_functions import calculate_metrics

# get directories
dir_path = os.path.dirname(os.path.realpath(__file__))
data_mouse = pd.read_csv(dir_path + '/data/mouse/ephys_data.csv')
data_human = pd.read_csv(dir_path + '/data/human/ephys_data.csv')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class EfficientClassifier(Model):
    """
    EfficientClassifier classifies cells' dendrite types using 2 handpicked features based on their Shapley values.
    """
    def __init__(self, db: pd.DataFrame) -> None:
        """
        :param db: cell ephys features dataframe.
        """
        db = db.dropna(axis=1, how='all')
        db = db.dropna(axis=0)
        db = db[['fast_trough_v', 'width', 'dendrite_type']]
        super(EfficientClassifier, self).__init__(db)

    def _create_model(self) -> LogisticRegression:
        """
        :return: creates a logistic regression classifier
        """
        model = LogisticRegression()
        return model

    def train_and_test(self) -> pd.DataFrame:
        """
        :return: results of the logistic regression classifier on the testing data.
        """
        df = self.data
        df['dendrite_type'] = pd.Categorical(df['dendrite_type'])
        df['dendrite_type'] = df['dendrite_type'].cat.codes
        y = df.pop('dendrite_type')
        y = y.values.astype(float)
        x = df.values
        kf = StratifiedKFold(n_splits=5)
        stats = []
        for train_index, test_index in kf.split(x, y):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x[train_index])
            y_train = y[train_index]
            x_test = scaler.transform(x[test_index])
            y_test = y[test_index]
            self.model.fit(x_train, y_train)
            y_pred = self.model.predict(x_test)
            accuracy, f1, precision, recall, roc_auc = calculate_metrics(y_test, y_pred)
            stats.append([accuracy, f1, precision, recall, roc_auc])
        results = pd.DataFrame(stats, columns=['Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC AUC'])
        print('--------------------------------------------------------------')
        print("Logistic Regression Coefficients: " + str(self.model.coef_))
        print('--------------------------------------------------------------')
        print("Mean Accuracy: " + str(results["Accuracy"].mean()))
        print("Mean F1 Score: " + str(results["F1 Score"].mean()))
        print("Mean Precision: " + str(results["Precision"].mean()))
        print("Mean Recall: " + str(results["Recall"].mean()))
        print("Mean ROC AUC: " + str(results["ROC AUC"].mean()))
        return results

    def plot_data(self, title: str = None) -> None:
        """
        :param title: the title of the plot
        :return: a scatter plot of the reduced data
        """
        sns.scatterplot(data=self.data, x="fast_trough_v", y="width", hue="dendrite_type")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def save_results(results: pd.DataFrame, path: str, name: str) -> None:
        """
        :param results: results on the testing data.
        :param path: path to save file.
        :param name: name of the file.
        :return: None.
        """
        results.to_csv(os.path.join(path, name))


def main():
    data = [data_human, data_mouse]
    for idx, val in enumerate(['Human data', 'Mouse data']):
        print("====================================")
        print(val)
        clf = EfficientClassifier(data[idx])
        clf.plot_data(val)
        clf.train_and_test()


if __name__ == '__main__':
    main()



