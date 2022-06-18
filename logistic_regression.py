import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from classifier import Model
from helper_functions import calculate_metrics


class LogisticClassifier(Model):
    def __init__(self, db: pd.DataFrame):
        db = db.dropna(axis=1, how='all')
        db = db.dropna(axis=0)
        irrelevant_columns = ['layer', 'structure_area_abbrev', 'sampling_rate', 'mean_clipped', 'file_name']
        db = db.drop([x for x in irrelevant_columns if x in db.columns], axis=1)
        super(LogisticClassifier, self).__init__(db)

    def _create_model(self) -> LogisticRegression:
        model = LogisticRegression()
        return model

    def train_and_test(self):
        df = self._data
        df['dendrite_type'] = pd.Categorical(df['dendrite_type'])
        df['dendrite_type'] = df['dendrite_type'].cat.codes
        y = df.pop('dendrite_type')
        y = y.values.astype(float)
        x = df.values

        kf = StratifiedKFold(n_splits=3)
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
            results = [accuracy, f1, precision, recall, roc_auc]
            stats.append(results)
            print('Accuracy: {}, F1: {}, Precision: {}, Recall {}, ROC_AUC: {}'.format(
                accuracy, f1, precision, recall, roc_auc))
        print("Logistic Regression Coefficients: " + str(self.model.coef_))

        # sum_cm = np.asarray([x[2] for x in stats]).sum(axis=0)
        # sum_cm = sum_cm / sum_cm.astype(np.float).sum(axis=1)
        # params = {'a': None}
        # res = {'mean_accuracy': mean_accuracy, 'mean_f1': mean_f1}
        # self._save_results(params, res, sum_cm, 'nb')

    @staticmethod
    def save_results(results: pd.DataFrame, path: str, name: str):
        pass


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataframe_path = dir_path + '\\data\\dataframe'
    dataframe_name = 'extracted_mean_ephys_data.csv'
    df = pd.read_csv(dataframe_path + '\\' + dataframe_name)
    lrclf = LogisticClassifier(df)
    lrclf.train_and_test()


