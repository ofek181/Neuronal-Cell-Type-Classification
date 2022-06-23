import os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from classifier import Model
import warnings
warnings.simplefilter("ignore")


class Tsne(Model):
    def __init__(self, db: pd.DataFrame) -> None:
        """
        :param db: cell ephys features dataframe.
        """
        encoder = LabelEncoder()
        scaler = StandardScaler()
        db = db.dropna(axis=1, how='all')
        db = db.dropna(axis=0)
        irrelevant_columns = ['layer', 'structure_area_abbrev', 'sampling_rate', 'mean_clipped', 'file_name']
        db = db.drop([x for x in irrelevant_columns if x in db.columns], axis=1)
        db['dendrite_type'] = encoder.fit_transform(db['dendrite_type'])
        db = scaler.fit_transform(db)
        super(Tsne, self).__init__(db)

    def _create_model(self) -> TSNE:
        """
        :return: creates a logistic regression classifier
        """
        model = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, init='pca')
        return model

    def train_and_test(self) -> np.ndarray:
        """
        :return: results of the logistic regression classifier on the testing data.
        """
        tsne_results = self.model.fit_transform(self.data)
        return tsne_results

    @staticmethod
    def save_results() -> None:
        pass

    def plot_tsne(self, results: np.ndarray) -> None:
        """
        :param results: results of the TSNE algorithm.
        :return: plot an image of the results.
        """
        tsne_res = pd.DataFrame()
        tsne_res['tsne-1'] = results[:, 0]
        tsne_res['tsne-2'] = results[:, 1]
        plt.figure(figsize=(16, 10))
        sns.scatterplot(x="tsne-1", y="tsne-2", data=tsne_res, legend="full", alpha=0.3)
        plt.title("t-SNE plot of the extracted cell type features data")
        plt.show()


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataframe_path = dir_path + '\\data\\dataframe'
    dataframe_name = 'extracted_mean_ephys_data.csv'
    data = pd.read_csv(dataframe_path + '\\' + dataframe_name)
    tsne = Tsne(data)
    res = tsne.train_and_test()
    tsne.plot_tsne(res)



