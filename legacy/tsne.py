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

# get directories
dir_path = os.path.dirname(os.path.realpath(__file__))
results_path = dir_path + '/results/dann'
model_path = results_path + '/model'
mouse_data = pd.read_csv(dir_path + '/data/mouse/ephys_data.csv')


class Tsne(Model):
    def __init__(self, db: pd.DataFrame) -> None:
        """
        Initializes the model.
        :param db: cell ephys features dataframe.
        """
        db = db.dropna(axis=1, how='all')
        db = db.dropna(axis=0)
        irrelevant_columns = ['dendrite_type', 'neurotransmitter', 'reporter_status', 'layer',
                              'clipped', 'file_name', 'threshold_index', 'peak_index', 'trough_index',
                              'upstroke_index', 'downstroke_index', 'fast_trough_index']
        db = db.drop([x for x in irrelevant_columns if x in db.columns], axis=1, errors='ignore')
        super(Tsne, self).__init__(db)

    def _create_model(self) -> TSNE:
        """
        :return: constructs the t-SNE algorithm.
        """
        model = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, init='pca')
        return model

    def train_and_test(self) -> np.ndarray:
        """
        :return: fit the t-SNE algorithm on the data
        """
        tsne_results = self.model.fit_transform(self.data)
        return tsne_results

    @staticmethod
    def save_results(**kwargs) -> None:
        pass

    @staticmethod
    def plot_tsne(results: np.ndarray, labels: pd.Series) -> None:
        """
        :param results: results of the t-SNE dimensionality reductions.
        :param labels: future hue for the plot.
        :return: plot the first 2 dimensions of the t-SNE.
        """
        res = pd.DataFrame()
        res['tsne-1'] = results[:, 0]
        res['tsne-2'] = results[:, 1]
        res['organism'] = labels
        plt.figure(figsize=(16, 10))
        sns.scatterplot(x="tsne-1", y="tsne-2", hue="organism", data=res)
        plt.title("t-SNE plot of the extracted cell type features data")
        plt.show()


def main():
    transgenic_targeting = mouse_data.pop('transgenic_line')
    tsne = Tsne(mouse_data)
    res = tsne.train_and_test()
    tsne.plot_tsne(res, transgenic_targeting)


if __name__ == '__main__':
    main()

    # TODO optimize
    # TODO TSne for cre lines



