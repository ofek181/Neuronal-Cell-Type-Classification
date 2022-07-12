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

dir_path = os.path.dirname(os.path.realpath(__file__))


class Tsne(Model):
    def __init__(self, db: pd.DataFrame) -> None:
        """
        Initializes the model.
        :param db: cell ephys features dataframe.
        """
        # encoder = LabelEncoder()
        scaler = StandardScaler()
        db = db.dropna(axis=1, how='all')
        db = db.dropna(axis=0)
        irrelevant_columns = ['layer', 'structure_area_abbrev', 'sampling_rate', 'mean_clipped', 'file_name']
        db = db.drop([x for x in irrelevant_columns if x in db.columns], axis=1)
        db = scaler.fit_transform(db)
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


def get_data() -> pd.DataFrame:
    """
    :return: tuple of merged mouse/human data.
    """
    # get data from directories
    dataframe_path_mouse = dir_path + '/data/dataframe/mouse'
    dataframe_path_human = dir_path + '/data/dataframe/human'
    name = 'extracted_mean_ephys_data.csv'
    data_mouse = pd.read_csv(dataframe_path_mouse + '/' + name)
    data_human = pd.read_csv(dataframe_path_human + '/' + name)
    data_mouse['organism'] = 'mouse'
    data_human['organism'] = 'human'
    data = data_mouse.append(data_human, ignore_index=True)
    return data


def main():
    data = get_data()
    organism_type = data["organism"]
    dendrite_type = data["dendrite_type"]
    data = data.drop(["organism", "dendrite_type"], axis=1)
    tsne = Tsne(data)
    res = tsne.train_and_test()
    tsne.plot_tsne(res, organism_type)


if __name__ == '__main__':
    main()



