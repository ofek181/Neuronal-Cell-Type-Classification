import os
import pandas as pd
import numpy as np
from neural_net_dendrite_type import DNNClassifier
from gpu_check import get_device
import shap
import tensorflow as tf
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = dir_path + '/results/MLP/mouse/model'
dataframe_path_mouse = dir_path + '/data/dataframe/mouse'
dataframe_path_human = dir_path + '/data/dataframe/human'

data_mouse = pd.read_csv(dataframe_path_mouse + '/extracted_mean_ephys_data.csv')
data_human = pd.read_csv(dataframe_path_human + '/extracted_mean_ephys_data.csv')


class Explainer:
    """
    Explainer uses Shapley Values in order to explain feature importance in a model.
    """
    def __init__(self, title: str = None, data: pd.DataFrame = None) -> None:
        """
        :param title: name of the model.
        :param data: type of data to be processed.
        """
        self.model = tf.keras.models.load_model(filepath=model_path)
        self.title = title
        self.data = data

    def explain_model(self) -> None:
        """
        :return: shap.summary_plot for a multi-output model.
        """
        # use a dummy classifier to preprocess the data
        dummy = DNNClassifier(db=self.data, n_layers=0, weight_decay=0, dense_size=[], activation_function=[],
                              learning_rate=0, drop_rate=[], batch_size=0, n_epochs=0, optimizer='adam')
        data = dummy.preprocess_data(self.data)

        # get features' names
        features = list(data.columns)
        features.remove('dendrite_type')

        # split into train, val and test
        x_train, y_train, x_val, y_val, x_test, y_test = dummy.split_train_val_test(data)

        # use the kernel explainer to get the Shapley values
        kernel_explainer = shap.KernelExplainer(self.model, x_test)
        shapley_values = kernel_explainer.shap_values(x_test)

        # draw summary plot
        self._draw_summary_plot(shapley_values, x_test, features)

    def _draw_summary_plot(self, shapley_values: list, data: np.ndarray,
                           features: list, size: tuple = (10, 10)) -> None:
        """
        :param shapley_values: shap values obtained from the explainer.
        :param data: data that was tested.
        :param features: features' names
        :param size: size of the plot
        """
        plt.figure()
        shap.summary_plot(shap_values=shapley_values, features=data, feature_names=features, plot_size=size,
                          show=False, color=plt.get_cmap("Pastel1"), class_names=["aspiny", "spiny"])
        plt.title(self.title)
        plt.tight_layout()
        plt.draw()


def main():
    models = ['Human Feature Importance', 'Mouse Feature Importance']
    data = [data_human, data_mouse]
    for idx, val in enumerate(models):
        xai = Explainer(title=val, data=data[idx])
        xai.explain_model()

    plt.show()


if __name__ == '__main__':
    device = get_device()
    with tf.device(device):
        main()

    # TODO optimize
