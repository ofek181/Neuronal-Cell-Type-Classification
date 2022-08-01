import os
import pandas as pd
from dnn import DNNClassifier
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
    def __init__(self, title: str = None) -> None:
        """
        :param title: name of the model.
        """
        self.model = tf.keras.models.load_model(filepath=model_path)
        self.title = title

    def explain_model(self, size: tuple = (10, 10)) -> None:
        """
        :param size: size of the summary plot.
        :return: shap.summary_plot for a multi-output model.
        """
        dummy = DNNClassifier(db=data_mouse, n_layers=0, weight_decay=0, dense_size=[], activation_function=[],
                              learning_rate=0, drop_rate=[], batch_size=0, n_epochs=0, optimizer='adam')
        data = dummy.preprocess_data(data_mouse)

        features = list(data.columns)
        features.remove('dendrite_type')

        x_train, y_train, x_val, y_val, x_test, y_test = dummy.split_train_val_test(data)

        kernel_explainer = shap.KernelExplainer(self.model, x_test)
        shapley_vals = kernel_explainer.shap_values(x_test)
        shap.summary_plot(shap_values=shapley_vals, features=x_test, feature_names=features, plot_size=size,
                          color=plt.get_cmap("Pastel1"), title=self.title, class_names=["aspiny", "spiny"])


def main():
    models = ['Mouse Feature Importance', 'Human Feature Importance']
    for model in models:
        xai = Explainer(title=model)
        xai.explain_model()


if __name__ == '__main__':
    device = get_device()
    with tf.device(device):
        main()
