import os
import numpy as np
import pandas as pd
from dann import DANNClassifier, GradientReversal, get_data
from gpu_check import get_device
from sklearn.preprocessing import StandardScaler
import shap
import tensorflow as tf
from keras.models import Model
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = dir_path + '/results/dann/model'


class Explainer:
    """
    Explainer uses Shapley Values in order to explain feature importance in a model.
    """
    def __init__(self) -> None:
        """
        initializes the model with the label classifier as output
        """
        original_model = tf.keras.models.load_model(filepath=model_path,
                                                    custom_objects={"GradientReversal": GradientReversal})
        self.model = Model(inputs=original_model.input, outputs=original_model.output[0])
        self.data, self.data_human_test, self.data_mouse_test = get_data()

    def explain_model(self) -> None:
        """
        :return: shap.summary_plot for a multi-output model.
        """
        self.data = pd.concat([self.data_human_test[0:50], self.data_mouse_test[0:50]], ignore_index=True)
        dummy = DANNClassifier(db=self.data, weight_decay=0, dense_size=[], activation_function=[], learning_rate=0,
                               drop_rate=[0], batch_size=0, n_epochs=0, optimizer='adam', lamda=0)
        data = self._process_data(dummy.data)

        # get features' names
        features = list(dummy.data.columns)

        # use the kernel explainer to get the Shapley values
        kernel_explainer = shap.KernelExplainer(self.model, data)
        shapley_values = kernel_explainer.shap_values(data)

        # draw summary plot
        self._draw_summary_plot(shapley_values, data, features)

    @staticmethod
    def _process_data(data):
        """
        :param data: data to be processed
        :return: clean, processed data without labels
        """
        scaler = StandardScaler()
        y_label = data.pop('dendrite_type')
        y_label = y_label.values.astype(np.float32)
        y_domain = data.pop('organism')
        y_domain = y_domain.values.astype(np.float32)
        x = data.values.astype(np.float32)
        x = scaler.fit_transform(x)
        return x

    @staticmethod
    def _draw_summary_plot(shapley_values: list, data: np.ndarray,
                           features: list, size: tuple = (10, 10), title: str = 'DANN feature importance') -> None:
        """
        :param shapley_values: shap values obtained from the explainer.
        :param data: data that was tested.
        :param features: features' names
        :param size: size of the plot
        :param title: title of the plot
        """
        plt.figure()
        shap.summary_plot(shap_values=shapley_values, features=data, feature_names=features, plot_size=size,
                          show=False, color=plt.get_cmap("Pastel1"), class_names=["aspiny", "spiny"])
        plt.title(title)
        plt.tight_layout()
        plt.draw()


def main():
    model = Explainer()
    model.explain_model()
    plt.show()


if __name__ == '__main__':
    device = get_device()
    with tf.device(device):
        main()

    # TODO optimize
