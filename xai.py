from dann import DANNClassifier, GradientReversal, get_data
from gpu_check import get_device
import shap
import tensorflow as tf
import keras
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = dir_path + 'results/DANN/model'


class ExplainableAI:
    def __init__(self) -> None:
        self.model = keras.models.load_model(filepath=model_path, custom_objects={"GradientReversal": GradientReversal})

    def explain_model(self) -> None:
        pass


def main():
    explainer = ExplainableAI


if __name__ == '__main__':
    device = get_device()
    with tf.device(device):
        main()
