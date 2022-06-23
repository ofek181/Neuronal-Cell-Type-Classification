import os
import pandas as pd
import numpy as np
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from classifier import Model
from helper_functions import calculate_metrics

# TODO understand how to input the images into the CNN, maybe using a dataframe?
# TODO understand how to convert RGB images to dataframes
# TODO understand how to get the corresponding label to each image


class ConvNet(Model):
    def __init__(self, db=None) -> None:
        """
        :param db: cell ephys features dataframe.
        """
        super(ConvNet, self).__init__(db)

    def _create_model(self) -> Sequential:
        """
        :return: creates a logistic regression classifier
        """
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        return model

    def train_and_test(self, images_path: str = 'data/images/gaf') -> pd.DataFrame:
        """
        :return: results of the logistic regression classifier on the testing data.
        """
        stats = []
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        generator = ImageDataGenerator(validation_split=0.25)
        train_generator = generator.flow_from_directory(images_path,
                                                        target_size=(224, 224),
                                                        color_mode='rgb',
                                                        batch_size=8,
                                                        class_mode='binary',
                                                        shuffle=True,
                                                        subset='training')
        validation_generator = generator.flow_from_directory(images_path,
                                                             target_size=(224, 224),
                                                             color_mode='rgb',
                                                             batch_size=1,
                                                             class_mode='binary',
                                                             shuffle=False,
                                                             subset='validation')
        step_size_train = train_generator.n // train_generator.batch_size
        step_size_val = validation_generator.n // validation_generator.batch_size
        self.model.fit_generator(generator=train_generator, steps_per_epoch=step_size_train,
                                 validation_data=validation_generator, validation_steps=step_size_val,
                                 epochs=10)
        pred = self.model.predict_generator(validation_generator, steps=validation_generator.n)
        pred = np.where(pred > 0.5, 1.0, 0.0)
        accuracy, f1, precision, recall, roc_auc = calculate_metrics(validation_generator.classes, pred)
        stats.append([accuracy, f1, precision, recall, roc_auc])
        results = pd.DataFrame(stats, columns=['Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC AUC'])
        return results

    def save_results(self, results: pd.DataFrame, path: str, name: str) -> None:
        """
        :param results: results on the testing data.
        :param path: path to save file.
        :param name: name of the file.
        :return: None.
        """
        self.model.save(path)
        results.to_csv(os.path.join(path, name))


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cnn = ConvNet()
    cnn.train_and_test()



