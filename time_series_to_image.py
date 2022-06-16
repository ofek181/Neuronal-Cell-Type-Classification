import numpy as np
from pyts.image import GramianAngularField, RecurrencePlot, MarkovTransitionField

IMG_SIZE = 224


def activity_to_image_gaf(activity: np.array) -> np.ndarray:
    gaf = GramianAngularField(image_size=IMG_SIZE, method='d')
    image = np.empty((IMG_SIZE, IMG_SIZE, 3))
    segment_length = int(len(activity)/3)
    for i in range(3):
        segment = activity[i*segment_length:(i+1)*segment_length]
        image[:, :, i] = gaf.transform(segment.reshape(1, -1)).squeeze()
    return image

# TODO add RecurrencePlot and MarkovTransitionField implementations of activity_to_image
