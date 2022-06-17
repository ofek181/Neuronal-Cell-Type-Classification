import numpy as np
from pyts.image import GramianAngularField

IMG_SIZE = 224


def activity_to_image_gaf(activity: np.array) -> np.ndarray:
    """
    :param activity: time series response for the sweep data of the cell.
    :return: gramian angular field image of the activity time series.
    """
    gaf = GramianAngularField(image_size=IMG_SIZE, method='d')
    image = np.empty((IMG_SIZE, IMG_SIZE, 3))
    segment_length = int(len(activity)/3)
    for i in range(3):
        segment = activity[i*segment_length:(i+1)*segment_length]
        image[:, :, i] = gaf.fit_transform(segment.reshape(1, -1))
    return image

