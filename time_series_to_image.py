import numpy as np
from pyts.image import GramianAngularField
from consts import GAF_IMAGE_SIZE


def activity_to_image_gaf(activity: np.array) -> np.ndarray:
    """
    :param activity: time series response for the sweep data of the cell.
    :return: gramian angular field image of the activity time series.
    """
    gaf = GramianAngularField(image_size=GAF_IMAGE_SIZE, method='d')
    image = np.empty((GAF_IMAGE_SIZE, GAF_IMAGE_SIZE, 3))
    segment_length = int(len(activity)/3)
    for i in range(3):
        segment = activity[i*segment_length:(i+1)*segment_length]
        image[:, :, i] = gaf.fit_transform(segment.reshape(1, -1))
    return image

# TODO try and make a gray-scaled image of gaf instead of RGB

