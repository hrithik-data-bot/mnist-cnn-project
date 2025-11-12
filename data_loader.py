"""module to load image data using tensorflow"""

from typing import Tuple
from tensorflow.keras import datasets  # pyright: ignore[reportMissingImports]
import numpy as np
from matplotlib import pyplot as plt


def load_image_data() -> Tuple:
    """method to load MNIST images data"""

    # Loading the data
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

    # Normalizing the pixel values to [0,1]
    X_train, X_test = X_train/255.0, X_test/255.0

    # Reshape data to include channel dimension (28, 28, 1)
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

    return (X_train, y_train), (X_test, y_test)
