"""module for model training"""

import numpy as np
from tensorflow.keras import layers, models   # pyright: ignore[reportMissingImports]
from data_loader import load_image_data

(X_train, y_train), (X_test, y_test) = load_image_data()


class CNNModel:
    """class for CNN Model"""

    def __init__(self, x_train: np.ndarray, y_train: np.uint8) -> None:
        self.x_train = x_train
        self.y_train = y_train


    def model(self) -> models.Sequential:
        """model method return model"""

        model = models.Sequential([
                                    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                                    layers.MaxPooling2D((2, 2)),

                                    layers.Conv2D(64, (3, 3), activation='relu'),
                                    layers.MaxPooling2D((2, 2)),

                                    layers.Flatten(),
                                    layers.Dense(128, activation='relu'),
                                    layers.Dropout(0.4),
                                    layers.Dense(10, activation='softmax')
                                ])
        return model


    def train(self):
        """train method for model"""

        pass
