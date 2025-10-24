"""module for model training"""

import time
import numpy as np
from tensorflow.keras import layers, models   # pyright: ignore[reportMissingImports]
from data_loader import load_image_data

(X_train, y_train), (X_test, y_test) = load_image_data()


class CNNModel:
    """class for CNN Model"""

    def __init__(self, x_train: np.ndarray, y_train: np.uint8) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.input_shape = x_train.shape[1:]
        self.num_classes = len(np.unique(y_train))

    def cnn_model(self) -> models.Sequential:
        """method to build, train, and return the model's training history"""

        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(self.num_classes, activation='softmax')  
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"{'*'*20} Model Summary {'*'*20}")
        print(model.summary())
        print("\n")
        print(f"{'*'*20} Model Training Started {'*'*20}")
        start = time.time()

        history = model.fit(
            self.x_train,
            self.y_train,
            epochs=50,
            batch_size=64,
            validation_split=0.1
        )

        end = time.time()
        print(f"{'*'*20} Model Training Ended {'*'*20}")
        print(f"Total training time:- {(end-start)/60} minutes")

        print(f"{'*'*20} Model Evaluation {'*'*20}")
        train_loss, train_acc = model.evaluate(self.x_train, self.y_train, verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Training Accuracy: {train_acc:.4f}, Loss: {train_loss:.4f}")
        print(f"Testing  Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")
        return history, model


if __name__ == "__main__":
    m = CNNModel(X_train, y_train)
    m.cnn_model()

