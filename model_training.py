"""module for model training"""

import time
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models # pyright: ignore[reportMissingImports]
from data_loader import load_image_data

# Load image data
(X_train_full, y_train_full), (X_test, y_test) = load_image_data()

# Combine train and test to reshuffle and resplit (optional, for uniform distribution)
X_all = np.concatenate((X_train_full, X_test), axis=0)
y_all = np.concatenate((y_train_full, y_test), axis=0)

# Split into 80% train, 10% val, 10% test
X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Data Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")


class CNNModel:
    """class for CNN Model"""

    def __init__(self, x_train: np.ndarray, y_train: np.uint8,
                 x_val: np.ndarray, y_val: np.uint8,
                 x_test: np.ndarray, y_test: np.uint8) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.input_shape = x_train.shape[1:]
        self.num_classes = len(np.unique(y_train))

    def cnn_model(self):
        """Build, train, evaluate, and visualize CNN"""

        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),

            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"\n{'*' * 20} Model Summary {'*' * 20}")
        model.summary()
        print(f"\n{'*' * 20} Model Training Started {'*' * 20}")
        start = time.time()

        history = model.fit(
            self.x_train, self.y_train,
            epochs=5,
            batch_size=64,
            validation_data=(self.x_val, self.y_val),
            verbose=1
        )

        end = time.time()
        print(f"Total training time: {(end - start) / 60:.2f} minutes")
        print(f"{'*' * 20} Model Training Ended {'*' * 20}\n")

        # Evaluate model on all splits
        print(f"{'*' * 20} Model Evaluation {'*' * 20}")
        train_loss, train_acc = model.evaluate(self.x_train, self.y_train, verbose=1)
        val_loss, val_acc = model.evaluate(self.x_val, self.y_val, verbose=1)
        test_loss, test_acc = model.evaluate(self.x_test, self.y_test, verbose=1)

        print(f"\nFinal Performance:")
        print(f"Training   -> Accuracy: {train_acc * 100:.3f}%, Loss: {train_loss:.4f}")
        print(f"Validation -> Accuracy: {val_acc * 100:.3f}%, Loss: {val_loss:.4f}")
        print(f"Testing    -> Accuracy: {test_acc * 100:.3f}%, Loss: {test_loss:.4f}")

        # Plot and save accuracy/loss
        self.plot_history(history, save_path="accuracy_loss_plot.png")

        # Plot and save confusion matrix
        self.plot_confusion_matrix(model, save_path="confusion_matrix.png")

        # Save the model
        model.save("mnist_cnn_final_model.h5")
        print("\nModel saved successfully as 'mnist_cnn_final_model.h5'")

        return history, model

    @staticmethod
    def plot_history(history, save_path="accuracy_loss_plot.png"):
        """Plot training and validation accuracy/loss per epoch and save"""
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))

        # Loss Plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, 'b-', label='Training Loss')
        plt.plot(epochs, val_loss, 'r--', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Accuracy Plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, acc, 'b-', label='Training Accuracy')
        plt.plot(epochs, val_acc, 'r--', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Accuracy/Loss plot saved as '{save_path}'")

    def plot_confusion_matrix(self, model, save_path="confusion_matrix.png"):
        """Compute and plot confusion matrix without sklearn and save"""
        y_pred = np.argmax(model.predict(self.x_test), axis=1)
        conf_mat = np.zeros((self.num_classes, self.num_classes), dtype=int)

        for true, pred in zip(self.y_test, y_pred):
            conf_mat[true, pred] += 1

        print("\nConfusion Matrix:")
        print(conf_mat)

        plt.figure(figsize=(7, 6))
        plt.imshow(conf_mat, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.colorbar()

        # Add numbers on each cell
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                plt.text(j, i, str(conf_mat[i, j]),
                         ha='center', va='center',
                         color='white' if conf_mat[i, j] > conf_mat.max() / 2 else 'black')

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Confusion matrix plot saved as '{save_path}'")
