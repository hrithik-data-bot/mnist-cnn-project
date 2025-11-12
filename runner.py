"""runner module to run the code"""

# runner.py

from model_training import CNNModel  # Import your class
from data_loader import load_image_data
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt

def prepare_data():
    """Load and split data into train, validation, and test sets"""
    (X_train_full, y_train_full), (X_test, y_test) = load_image_data()

    # Optional: Combine train and test to reshuffle and resplit for uniform distribution
    X_all = np.concatenate((X_train_full, X_test), axis=0)
    y_all = np.concatenate((y_train_full, y_test), axis=0)

    # Split into 80% train, 10% val, 10% test
    X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Data Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def visualize_images_with_labels(save_path="sample_images.png"):
    """Visualize some sample images with labels and save the figure"""
    (X_train, y_train), _ = load_image_data()
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))

    for i, ax in enumerate(axes.flat):
        ax.imshow(X_train[i].squeeze(), cmap='gray')  # Remove channel dim
        ax.set_title(f"Label: {y_train[i]}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Sample images saved as '{save_path}'")

if __name__ == "__main__":
    # Step 1: Visualize and save sample images
    visualize_images_with_labels(save_path="sample_images.png")

    # Step 2: Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()

    # Step 3: Initialize CNN model
    cnn = CNNModel(X_train, y_train, X_val, y_val, X_test, y_test)

    # Step 4: Train, evaluate, plot, save model & confusion matrix
    history, model = cnn.cnn_model()

