import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# File name for saving and loading the trained model
model_file = "mnist_model.keras"

def load_or_train_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocess the data
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_test = y_test

    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        raise FileNotFoundError(f"{model_file} not found. Please train the model first by running main.py")

    return model, x_test, y_test

def plot_confusion_matrix(confusion_matrix, classes):
    plt.figure(figsize=(8, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Display the values in the confusion matrix
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > confusion_matrix.max() / 2. else "black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

def compute_confusion_matrix(y_true, y_pred, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, predicted_label in zip(y_true, y_pred):
        confusion_matrix[true_label, predicted_label] += 1

    return confusion_matrix

def main():
    model, x_test, y_test = load_or_train_model()

    # Predict labels
    predictions = model.predict(x_test)
    y_pred = np.argmax(predictions, axis=1)

    # Compute confusion matrix
    confusion_matrix = compute_confusion_matrix(y_test, y_pred, num_classes=10)

    # Plot confusion matrix
    plot_confusion_matrix(confusion_matrix, classes=[str(i) for i in range(10)])

if __name__ == "__main__":
    main()
