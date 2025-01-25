import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
import log_config  # Custom logging configuration
import state  # Global application state

def compute_confusion_matrix(y_true, y_pred, num_classes):
    """Compute a confusion matrix for a classification task.

    Args:
        y_true (array-like): True class labels.
        y_pred (array-like): Predicted class labels.
        num_classes (int): Number of classes.

    Returns:
        np.ndarray: A confusion matrix of shape (num_classes, num_classes).
    """
    # clear the current plot
    plt.clf()
    plt.close()

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, predicted_label in zip(y_true, y_pred):
        confusion_matrix[true_label, predicted_label] += 1
    return confusion_matrix

def plot_confusion_matrix(confusion_matrix, classes):
    """Visualize a confusion matrix using a heatmap.

    Args:
        confusion_matrix (np.ndarray): Confusion matrix to visualize.
        classes (list): List of class labels.
    """

    # clear the current plot
    plt.clf()
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {state.model_name}")
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

def extract_embeddings(x_data):
    """Extract embeddings from the penultimate layer of the model.

    Args:
        x_data (np.ndarray): Input data.

    Returns:
        np.ndarray: Extracted embeddings.
    """
    try:
        embedding_model = tf.keras.models.Model(
            inputs=state.model.inputs,
            outputs=state.model.layers[-2].output
        )
        embeddings = embedding_model.predict(x_data)
        return embeddings
    except Exception as e:
        logging.error(f"Error extracting embeddings: {e}")
        return None

def reduce_dimensions(embeddings, num_dimensions=2):
    """Reduce the dimensions of embeddings using PCA.

    Args:
        embeddings (np.ndarray): High-dimensional embeddings.
        num_dimensions (int): Target number of dimensions (default: 2).

    Returns:
        np.ndarray: Reduced embeddings.
    """
    try:
        mean = np.mean(embeddings, axis=0)
        embeddings_centered = embeddings - mean
        covariance_matrix = np.cov(embeddings_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        top_eigenvectors = eigenvectors[:, sorted_indices[:num_dimensions]]
        reduced_embeddings = np.dot(embeddings_centered, top_eigenvectors)
        return reduced_embeddings
    except Exception as e:
        logging.error(f"Error reducing dimensions: {e}")
        return None

def plot_embeddings(embeddings, labels):
    """Visualize embeddings in a 2D scatter plot.

    Args:
        embeddings (np.ndarray): 2D embeddings to visualize.
        labels (array-like): Class labels for coloring the points.
    """

    plt.figure(figsize=(10, 10))

    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
    plt.colorbar(scatter, ticks=range(10), label='Classes')

    plt.title(f"Scatterplot of Embeddings: {state.model_name}")
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("This module provides visualization utilities for model evaluation.")
