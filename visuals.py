import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import logging
import log_config # log_config.py
import state

def compute_confusion_matrix(y_true, y_pred, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, predicted_label in zip(y_true, y_pred):
        confusion_matrix[true_label, predicted_label] += 1

    return confusion_matrix

def plot_confusion_matrix(confusion_matrix, classes):

    # clear the current plot
    plt.clf()
    plt.close()


    plt.figure(figsize=(8, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix: " + state.model_name)
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

    # Use the penultimate layer's output as embeddings
    embedding_model = tf.keras.models.Model(
        inputs=state.model.inputs,
        outputs=state.model.layers[-2].output
    )

    embeddings = embedding_model.predict(x_data)
    return embeddings

def reduce_dimensions(embeddings, num_dimensions=2):
    # Simple PCA for dimensionality reduction
    mean = np.mean(embeddings, axis=0)
    embeddings_centered = embeddings - mean
    covariance_matrix = np.cov(embeddings_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_eigenvectors = eigenvectors[:, sorted_indices[:num_dimensions]]
    reduced_embeddings = np.dot(embeddings_centered, top_eigenvectors)
    return reduced_embeddings

def plot_embeddings(embeddings, labels):
    # clear the current plot
    plt.clf()
    plt.close()

    plt.figure(figsize=(10, 10))

    # Plot embeddings with colors corresponding to labels
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
    plt.colorbar(scatter, ticks=range(10), label='Classes')

    plt.title("Scatterplot of Embeddings: " + state.model_name)
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.tight_layout()
    plt.show()