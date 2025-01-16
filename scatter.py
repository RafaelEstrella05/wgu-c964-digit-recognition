import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import tensorflow as tf

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

def extract_embeddings(model, x_data):
    # Use the penultimate layer's output as embeddings
    embedding_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=model.layers[-2].output
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
    plt.figure(figsize=(10, 10))

    # Plot embeddings with colors corresponding to labels
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
    plt.colorbar(scatter, ticks=range(10), label='Classes')

    plt.title("Scatterplot of Embeddings")
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.tight_layout()
    plt.show()

def main():
    model, x_test, y_test = load_or_train_model()

    # Extract embeddings
    embeddings = extract_embeddings(model, x_test)

    # Reduce dimensions using PCA
    reduced_embeddings = reduce_dimensions(embeddings)

    # Plot scatterplot of embeddings
    plot_embeddings(reduced_embeddings, y_test)

if __name__ == "__main__":
    main()
