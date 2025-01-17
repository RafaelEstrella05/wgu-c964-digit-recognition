import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from PySide6.QtWidgets import QWidget, QLabel, QListWidget, QPushButton, QVBoxLayout, QMessageBox, QApplication
from keras.src.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

global model
global model_name
global x_test
global y_test

def extract_embeddings(x_data):
    global model

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

    plt.title("Scatterplot of Embeddings: " + model_name)
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.tight_layout()
    plt.show()

'''
This window is in charge of reading the contents of the /models directory and displaying the available models in a list 
for the user to select. The models are displayed in a dropdown list, in which the default option is to train a new model.
the name of the new model will be "mnist_model_v_X.keras" where X is the next available number in the sequence.
The "continue" button allows the user to continue with or without selecting a model.
'''
class ModelSelectionWindow(QWidget):
    global model_list

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Model Selection")
        self.setGeometry(100, 100, 800, 600)

        self.model_list_label = QLabel("Select a model to use:")
        self.model_list_label.setStyleSheet("font-size: 14px; font-weight: bold;")

        self.model_list_widget = QListWidget()
        self.model_list_widget.setStyleSheet("font-size: 12px;")
        self.model_list_widget.addItem("Select a model")

        for m in model_list:
            self.model_list_widget.addItem(m)

        self.continue_button = QPushButton("Continue")
        self.continue_button.clicked.connect(self.continue_button_clicked)
        self.continue_button.setStyleSheet("font-size: 14px; padding: 10px;")

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.model_list_label)
        self.layout.addWidget(self.model_list_widget)
        self.layout.addWidget(self.continue_button)

        self.setLayout(self.layout)
        self.show()
        self.model_selection_window = self

    def continue_button_clicked(self):
        # Get the selected model
        selected_model_name = self.model_list_widget.currentItem().text()
        print(selected_model_name)

        #verify that the extension is .keras
        if selected_model_name[-6:] != ".keras":
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setText(f"Please select a keras model to continue.")
            msg_box.exec()
            return

        load_keras_model(selected_model_name)

        self.model_selection_window.close()
        main()

def load_keras_model(model_file):
    global model
    global model_name
    global x_test
    global y_test

    # Load MNIST dataset and preprocess
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    #y_test = to_categorical(y_test, 10)
    y_test = y_test


    file_name = f"models/{model_file}"

    # load model
    model = load_model(file_name)

    # remove the models/ prefix from the model name
    model_name = model_file.split("/")[0]

def main():
    global model
    global model_name
    global x_test
    global y_test


    # Extract embeddings
    embeddings = extract_embeddings(x_test)

    # Reduce dimensions using PCA
    reduced_embeddings = reduce_dimensions(embeddings)

    # Plot scatterplot of embeddings
    plot_embeddings(reduced_embeddings, y_test)

if __name__ == "__main__":
    global model_list

    app = QApplication(sys.argv)

    # get list of models from /models directory
    model_list = [f for f in os.listdir("models") if f.endswith(".keras")]

    # if model_list empty, alert user they have to train a new model by running main.py
    if len(model_list) == 0:
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(f"No models found in the models directory. Please train a model first by running main.py")
        msg_box.exec()
        sys.exit()
    else:
        model_selection_window = ModelSelectionWindow()


    sys.exit(app.exec())